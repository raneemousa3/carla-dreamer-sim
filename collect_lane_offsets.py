"""Collect synchronized RGB frames and signed """

import argparse
import math
import random
from pathlib import Path
from queue import Empty, Queue

import carla
import numpy as np
import torch

from generate_dataset import TIME_OF_DAY, WEATHER_PRESETS, make_weather, save_atomic


def signed_lane_offset(location, road_map):
    """Horizontal offset in meters, positive toward the lane's right side."""
    waypoint = road_map.get_waypoint(
        location, project_to_road=True, lane_type=carla.LaneType.Driving
    )
    if waypoint is None:
        raise RuntimeError("No driving lane found for the ego vehicle.")
    center = waypoint.transform.location
    right = waypoint.transform.get_right_vector()
    return ((location.x - center.x) * right.x
            + (location.y - center.y) * right.y) / math.hypot(right.x, right.y)


def collect_episode(world, road_map, traffic, args, rng):
    blueprints = world.get_blueprint_library()
    vehicle = blueprints.find("vehicle.tesla.model3")
    spawn_points = road_map.get_spawn_points()
    rng.shuffle(spawn_points)
    ego = None
    for spawn in spawn_points:
        # Small initial displacements give the model examples on both sides.
        displacement = rng.uniform(-0.5, 0.5)
        right = spawn.get_right_vector()
        spawn.location.x += displacement * right.x
        spawn.location.y += displacement * right.y
        ego = world.try_spawn_actor(vehicle, spawn)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("Could not spawn an ego vehicle on the requested map.")

    camera = None
    try:
        camera_bp = blueprints.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "224")
        camera_bp.set_attribute("image_size_y", "224")
        camera_bp.set_attribute("sensor_tick", "0.0")
        camera = world.spawn_actor(
            camera_bp, carla.Transform(carla.Location(x=1.5, z=1.6)),
            attach_to=ego, attachment_type=carla.AttachmentType.Rigid,
        )
        images = Queue()
        camera.listen(images.put)
        ego.set_autopilot(True, traffic.get_port())
        traffic.auto_lane_change(ego, False)

        visuals = []
        offsets = []
        frame_ids = []
        pending_offsets = {}
        warmup_steps = 5
        last_record_step = warmup_steps + args.frames
        # GPU camera callbacks can arrive several ticks after their world snapshot.
        for step in range(last_record_step + 30):
            frame = world.tick()
            if warmup_steps <= step < last_record_step:
                snapshot = world.get_snapshot()
                vehicle_snapshot = snapshot.find(ego.id)
                if snapshot.frame != frame or vehicle_snapshot is None:
                    raise RuntimeError(f"Missing ego pose for simulation frame {frame}.")
                location = vehicle_snapshot.get_transform().location
                offset = signed_lane_offset(location, road_map)
                if not math.isfinite(offset):
                    raise RuntimeError(f"Invalid lane offset for frame {frame}.")
                pending_offsets[frame] = offset

            while True:
                try:
                    image = images.get(timeout=0.2) if step >= last_record_step else images.get_nowait()
                except Empty:
                    break
                if image.frame not in pending_offsets:
                    continue
                pixels = np.frombuffer(image.raw_data, dtype=np.uint8)
                rgb = pixels.reshape(image.height, image.width, 4)[:, :, 2::-1]
                if not rgb.any():
                    raise RuntimeError(f"Camera frame {image.frame} is black. Check rendering.")
                visuals.append(rgb.copy())
                offsets.append(pending_offsets.pop(image.frame))
                frame_ids.append(image.frame)
                if len(frame_ids) == args.frames:
                    break
            if len(frame_ids) == args.frames:
                break
        if len(frame_ids) != args.frames or frame_ids != sorted(frame_ids):
            raise RuntimeError(
                f"Expected {args.frames} ordered camera frames, received {len(frame_ids)}."
            )

        return {
            "images": torch.from_numpy(np.stack(visuals)),
            "lane_offsets_m": torch.tensor(offsets, dtype=torch.float32),
            "frame_ids": torch.tensor(frame_ids, dtype=torch.int64),
        }
    finally:
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
        finally:
            ego.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory")
    parser.add_argument("--n-traj", type=int, default=3)
    parser.add_argument("--frames", type=int, default=100, help="Frames per episode")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--map", default="Town03")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-of-day", choices=TIME_OF_DAY, default="day")
    parser.add_argument("--weather", choices=WEATHER_PRESETS, default="sunny")
    args = parser.parse_args()
    if args.n_traj < 1 or args.frames < 1:
        parser.error("--n-traj and --frames must be positive")
    if args.outdir.exists():
        parser.error("--outdir already exists; choose a new directory")

    client = carla.Client("127.0.0.1", args.port)
    client.set_timeout(30.0)
    world = client.get_world()
    if world.get_settings().synchronous_mode:
        raise RuntimeError("Start a fresh CARLA server dedicated to this collector.")
    if world.get_map().name.split("/")[-1] != args.map:
        world = client.load_world(args.map)
    road_map = world.get_map()
    original_settings = world.get_settings()
    original_weather = world.get_weather()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.no_rendering_mode = False
    traffic = client.get_trafficmanager(args.tm_port)
    rng = random.Random(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=False)
    try:
        world.apply_settings(settings)
        traffic.set_synchronous_mode(True)
        traffic.set_random_device_seed(args.seed)
        world.set_weather(make_weather(args.time_of_day, args.weather))
        for episode in range(args.n_traj):
            data = collect_episode(world, road_map, traffic, args, rng)
            path = args.outdir / f"episode_{episode:03d}.pth"
            save_atomic(data, path)
            offsets = data["lane_offsets_m"]
            print(f"Saved {path} | {len(offsets)} frames | offsets "
                  f"{offsets.min():+.3f} to {offsets.max():+.3f} m", flush=True)
    finally:
        try:
            traffic.set_synchronous_mode(False)
        finally:
            world.apply_settings(original_settings)
            world.set_weather(original_weather)


if __name__ == "__main__":
    main()
