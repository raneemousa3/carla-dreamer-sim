"""Collect full car-following episodes with synchronized RGB frames and lane offsets."""

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


def build_lead_route(start):
    """Sample one unambiguous 30 m route ahead of the lead vehicle."""
    route = []
    for _ in range(15):
        ahead = start.next(2.0)
        if len(ahead) != 1:
            return []
        start = ahead[0]
        route.append(start.transform.location)
    return route


def horizontal_distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def collect_episode(world, road_map, traffic, args, rng):
    blueprints = world.get_blueprint_library()
    vehicle = blueprints.find("vehicle.tesla.model3")
    spawn_points = road_map.get_spawn_points()
    rng.shuffle(spawn_points)
    ego = None
    lead = None
    for spawn in spawn_points:
        start = road_map.get_waypoint(spawn.location, lane_type=carla.LaneType.Driving)
        if start is None or start.is_junction:
            continue
        ahead = start.next(args.lead_distance)
        if len(ahead) != 1:
            continue
        target = ahead[0]
        if target.is_junction or (target.road_id, target.lane_id) != (start.road_id, start.lane_id):
            continue
        lead_route = build_lead_route(target)
        if not lead_route:
            continue
        lead_spawn = target.transform
        lead_spawn.location.z += spawn.location.z - start.transform.location.z

        # Small initial displacements give the model examples on both sides.
        displacement = rng.uniform(-0.5, 0.5)
        right = spawn.get_right_vector()
        spawn.location.x += displacement * right.x
        spawn.location.y += displacement * right.y
        ego = world.try_spawn_actor(vehicle, spawn)
        if ego is None:
            continue
        lead = world.try_spawn_actor(vehicle, lead_spawn)
        if lead is not None:
            break
        ego.destroy()
        ego = None
    if ego is None:
        raise RuntimeError("Could not find a suitable spawn on the requested map.")

    camera = None
    collision_sensor = None
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
        collisions = Queue()
        collision_sensor = world.spawn_actor(
            blueprints.find("sensor.other.collision"), carla.Transform(), attach_to=ego
        )
        collision_sensor.listen(lambda event: collisions.put(event.frame))
        lead.set_autopilot(True, traffic.get_port())
        traffic.auto_lane_change(lead, False)
        traffic.set_path(lead, lead_route)
        ego.set_autopilot(True, traffic.get_port())
        traffic.auto_lane_change(ego, False)
        traffic.set_path(ego, [target.transform.location, *lead_route])
        route_points = [start.transform.location, target.transform.location, *lead_route]

        samples = {}
        pending = {}
        terminal_frame = None
        terminal_reason = None
        first_frame = None
        for step in range(1, args.max_steps + 31):
            frame = world.tick()
            if first_frame is None:
                first_frame = frame
            if terminal_frame is None:
                snapshot = world.get_snapshot()
                ego_state = snapshot.find(ego.id)
                lead_state = snapshot.find(lead.id)
                if snapshot.frame != frame or ego_state is None or lead_state is None:
                    raise RuntimeError(f"Missing vehicle pose for simulation frame {frame}.")
                ego_location = ego_state.get_transform().location
                lead_location = lead_state.get_transform().location
                offset = signed_lane_offset(ego_location, road_map)
                if not math.isfinite(offset):
                    raise RuntimeError(f"Invalid lane offset for frame {frame}.")
                gap = horizontal_distance(ego_location, lead_location)
                pending[frame] = (offset, gap)
                if horizontal_distance(lead_location, lead_route[-1]) <= 2.0:
                    terminal_reason = "lead_reached_goal"
                elif min(horizontal_distance(ego_location, point) for point in route_points) > 5:
                    terminal_reason = "ego_off_route"
                elif gap > 30:
                    terminal_reason = "lead_too_far"
                elif step == args.max_steps:
                    terminal_reason = "time_limit"
                if terminal_reason is not None:
                    terminal_frame = frame

            while True:
                try:
                    collision_frame = collisions.get_nowait()
                except Empty:
                    break
                if terminal_frame is None or collision_frame <= terminal_frame:
                    terminal_frame = collision_frame
                    terminal_reason = "ego_collision"

            # The camera can deliver earlier frames several ticks late.
            while True:
                try:
                    image = images.get(timeout=0.2) if terminal_frame is not None else images.get_nowait()
                except Empty:
                    break
                if image.frame not in pending:
                    continue
                pixels = np.frombuffer(image.raw_data, dtype=np.uint8)
                rgb = pixels.reshape(image.height, image.width, 4)[:, :, 2::-1]
                offset, gap = pending.pop(image.frame)
                if rgb.any():
                    samples[image.frame] = (rgb.copy(), offset, gap)
            if terminal_frame is not None and frame >= terminal_frame + 8:
                if not any(candidate <= terminal_frame for candidate in pending):
                    break
                if frame >= terminal_frame + 30:
                    break

        frame_ids = sorted(frame for frame in samples if frame <= terminal_frame)
        if not frame_ids:
            raise RuntimeError("The episode ended without any matched camera frames.")
        visuals, offsets, gaps = zip(*(samples[frame] for frame in frame_ids))
        return {
            "images": torch.from_numpy(np.stack(visuals)),
            "lane_offsets_m": torch.tensor(offsets, dtype=torch.float32),
            "lead_center_distance_m": torch.tensor(gaps, dtype=torch.float32),
            "frame_ids": torch.tensor(frame_ids, dtype=torch.int64),
            "terminal_reason": terminal_reason,
            "terminal_frame": terminal_frame,
            "missing_camera_frames": terminal_frame - first_frame + 1 - len(frame_ids),
        }
    finally:
        try:
            if camera is not None:
                camera.stop()
                camera.destroy()
        finally:
            try:
                if collision_sensor is not None:
                    collision_sensor.stop()
                    collision_sensor.destroy()
            finally:
                destroy_vehicles(ego, lead, traffic)


def destroy_vehicles(ego, lead, traffic):
    try:
        if ego.is_alive:
            ego.set_autopilot(False, traffic.get_port())
            ego.destroy()
    finally:
        if lead.is_alive:
            lead.set_autopilot(False, traffic.get_port())
            lead.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, required=True, help="New output directory")
    parser.add_argument("--n-traj", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=501, help="Time-limit ticks per episode")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--map", default="Town10HD_Opt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-of-day", choices=TIME_OF_DAY, default="day")
    parser.add_argument("--weather", choices=WEATHER_PRESETS, default="rainy")
    parser.add_argument("--lead-distance", type=float, default=8.0, help="Initial lead-car spacing along the lane, in meters")
    args = parser.parse_args()
    if (
        not math.isfinite(args.lead_distance) or args.lead_distance < 6
    ):
        parser.error("--lead-distance must be at least 6 meters and finite")
    if args.n_traj < 1 or args.max_steps < 1:
        parser.error("--n-traj and --max-steps must be positive")
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
                  f"{offsets.min():+.3f} to {offsets.max():+.3f} m | "
                  f"{data['terminal_reason']} | {data['missing_camera_frames']} missing frames", flush=True)
    finally:
        try:
            traffic.set_synchronous_mode(False)
        finally:
            world.apply_settings(original_settings)
            world.set_weather(original_weather)


if __name__ == "__main__":
    main()
