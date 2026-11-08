import argparse
import carla
import numpy as np
import os
import gym
import car_dreamer
import torch
from gym.envs.registration import register
from pathlib import Path
from tqdm import trange


TIME_OF_DAY = {'day': 45.0, 'sunset': 5.0, 'night': -30.0}
WEATHER_PRESETS = {'sunny': 'ClearNoon', 'cloudy': 'CloudyNoon', 'rainy': 'MidRainyNoon'}


def make_weather(time_of_day, weather):
    if time_of_day not in TIME_OF_DAY:
        raise ValueError(f"Unknown time of day: {time_of_day}")
    if weather not in WEATHER_PRESETS:
        raise ValueError(f"Unknown weather: {weather}")
    preset = getattr(carla.WeatherParameters, WEATHER_PRESETS[weather])
    # Copy the preset so changing the sun never mutates CARLA's shared presets.
    params = carla.WeatherParameters()
    for name in (
        'cloudiness', 'precipitation', 'precipitation_deposits', 'wind_intensity',
        'fog_density', 'fog_distance', 'fog_falloff', 'wetness',
        'scattering_intensity', 'mie_scattering_scale', 'rayleigh_scattering_scale',
        'dust_storm',
    ):
        setattr(params, name, getattr(preset, name))
    params.sun_azimuth_angle = 0.0
    params.sun_altitude_angle = TIME_OF_DAY[time_of_day]
    if weather == 'rainy':
        params.wetness = 80.0
    return params


def save_atomic(data, path):
    path = Path(path)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary_path)
    os.replace(temporary_path, path)


def collect_carla_dataset(
    outdir="/storage1/sibai/Active/ihab/research_new/datasets_dino/carla",
    n_traj=5000,
    seed=42,
    display=True,
    time_of_day='day',
    weather='sunny',
    port=2000,
):
    weather_params = make_weather(time_of_day, weather)
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)
    obs_dir = Path(outdir) / "obses"
    obs_dir.mkdir(parents=True, exist_ok=True)
    actions_list = []
    states_list = []
    seq_lengths = []
    costs_list = []
    distances_list = []

    config = car_dreamer.load_task_configs('carla_follow')
    config = config.update({'env.display.enable': display, 'env.world.carla_port': port})
    register(
        id='CarlaFollowWrapperEnv-v0',
        entry_point='carla_wrapper:CarlaWrapper',
    )
    env = gym.make('CarlaFollowWrapperEnv-v0', config=config.env)
    # Apply after map loading; ordinary episode resets preserve this weather.
    env.unwrapped._world.carla_world.set_weather(weather_params)
    print(f"Collecting with time_of_day={time_of_day}, weather={weather}")

    for idx in trange(n_traj, desc="Collecting Carla Trajectories"):
        obs, full_state = env.reset()
        # Sensors are attached during reset and initially return a zero placeholder.
        for _ in range(20):
            if np.any(obs['visual']):
                break
            obs, _, done, _ = env.step(np.zeros(2, dtype=np.float32))
            full_state = env.state()
            if done:
                raise RuntimeError("Episode ended during camera warm-up; retry collection.")
        if not np.any(obs['visual']):
            raise RuntimeError("Camera produced no nonzero image after 20 warm-up steps.")
        visuals = []
        actions = []
        states = []
        costs = []
        distances = []
        done = False
        t = 0
        while not done:
            # Pair the pre-step observation/state with the action taken there.
            # Sensor arrays may reference buffers reused by later callbacks.
            visuals.append(obs['visual'].copy())
            states.append(full_state)
            action = env.compute_continuous_action() 
            obs, reward, done, _ = env.step(action)
            actions.append(action)
            full_state = env.state()
            cost = int(env.is_collision())  
            costs.append(cost)
            distances.append(env.get_distance())
            t += 1
        # Save trajectory visuals (images)
        visuals = np.stack(visuals).astype(np.uint8)  # [T, H, W, C]
        save_atomic(torch.from_numpy(visuals), obs_dir / f"episode_{idx:03d}.pth")
        # Save actions and states as tensors (not stacking across trajectories)
        actions_list.append(torch.tensor(np.stack(actions), dtype=torch.float32))
        states_list.append(torch.tensor(np.stack(states), dtype=torch.float32))
        seq_lengths.append(t)
        costs_list.append(torch.tensor(np.stack(costs), dtype=torch.float32))
        distances_list.append(torch.tensor(np.stack(distances), dtype=torch.float32))

        # Checkpoint after each episode, preserving the existing file formats.
        # Write lengths last as the record of completed episodes.
        save_atomic(actions_list, Path(outdir) / "actions.pth")
        save_atomic(states_list, Path(outdir) / "states.pth")
        save_atomic(costs_list, Path(outdir) / "costs.pth")
        save_atomic(distances_list, Path(outdir) / "distances.pth")
        save_atomic(torch.tensor(seq_lengths, dtype=torch.long), Path(outdir) / "seq_lengths.pth")
    print(f"Saved {n_traj} trajectories to {outdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CARLA following trajectories.")
    parser.add_argument('--outdir', default='./dataset')
    parser.add_argument('--n-traj', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--port', type=int, default=2000, help='CARLA server RPC port.')
    parser.add_argument('--headless', action='store_true', help='Disable the Flask observation monitor.')
    parser.add_argument('--time-of-day', choices=TIME_OF_DAY, default='day',
                        help='Fixed lighting condition (default: day).')
    parser.add_argument('--weather', choices=WEATHER_PRESETS, default='sunny',
                        help='Fixed weather condition (default: sunny).')
    args = parser.parse_args()
    collect_carla_dataset(outdir=args.outdir, n_traj=args.n_traj, seed=args.seed,
                          display=not args.headless, time_of_day=args.time_of_day,
                          weather=args.weather, port=args.port)
