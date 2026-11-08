"""Run two isolated Epic-quality CARLA servers and collectors."""
import argparse
import datetime
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import carla


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    # The launcher shell can exit before Unreal does; wait for the whole group.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-traj', type=int, default=3, help='Episodes per dataset (default: 3).')
    parser.add_argument('--outdir', type=Path, default=Path('dataset') / ('parallel_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')))
    parser.add_argument('--server', type=Path, default=Path('.local/carla-0.9.15/CarlaUE4.sh'))
    parser.add_argument('--base-port', type=int, default=2000)
    parser.add_argument('--single', action='store_true', help='Run only the first dataset/server.')
    parser.add_argument('--first-time-of-day', choices=['day', 'sunset', 'night'], default='night')
    parser.add_argument('--first-weather', choices=['sunny', 'cloudy', 'rainy'], default='sunny')
    parser.add_argument('--second-time-of-day', choices=['day', 'sunset', 'night'], default='day')
    parser.add_argument('--second-weather', choices=['sunny', 'cloudy', 'rainy'], default='rainy')
    args = parser.parse_args()
    if args.n_traj < 1:
        parser.error('--n-traj must be positive')
    server = args.server.resolve()
    if not server.is_file():
        parser.error(f'CARLA launcher not found: {server}')
    ports = [args.base_port, args.base_port + 10]
    if args.single:
        ports = ports[:1]
    # Reserve RPC, streaming, secondary, traffic-manager, and TM streaming ports.
    reserved = []
    try:
        for port in ports:
            for value in (port, port + 1, port + 2, port + 6000, port + 6001):
                sock = socket.socket()
                reserved.append(sock)
                try:
                    sock.bind(('0.0.0.0', value))
                except OSError as exc:
                    parser.error(
                        f'Port {value} is unavailable ({exc.strerror}). '
                        'Stop the existing CARLA server/collector before running this launcher, '
                        'which starts its own servers.'
                    )
    finally:
        for sock in reserved:
            sock.close()
    # Refuse existing directories rather than overwriting a previous collection.
    args.outdir.mkdir(parents=True, exist_ok=False)
    processes, logs = [], []
    try:
        for port in ports:
            log = open(args.outdir / f'server_{port}.log', 'w')
            logs.append(log)
            process = subprocess.Popen(
                [str(server), '-RenderOffScreen', '-quality-level=Epic', '-nosound',
                 '-ResX=256', '-ResY=256', '-fps=20',
                 f'-carla-rpc-port={port}', '-stdout', '-FullStdOutLogOutput'],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(process)
            print(f'Started Epic server on {port}, PID {process.pid}', flush=True)
            # Stagger startup to avoid simultaneous map-loading memory spikes.
            deadline = time.monotonic() + 120
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'Server {port} exited; see its log.')
                try:
                    # A client created before RPC startup may retain a failed connection.
                    client = carla.Client('127.0.0.1', port)
                    client.set_timeout(5)
                    version = client.get_server_version()
                    break
                except RuntimeError:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'Server {port} did not become ready within 120 seconds.')
                    time.sleep(1)
            print(f'Server {port} ready: {version}', flush=True)
        collectors = []
        conditions = [(args.first_time_of_day, args.first_weather),
                      (args.second_time_of_day, args.second_weather)]
        for index, (port, (daytime, weather)) in enumerate(zip(ports, conditions)):
            name = f'worker_{index}_{daytime}_{weather}'
            log = open(args.outdir / f'{name}.log', 'w')
            logs.append(log)
            process = subprocess.Popen(
                [sys.executable, '-B', '-u', str(Path(__file__).with_name('generate_dataset.py')),
                 '--headless', '--port', str(port), '--n-traj', str(args.n_traj),
                 '--seed', str(42 + index), '--time-of-day', daytime, '--weather', weather,
                 '--outdir', str(args.outdir / name)],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
            processes.append(process)
            collectors.append(process)
            print(f'Started {name} on {port}, PID {process.pid}', flush=True)
        while any(p.poll() is None for p in collectors):
            if any(p.poll() not in (None, 0) for p in processes):
                raise RuntimeError(f'A worker failed; inspect logs in {args.outdir}')
            time.sleep(1)
        if any(p.returncode != 0 for p in collectors):
            raise RuntimeError(f'Collection failed; inspect logs in {args.outdir}')
        print(f'All datasets completed: {args.outdir}', flush=True)
    finally:
        for process in reversed(processes):
            stop(process)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
