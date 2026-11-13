#!/usr/bin/env python3
"""Generate local configuration from a user-selected PWM and NVIDIA inventory.

This command reads hardware and writes a new output directory only. It never
changes fan speed, clock settings, installed configuration or services.
"""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import core_control as core

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('fan_control', HERE / 'cryptone-fan-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


def make_configs(selector, uuids, target=65, minimum=30, maximum=100, min_core=2400):
    settings = core.Settings.parse({'min_core_mhz': min_core})
    cfg = control.validate_config(dict(
        controller=dict(selector), expected_gpu_uuids=list(uuids),
        min_pwm_percent=minimum, max_pwm_percent=maximum, startup_pwm_percent=55,
        target_c=target, critical_c=75, recovery_c=max(70, target + 2),
        deadband_c=1, poll_seconds=5, rise_interval_seconds=10, fall_interval_seconds=20))
    text = ('# Created by configure.py. Frequencies MHz; temperatures C; times seconds.\n'
            '# Live reload: 5 seconds. UUID changes require stopping both services.\n'
            '# The frequency ceiling comes only from HiveOS; no max_core_mhz.\n\n[defaults]\n')
    text += ''.join(f'{key} = {value}\n' for key, value in asdict(settings).items())
    text += '\n[rig]\ngpu_uuids = ' + json.dumps(list(uuids)) + '\n'
    text += ('\n# Per-GPU overrides inherit [defaults], for example:\n'
             f'# [gpu."{uuids[0]}"]\n# min_core_mhz = {min_core}\n')
    return cfg, text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true', help='List PWM candidates without writes')
    parser.add_argument('--hwmon', type=Path, help='Selected /sys/class/hwmon/hwmonN directory')
    parser.add_argument('--pwm', type=int, help='Selected PWM channel number')
    parser.add_argument('--output', type=Path, default=Path('build'))
    parser.add_argument('--target', type=int, default=65)
    parser.add_argument('--min-pwm', type=int, default=30)
    parser.add_argument('--max-pwm', type=int, default=100)
    parser.add_argument('--min-core', type=int, default=2400,
                        help='Thermal reduction floor; choose for your GPU model')
    args = parser.parse_args(argv)
    if args.list:
        print(json.dumps(control.discover_controllers(), indent=2))
        return 0
    if args.hwmon is None or args.pwm is None:
        parser.error('Select --hwmon and --pwm after inspecting --list')
    try:
        candidates = [c for c in control.discover_controllers()
                      if Path(c['hwmon_path']) == args.hwmon and c['pwm_channel'] == args.pwm]
        if len(candidates) != 1:
            raise ValueError('Selected PWM candidate not found uniquely')
        selector = {k: candidates[0][k] for k in ('hwmon_name', 'device_path', 'pwm_channel')}
        result = subprocess.run(['/usr/bin/nvidia-smi', '--query-gpu=uuid',
                                 '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, check=True, timeout=10)
        uuids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        cfg, toml = make_configs(selector, uuids, args.target, args.min_pwm, args.max_pwm, args.min_core)
        # Refuse to overwrite previously generated or installed configurations.
        args.output.mkdir(parents=True, exist_ok=False)
        core.atomic_json(args.output / 'cryptone-fan-control.json', cfg)
        (args.output / 'cryptone-core-control.toml').write_text(toml)
        print(f'Generated {len(uuids)} GPU UUIDs and PWM{args.pwm} in {args.output.resolve()}')
        print('Hardware unchanged. Review both files before installation.')
        return 0
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'configure: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
