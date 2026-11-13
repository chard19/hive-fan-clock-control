#!/usr/bin/env python3
"""Install a fresh, reviewed configuration. Existing installations are refused."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import configure
import core_control as core

HERE = Path(__file__).resolve().parent
BACKUP_ROOT = Path('/var/backups')
SERVICE_NAMES = ('cryptone-fan-control.service', 'cryptone-core-control.service')
FILES = {
    'cryptone-fan-control.py': ('/usr/local/sbin/cryptone-fan-control.py', 0o755),
    'core_control.py': ('/usr/local/sbin/cryptone-core-control.py', 0o755),
    'fancheck': ('/usr/local/bin/fancheck', 0o755),
    'fanset': ('/usr/local/bin/fanset', 0o755),
    **{s: (f'/etc/systemd/system/{s}', 0o644) for s in SERVICE_NAMES},
    'cryptone-fan-control.json': ('/etc/cryptone-fan-control.json', 0o644),
    'cryptone-core-control.toml': ('/etc/cryptone-core-control.toml', 0o644),
}


def verify_pair(fan_cfg, core_cfg):
    if set(fan_cfg['expected_gpu_uuids']) != set(core_cfg.uuids):
        raise ValueError('Case and core configurations must have identical GPU inventories')


def preflight(config_dir):
    cfg = configure.control.load_config(config_dir / 'cryptone-fan-control.json')
    core_cfg = core.load_config(config_dir / 'cryptone-core-control.toml')
    verify_pair(cfg, core_cfg)
    collisions = [str(p) for p, _ in FILES.values() if Path(p).exists() or Path(p).is_symlink()]
    collisions += [str(p) for p in (Path('/var/lib/cryptone-core-control/state.json'),)
                   if p.exists()]
    if collisions:
        raise ValueError('Existing installation found; use the migration guide: ' + ', '.join(collisions))
    for service in SERVICE_NAMES:
        result = subprocess.run(['systemctl', 'is-active', service],
                                capture_output=True, text=True, check=False, timeout=5)
        if result.returncode == 0:
            raise ValueError(f'Service is already running: {service}')
    fan = configure.control.Fan(cfg)
    snapshot = fan.state()
    configure.control.read_temperatures(cfg)
    gpus = core.read_gpus()
    if set(gpus) != set(core_cfg.uuids):
        raise ValueError('GPU inventory changed after configuration generation')
    locks = core.read_locks(gpus)
    baselines = core.hive_baselines(gpus)
    # Read AutoFan settings without requiring our case daemon to exist yet.
    fans = core.shell_values(core.bounded_read(core.HIVE_FANS).decode(), {'ENABLED', 'MAX_FAN'})
    if fans.get('ENABLED') != '1' or not 1 <= int(fans.get('MAX_FAN', '0')) <= 100:
        raise ValueError('Enable HiveOS GPU AutoFan and set a valid maximum first')
    for uuid in core_cfg.uuids:
        if (gpus[uuid]['temperature_c'] is None or gpus[uuid]['fan_percent'] is None or
                locks.get(uuid) is None or locks[uuid] < 500 or baselines.get(uuid) is None):
            raise ValueError(f'GPU temperatures, fan PWM and fixed-core lock are required: {uuid}')
    return cfg, dict(snapshot, core_config_hash=core_cfg.digest)


def wait_ready(timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        complete = True
        for service in SERVICE_NAMES:
            runtime = Path('/run') / service.removesuffix('.service') / 'status.json'
            try:
                status = json.loads(core.bounded_read(runtime, 1048576))
                pid = int(subprocess.check_output(['systemctl', 'show', service,
                          '--property=MainPID', '--value'], text=True, timeout=5))
                active = subprocess.run(['systemctl', 'is-active', '--quiet', service],
                                        timeout=5, check=False).returncode == 0
                complete &= (active and pid > 0 and status['controller_pid'] == pid and
                             0 <= time.time() - status['updated_at'] <= 15 and
                             not status.get('config_error') and not status.get('error') and
                             not status.get('errors') and
                             not any(g.get('error') for g in status.get('gpu', {}).values()))
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError):
                complete = False
        if complete:
            return
        time.sleep(1)
    raise RuntimeError('No fresh error-free status from both services within 30 seconds')


def install(config_dir):
    cfg, baseline = preflight(config_dir)
    backup = Path(tempfile.mkdtemp(prefix='cryptone-install-', dir=BACKUP_ROOT))
    os.chmod(backup, 0o700)
    (backup / 'baseline.json').write_text(json.dumps(dict(
        captured_at=datetime.now(timezone.utc).isoformat(), controller=cfg['controller'],
        **baseline), indent=2) + '\n')
    staged = {}
    for source, (destination, mode) in FILES.items():
        origin = config_dir / source if source.endswith(('.json', '.toml')) else HERE / source
        copied = backup / source
        shutil.copyfile(origin, copied)
        staged[copied] = (Path(destination), mode)
    # Reject edits between the hardware preflight and the staged file snapshot.
    copied_cfg = configure.control.load_config(backup / 'cryptone-fan-control.json')
    copied_core = core.load_config(backup / 'cryptone-core-control.toml')
    verify_pair(copied_cfg, copied_core)
    if copied_cfg != cfg or (baseline.get('core_config_hash') and
                             copied_core.digest != baseline['core_config_hash']):
        raise ValueError('Configuration changed during preflight; installation was not started')
    print(f'Configuration and baseline backup: {backup}', flush=True)
    started = False
    try:
        for origin, (destination, mode) in staged.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive create prevents overwriting even if another installer races us.
            with destination.open('xb') as stream:
                stream.write(origin.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
            destination.chmod(mode)
        subprocess.run(['systemd-analyze', 'verify',
                        *[str(Path('/etc/systemd/system') / s) for s in SERVICE_NAMES]],
                       check=True, timeout=15)
        subprocess.run(['systemctl', 'daemon-reload'], check=True, timeout=15)
        # Enable only after both services passed startup verification.
        started = True
        for service in SERVICE_NAMES:
            subprocess.run(['systemctl', 'start', service], check=True, timeout=30)
        wait_ready()
        subprocess.run(['systemctl', 'enable', *SERVICE_NAMES], check=True, timeout=15)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError):
        # Leave files for inspection. Stopping the fan service requests full cooling.
        if started:
            for command in (['systemctl', 'disable', *SERVICE_NAMES],
                            ['systemctl', 'stop', *reversed(SERVICE_NAMES)],
                            ['/usr/bin/python3', '/usr/local/sbin/cryptone-fan-control.py', '--failsafe']):
                try:
                    subprocess.run(command, check=False, timeout=40)
                except (OSError, subprocess.SubprocessError) as cleanup_error:
                    print(f'Cleanup failed: {cleanup_error}', file=sys.stderr)
        print(f'Installation failed. Services stopped; inspect {backup} and journalctl.', file=sys.stderr)
        raise
    print('Installed and enabled. Run fancheck; use fanset to change the case-fan target.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_dir', type=Path, help='Reviewed directory generated by configure.py')
    args = parser.parse_args(argv)
    try:
        if os.geteuid() != 0 or sys.platform != 'linux':
            raise ValueError('Installation requires root on Linux with systemd and HiveOS')
        # Same environment required by HiveOS nvtool as in the service unit.
        os.environ['PATH'] = '/hive/sbin:/hive/bin:' + os.environ.get('PATH', '/usr/bin:/bin')
        if Path('/hive/lib').is_dir():
            os.environ['LD_LIBRARY_PATH'] = '/hive/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        for executable in ('systemctl', 'systemd-analyze', 'nvtool', 'nvidia-smi'):
            if shutil.which(executable) is None:
                raise ValueError(f'Required executable not found: {executable}')
        with Path('/run/cryptone-installer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            install(args.config_dir.resolve())
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'install: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
