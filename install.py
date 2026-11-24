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
CORE_READY_MODES = ('normal', 'holding', 'waiting_hot', 'waiting_fans', 'waiting_cool',
                    'reduced', 'minimum', 'recovering', 'emergency', 'emergency_cooling')
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
                   if p.exists() or p.is_symlink()]
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
    core.gpu_fan_max(core.HIVE_FANS)
    for uuid in core_cfg.uuids:
        if (gpus[uuid]['temperature_c'] is None or gpus[uuid]['fan_percent'] is None or
                locks.get(uuid) is None or locks[uuid] < 500 or baselines.get(uuid) is None):
            raise ValueError(f'GPU temperatures, fan PWM and fixed-core lock are required: {uuid}')
        if baselines[uuid] < core_cfg.for_gpu(uuid).min_core_mhz:
            raise ValueError(f'HiveOS clock is below the configured core minimum: {uuid}')
    return cfg, dict(snapshot, core_config_hash=core_cfg.digest)


def healthy_status(status, pid, expected_hash, expected_gpus, *, is_core):
    if (not isinstance(status, dict) or type(status.get('updated_at')) not in (int, float) or
            type(status.get('controller_pid')) is not int or status['controller_pid'] != pid or
            pid <= 0 or not 0 <= time.time() - status['updated_at'] <= 15 or
            status.get('config_hash') != expected_hash or
            any(status.get(key) for key in ('error', 'errors', 'config_error'))):
        return False
    if is_core:
        rows = status.get('gpu')
        return (isinstance(rows, dict) and set(rows) == expected_gpus and
                all(isinstance(row, dict) and not row.get('error') and
                    row.get('mode') in CORE_READY_MODES
                    for row in rows.values()))
    readings = status.get('gpu_temperatures')
    return (isinstance(readings, dict) and set(readings) == expected_gpus and
            status.get('mode') in ('normal', 'emergency'))


def wait_ready(fan_hash, core_hash, uuids, timeout=30):
    deadline = time.monotonic() + timeout
    expected = set(uuids)
    while time.monotonic() < deadline:
        complete = []
        for index, (service, expected_hash) in enumerate(zip(SERVICE_NAMES, (fan_hash, core_hash))):
            runtime = Path('/run') / service.removesuffix('.service') / 'status.json'
            try:
                status = json.loads(core.bounded_read(runtime, 1048576))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                output = subprocess.check_output(
                    ['systemctl', 'show', service, '-p', 'ActiveState', '-p', 'MainPID'],
                    text=True, timeout=min(3, remaining))
                fields = dict(line.split('=', 1) for line in output.splitlines() if '=' in line)
                if fields.get('ActiveState') == 'active' and healthy_status(
                        status, int(fields.get('MainPID', 0)), expected_hash, expected,
                        is_core=index == 1):
                    complete.append(service)
            except (OSError, ValueError, TypeError, OverflowError, RecursionError,
                    subprocess.SubprocessError):
                continue
        if len(complete) == len(SERVICE_NAMES):
            return
        time.sleep(max(0, min(1, deadline - time.monotonic())))
    raise RuntimeError(f'No fresh matching status from both services within {timeout:g} seconds')


def publish_file(origin, destination, mode, created):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with origin.open('rb') as source:
                shutil.copyfileobj(source, stream)
            os.fchmod(stream.fileno(), mode)
            stream.flush()
            os.fsync(stream.fileno())
            identity = os.fstat(stream.fileno())
            # A hard link publishes the complete file atomically without clobbering an existing one.
            os.link(temporary, destination)
            created.append((destination, identity))
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            temporary.unlink(missing_ok=True)


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
    reload_attempted = False
    created = []
    try:
        for origin, (destination, mode) in staged.items():
            publish_file(origin, destination, mode, created)
        subprocess.run(['systemd-analyze', 'verify',
                        *[str(Path('/etc/systemd/system') / s) for s in SERVICE_NAMES]],
                       check=True, timeout=15)
        reload_attempted = True
        subprocess.run(['systemctl', 'daemon-reload'], check=True, timeout=15)
        # Enable only after both services passed startup verification.
        started = True
        for service in SERVICE_NAMES:
            subprocess.run(['systemctl', 'start', service], check=True, timeout=30)
        wait_ready(configure.control.config_hash(copied_cfg), copied_core.digest, copied_core.uuids)
        subprocess.run(['systemctl', 'enable', *SERVICE_NAMES], check=True, timeout=15)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt):
        cleanup_ok = True
        commands = []
        if started:
            # Leave the installed files available for diagnosis and failsafe operation.
            commands = (['systemctl', 'stop', *reversed(SERVICE_NAMES)],
                        ['systemctl', 'disable', *SERVICE_NAMES],
                        ['/usr/bin/python3', '/usr/local/sbin/cryptone-fan-control.py', '--failsafe'])
        else:
            for destination, identity in reversed(created):
                try:
                    if os.path.samestat(destination.lstat(), identity):
                        destination.unlink()
                except FileNotFoundError:
                    continue
                except OSError as cleanup_error:
                    cleanup_ok = False
                    print(f'Cleanup failed: {cleanup_error}', file=sys.stderr)
            if reload_attempted:
                commands = (['systemctl', 'daemon-reload'],)
        for command in commands:
            try:
                subprocess.run(command, check=True, timeout=40)
            except (OSError, subprocess.SubprocessError) as cleanup_error:
                cleanup_ok = False
                print(f'Cleanup failed: {cleanup_error}', file=sys.stderr)
        detail = 'Check service and fan state immediately.' if not cleanup_ok else 'Cleanup completed.'
        print(f'Installation failed. {detail} Inspect {backup} and journalctl.', file=sys.stderr)
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
        subprocess.run(['/usr/bin/python3', '-c', 'import tomllib'], check=True, timeout=5)
        with Path('/run/cryptone-installer.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            install(args.config_dir.resolve())
        return 0
    except KeyboardInterrupt:
        print('install: interrupted', file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'install: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
