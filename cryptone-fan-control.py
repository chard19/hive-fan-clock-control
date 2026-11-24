#!/usr/bin/env python3
"""Control one explicitly selected Linux hwmon PWM from NVIDIA temperatures."""
import argparse
from contextlib import contextmanager
import csv
import errno
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time

CONFIG_PATH = Path('/etc/cryptone-fan-control.json')
HWMON_ROOT = Path('/sys/class/hwmon')
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
RUNTIME = Path('/run/cryptone-fan-control')
MAX_CONFIG_BYTES = 65536
MAX_STATUS_BYTES = 262144
LOG = logging.getLogger('cryptone-fans')


def finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def validate_config(cfg):
    if not isinstance(cfg, dict):
        raise ValueError('Configuration must be a JSON object')
    numeric = ('min_pwm_percent', 'max_pwm_percent', 'target_c', 'critical_c',
               'recovery_c', 'poll_seconds', 'rise_interval_seconds',
               'fall_interval_seconds', 'deadband_c')
    for key in numeric:
        value = cfg.get(key)
        if not finite_number(value):
            raise ValueError(f'Invalid numeric setting: {key}')
    for key in ('target_c', 'min_pwm_percent', 'max_pwm_percent'):
        if not isinstance(cfg[key], int):
            raise ValueError(f'{key} must be an integer')
    if not 30 <= cfg['min_pwm_percent'] < cfg['max_pwm_percent'] <= 100:
        raise ValueError('Invalid normal PWM bounds')
    if not 20 <= cfg['target_c'] <= 70:
        raise ValueError('target_c must be in [20, 70]')
    if cfg['critical_c'] != 75 or cfg['recovery_c'] != max(70, cfg['target_c'] + 2):
        raise ValueError('Safety thresholds must match the approved 75C emergency policy')
    if not 0 < cfg['target_c'] < cfg['recovery_c'] < cfg['critical_c'] <= 100:
        raise ValueError('Invalid temperature thresholds')
    if not 1 <= cfg['poll_seconds'] <= 10:
        raise ValueError('poll_seconds must be in [1, 10]')
    if min(cfg['rise_interval_seconds'], cfg['fall_interval_seconds']) < cfg['poll_seconds']:
        raise ValueError('Adjustment intervals must be at least one poll')
    if not 0 <= cfg['deadband_c'] <= 2:
        raise ValueError('Invalid deadband')
    startup = cfg.get('startup_pwm_percent', 55)
    if not finite_number(startup) or not 30 <= startup <= 100:
        raise ValueError('Invalid startup PWM')
    ids = cfg.get('expected_gpu_uuids')
    if (not isinstance(ids, list) or not ids or
            any(not isinstance(x, str) or not UUID_RE.fullmatch(x) for x in ids) or
            len(set(ids)) != len(ids)):
        raise ValueError('A nonempty list of distinct NVIDIA GPU UUIDs is required')
    validate_selector(cfg.get('controller'))
    try:
        json.dumps(cfg, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()
    except (RecursionError, TypeError) as exc:
        raise ValueError('Invalid configuration value') from exc
    return cfg


def validate_selector(selector):
    if not isinstance(selector, dict) or set(selector) != {'hwmon_name', 'device_path', 'pwm_channel'}:
        raise ValueError('controller requires hwmon_name, device_path and pwm_channel')
    name, device, channel = (selector[k] for k in ('hwmon_name', 'device_path', 'pwm_channel'))
    if not isinstance(name, str) or not re.fullmatch(r'[\w.-]+', name):
        raise ValueError('Invalid hwmon chip name')
    if (not isinstance(device, str) or not device.startswith('/sys/devices/') or
            any(p in ('.', '..', 'hwmon') for p in device.split('/')) or
            str(Path(device)) != device):
        raise ValueError('device_path must be a stable absolute /sys/devices path')
    if type(channel) is not int or channel < 1:
        raise ValueError('pwm_channel must be a positive integer')
    return selector


def device_identity(hwmon):
    device = hwmon / 'device'
    if device.exists():
        return str(device.resolve())
    resolved = hwmon.resolve()
    if resolved.parent.name == 'hwmon':
        return str(resolved.parent.parent)
    raise ValueError('Stable hwmon device identity is unavailable')


def discover_controllers(root=None):
    """Read-only inventory. Never probe a channel by changing its duty."""
    result = []
    for hwmon in sorted(Path(root or HWMON_ROOT).glob('hwmon*')):
        try:
            name = (hwmon / 'name').read_text().strip()
            device = device_identity(hwmon)
            for pwm in sorted(hwmon.glob('pwm[0-9]*')):
                match = re.fullmatch(r'pwm([1-9][0-9]*)', pwm.name)
                if match and (hwmon / f'{pwm.name}_enable').is_file():
                    result.append(dict(hwmon_name=name, device_path=device,
                                       pwm_channel=int(match[1]), hwmon_path=str(hwmon)))
        except (OSError, ValueError):
            continue
    return result


def assert_live_change(old, new):
    if (old['controller'] != new['controller'] or
            set(old['expected_gpu_uuids']) != set(new['expected_gpu_uuids'])):
        raise ValueError('PWM hardware or GPU inventory changes require stopping both services')


def load_config(path):
    with Path(path).open('rb') as stream:
        data = stream.read(MAX_CONFIG_BYTES + 1)
    if len(data) > MAX_CONFIG_BYTES:
        raise ValueError('Configuration is too large')
    try:
        return validate_config(json.loads(data, object_pairs_hook=unique_fields))
    except RecursionError as exc:
        raise ValueError('Configuration is nested too deeply') from exc


def unique_fields(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate configuration field: {key}')
        result[key] = value
    return result


def config_hash(cfg):
    data = json.dumps(cfg, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def parse_temperatures(output, expected):
    readings = {}
    for row in csv.reader(output.splitlines()):
        if len(row) != 2:
            raise ValueError('Unexpected nvidia-smi output')
        uuid, temperature = (item.strip() for item in row)
        if uuid in readings:
            raise ValueError('Duplicate GPU reading')
        value = int(temperature)
        if not 0 <= value <= 120:
            raise ValueError('GPU temperature outside plausible range')
        readings[uuid] = value
    if set(readings) != set(expected):
        raise ValueError('Expected GPU set does not match available readings')
    return readings


def read_temperatures(cfg):
    result = subprocess.run(
        ['/usr/bin/nvidia-smi', '--query-gpu=uuid,temperature.gpu',
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True, timeout=4)
    return parse_temperatures(result.stdout, cfg['expected_gpu_uuids'])


def pwm_value(percent, cfg=None):
    if not finite_number(percent) or not 0 < percent <= 100:
        raise ValueError('Invalid PWM percentage')
    value = round(percent * 255 / 100)
    if cfg is not None and percent != 100:
        lower = math.ceil(cfg['min_pwm_percent'] * 255 / 100)
        upper = math.floor(cfg['max_pwm_percent'] * 255 / 100)
        value = max(lower, min(upper, value))
    return max(1, min(255, value))


class Fan:
    def __init__(self, cfg, root=None):
        self.selector = dict(validate_selector(cfg['controller']))
        candidates = [c for c in discover_controllers(root)
                      if all(c[k] == v for k, v in self.selector.items())]
        if len(candidates) != 1:
            raise RuntimeError('Configured hwmon PWM channel was not found uniquely')
        self.path = Path(candidates[0]['hwmon_path'])
        # Pin the selected sysfs node. A hwmon class symlink can be re-used
        # for a different device while this process is still running.
        self.node = self.path.resolve(strict=True)
        self.pwm = self.node / f'pwm{self.selector["pwm_channel"]}'
        self.enable = self.node / f'{self.pwm.name}_enable'

    def verify_identity(self):
        if ((self.node / 'name').read_text().strip() != self.selector['hwmon_name'] or
                device_identity(self.node) != self.selector['device_path']):
            raise RuntimeError('PWM device identity changed; refusing to access another controller')

    def state(self):
        self.verify_identity()
        raw = int(self.pwm.read_text())
        mode = int(self.enable.read_text())
        if not 0 <= raw <= 255 or mode < 0:
            raise RuntimeError('Invalid PWM hardware reading')
        return {'pwm_raw': raw, 'pwm_mode': mode, 'hwmon_path': str(self.path)}

    def set(self, percent, cfg=None):
        value = pwm_value(percent, cfg)
        self.verify_identity()
        # it87 may report mode 0 for the manual full-speed value 255.
        # Preload full duty before taking manual control, avoiding an old low duty.
        if int(self.enable.read_text()) != 1:
            try:
                self.pwm.write_text('255\n')
            except OSError as exc:
                if exc.errno != errno.EBUSY:
                    raise
                # NCT6775 rejects duty writes in automatic modes. Disable
                # controller regulation at full speed before going manual.
                self.enable.write_text('0\n')
            self.enable.write_text('1\n')
        self.pwm.write_text(f'{value}\n')
        state = self.state()
        if state['pwm_mode'] == 0 and value == 255:
            return state
        if state['pwm_mode'] != 1 or abs(state['pwm_raw'] - value) > 1:
            raise RuntimeError(f'PWM write did not persist: {state}')
        return state


class Controller:
    def __init__(self, cfg):
        self.cfg = cfg
        self.current = max(cfg['min_pwm_percent'],
                           min(cfg['max_pwm_percent'], cfg.get('startup_pwm_percent', 55)))
        self.emergency = False
        self.healthy = 0
        self.last_adjust = None

    def reconfigure(self, cfg):
        if cfg['recovery_c'] != self.cfg['recovery_c']:
            # Samples taken under a different recovery threshold cannot count.
            self.healthy = 0
        self.cfg = cfg
        self.current = max(cfg['min_pwm_percent'], min(cfg['max_pwm_percent'], self.current))

    def update(self, temperature, now):
        cfg = self.cfg
        if temperature is None or temperature >= cfg['critical_c']:
            self.emergency, self.healthy = True, 0
            return 100
        if self.emergency:
            self.healthy = self.healthy + 1 if temperature <= cfg['recovery_c'] else 0
            if self.healthy < 3:
                return 100
            self.emergency = False
            self.current = cfg['max_pwm_percent']
            self.last_adjust = now
            return self.current
        if self.last_adjust is None:
            self.last_adjust = now
        error = temperature - cfg['target_c']
        elapsed = now - self.last_adjust
        if error > cfg['deadband_c'] and elapsed >= cfg['rise_interval_seconds']:
            self.current += min(3, max(1, error - cfg['deadband_c']))
            self.last_adjust = now
        elif error < -cfg['deadband_c'] and elapsed >= cfg['fall_interval_seconds']:
            self.current -= 1
            self.last_adjust = now
        self.current = max(cfg['min_pwm_percent'], min(cfg['max_pwm_percent'], self.current))
        return self.current


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if not address:
        return
    if address.startswith('@'):
        address = '\0' + address[1:]
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
        sock.sendto(message.encode(), address)


def write_status(status):
    temporary = RUNTIME / 'status.tmp'
    temporary.write_text(json.dumps(status, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(RUNTIME / 'status.json')


@contextmanager
def controller_lock():
    RUNTIME.mkdir(exist_ok=True)
    with (RUNTIME / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def previous_emergency():
    try:
        with (RUNTIME / 'status.json').open('rb') as stream:
            raw = stream.read(MAX_STATUS_BYTES + 1)
        if len(raw) <= MAX_STATUS_BYTES:
            status = json.loads(raw)
            if isinstance(status, dict):
                return status.get('mode') == 'emergency' or status.get('emergency_latched') is True
    except (OSError, ValueError, RecursionError):
        pass
    return False


def apply_failsafe(fan):
    state = fan.set(100)
    write_status(dict(state, updated_at=time.time(), mode='failsafe',
                      pwm_requested_percent=100, pwm_actual_percent=100,
                      emergency_latched=previous_emergency(),
                      error='Controller stopped or failed; full cooling applied'))
    return state


def run(fan, cfg, config_path=None):
    with controller_lock():
        run_locked(fan, cfg, config_path)


def run_locked(fan, cfg, config_path):
    binding = RUNTIME / 'hardware.json'
    temporary = binding.with_suffix('.tmp')
    temporary.write_text(json.dumps({'controller': cfg['controller']}) + '\n')
    temporary.replace(binding)
    stopping = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopping.set())
    control = Controller(cfg)
    control.emergency = previous_emergency()
    first_sample = True
    last_summary, last_logged = 0, None
    try:
        fan.set(100)
        notify('READY=1\nSTATUS=Starting GPU temperature sampling')
        while not stopping.is_set():
            started = time.monotonic()
            readings, error, config_error = {}, None, None
            if config_path is not None:
                try:
                    candidate = load_config(config_path)
                    if candidate != cfg:
                        assert_live_change(cfg, candidate)
                        control.reconfigure(candidate)
                        cfg = candidate
                except (OSError, ValueError, UnicodeError) as exc:
                    config_error = f'{type(exc).__name__}: {str(exc)[:200]}'
            try:
                readings = read_temperatures(cfg)
                hottest = max(readings.values())
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                hottest = None
                error = f'{type(exc).__name__}: {str(exc)[:200]}'
            if config_error:
                error = f'Invalid live configuration: {config_error}'
            if first_sample and hottest is not None and hottest > cfg['recovery_c']:
                control.emergency = True
            first_sample = False
            requested = control.update(None if config_error else hottest, started)
            state = fan.set(requested, cfg)
            mode = 'emergency' if control.emergency else 'normal'
            limited = (mode == 'normal' and requested >= cfg['max_pwm_percent'] and
                       hottest > cfg['target_c'] + cfg['deadband_c'])
            status = dict(state, updated_at=time.time(), mode=mode,
                          gpu_temperatures=readings, hottest_c=hottest,
                          target_c=cfg['target_c'], pwm_requested_percent=requested,
                          pwm_actual_percent=100 if state['pwm_mode'] == 0 else
                          round(state['pwm_raw'] * 100 / 255, 1),
                          target_unmet_at_limit=limited, error=error,
                          config_error=config_error, applied_config=cfg,
                          config_hash=config_hash(cfg), controller_pid=os.getpid())
            write_status(status)
            summary = (mode, requested, limited, bool(error), config_hash(cfg))
            if summary != last_logged or started - last_summary >= 60:
                LOG.info('mode=%s hottest=%sC PWM=%.1f%% target=%sC limited=%s%s',
                         mode, hottest, status['pwm_actual_percent'], cfg['target_c'],
                         limited, f' error={error}' if error else '')
                last_logged, last_summary = summary, started
            notify(f'WATCHDOG=1\nSTATUS={mode}; hottest={hottest}C; '
                   f'PWM={status["pwm_actual_percent"]}%; target={cfg["target_c"]}C')
            stopping.wait(max(0, cfg['poll_seconds'] - (time.monotonic() - started)))
    finally:
        fan.set(100)
        LOG.info('Controller exiting; selected PWM set to 100%')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG_PATH)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true', help='Read-only configuration/hardware check')
    modes.add_argument('--discover', action='store_true', help='List hwmon PWM channels; no writes')
    modes.add_argument('--failsafe', action='store_true', help='Set selected PWM to full speed')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    if args.discover:
        print(json.dumps(discover_controllers(), indent=2))
        return
    if args.failsafe:
        # Never follow an unaccepted live edit to a different physical channel.
        with controller_lock():
            binding = RUNTIME / 'hardware.json'
            selected = json.loads(binding.read_text()) if binding.exists() else load_config(args.config)
            print(json.dumps(apply_failsafe(Fan(selected))))
        return
    cfg = load_config(args.config)
    fan = Fan(cfg)
    if args.check:
        print(json.dumps({'fan': fan.state(), 'config': cfg,
                          'gpu_temperatures': read_temperatures(cfg)}, indent=2))
        return
    run(fan, cfg, args.config)


if __name__ == '__main__':
    main()
