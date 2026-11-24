#!/usr/bin/env python3
"""Per-UUID thermal core-clock guard for NVIDIA GPUs on HiveOS.

Only the graphics-clock lock is written. HiveOC files, offsets, memory, power,
GPU/case fans and the miner are never modified by this daemon.
"""
import argparse
import csv
from dataclasses import asdict, dataclass, fields
import fcntl
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import threading
import time
import tomllib

CONFIG = Path('/etc/cryptone-core-control.toml')
STATE = Path('/var/lib/cryptone-core-control/state.json')
RUNTIME = Path('/run/cryptone-core-control')
HIVE_OC = Path('/hive-config/nvidia-oc.conf')
HIVE_DETECT = Path('/run/hive/gpu-detect.json')
HIVE_BUS = Path('/hive-config/busid.conf')
HIVE_FANS = Path('/hive-config/autofan.conf')
CASE_STATUS = Path('/run/cryptone-fan-control/status.json')
POLL = 5
MAX_SAMPLE_GAP_SECONDS = POLL * 3
RESERVE_TEMP_C = 70
RESERVE_HOLD_SECONDS = 60
RESERVE_STEP_MHZ = 50
FAN_TOLERANCE_PERCENT = 2
CLOCK_TOLERANCE = 15  # NVIDIA chooses the nearest supported clock.
UUID_RE = re.compile(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z')
LOG = logging.getLogger('cryptone-core')


@dataclass(frozen=True)
class Settings:
    min_core_mhz: int = 2400
    step_mhz: int = 50
    hot_c: int = 68
    cool_c: int = 65
    hot_hold_seconds: int = 180
    cool_hold_seconds: int = 180
    emergency_c: int = 75

    @classmethod
    def parse(cls, data):
        if isinstance(data, dict) and 'max_core_mhz' in data:
            raise ValueError('Remove max_core_mhz: the maximum comes only from HiveOS')
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown settings or invalid settings table')
        s = cls(**data)
        if any(type(v) is not int for v in asdict(s).values()):
            raise ValueError('All settings must be integers')
        if not 500 <= s.min_core_mhz <= 5000:
            raise ValueError('Minimum core clock must be 500..5000 MHz')
        if not 1 <= s.step_mhz <= 300:
            raise ValueError('Invalid frequency step')
        if not 20 <= s.cool_c < s.hot_c < s.emergency_c <= 85:
            raise ValueError('Require 20 <= cool < hot < emergency <= 85 C')
        if not all(POLL <= v <= 3600 for v in
                   (s.hot_hold_seconds, s.cool_hold_seconds)):
            raise ValueError('Hold times must be 5..3600 seconds')
        return s


@dataclass(frozen=True)
class Config:
    uuids: tuple
    defaults: Settings
    overrides: dict
    digest: str

    def for_gpu(self, uuid):
        return self.overrides.get(uuid, self.defaults)


def bounded_read(path, limit=65536):
    with Path(path).open('rb') as f:
        data = f.read(limit + 1)
    if len(data) > limit:
        raise ValueError(f'{path}: file too large')
    return data


def load_config(path=CONFIG):
    raw = bounded_read(path)
    try:
        data = tomllib.loads(raw.decode())
    except RecursionError as exc:
        raise ValueError('Core configuration nesting is too deep') from exc
    if set(data) - {'rig', 'defaults', 'gpu'}:
        raise ValueError('Unknown configuration section')
    rig = data.get('rig', {})
    if not isinstance(rig, dict) or set(rig) != {'gpu_uuids'}:
        raise ValueError('[rig] must contain gpu_uuids')
    uuids = rig['gpu_uuids']
    if (not isinstance(uuids, list) or not uuids or
            any(not isinstance(u, str) or not UUID_RE.fullmatch(u) for u in uuids) or
            len(set(uuids)) != len(uuids)):
        raise ValueError('A nonempty list of distinct NVIDIA GPU UUIDs is required')
    defaults = Settings.parse(data.get('defaults', {}))
    override_data = data.get('gpu', {})
    if not isinstance(override_data, dict) or set(override_data) - set(uuids):
        raise ValueError('Override for an unknown GPU UUID')
    overrides = {}
    for uuid, values in override_data.items():
        if not isinstance(values, dict):
            raise ValueError('GPU override must be a table')
        overrides[uuid] = Settings.parse(asdict(defaults) | values)
    return Config(tuple(uuids), defaults, overrides, hashlib.sha256(raw).hexdigest())


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as f:
            name = f.name
            json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.chmod(name, 0o644)
        os.replace(name, path)
        name = None
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if name is not None:
            Path(name).unlink(missing_ok=True)


def shell_values(text, wanted):
    """Parse only literal scalar assignments; never source/eval shell code."""
    result = {}
    for line in text.splitlines():
        m = re.match(r'^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)=(.*)$', line)
        if not m or m[1] not in wanted:
            continue
        value = m[2]
        if any(char in value for char in '$`;|&<>'):
            raise ValueError(f'Nonliteral Hive setting {m[1]}')
        parts = shlex.split(value, comments=True)
        if len(parts) > 1:
            raise ValueError(f'Invalid Hive assignment {m[1]}')
        result[m[1]] = parts[0] if parts else ''
    return result


def bus_key(value):
    m = re.fullmatch(r'(?:(?:([0-9a-fA-F]{4}|[0-9a-fA-F]{8})):)?'
                     r'([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
    if not m:
        raise ValueError(f'Invalid PCI bus address: {value}')
    return f'{int(m[1] or "0", 16):04x}:{m[2].lower()}:{m[3].lower()}.{m[4]}'


def command(args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=4)
    if result.returncode:
        raise RuntimeError(f'{Path(args[0]).name}: exit {result.returncode}: '
                           f'{result.stderr.strip()[:160]}')
    return result.stdout


def optional_number(text, low, high):
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) and low <= number <= high else None


def parse_gpus(output):
    gpus = {}
    indices = set()
    buses = set()
    for row in csv.reader(output.splitlines()):
        if len(row) != 6:
            raise ValueError('Unexpected nvidia-smi output')
        index, uuid, bus, temp, clock, fan = (x.strip() for x in row)
        index = int(index)
        bus = bus_key(bus)
        if (index < 0 or not UUID_RE.fullmatch(uuid) or uuid in gpus or
                index in indices or bus in buses):
            raise ValueError('Duplicate or invalid GPU identity')
        indices.add(index)
        buses.add(bus)
        gpus[uuid] = dict(index=index, bus=bus,
                          temperature_c=optional_number(temp, 0, 120),
                          actual_core_mhz=optional_number(clock, 0, 10000),
                          fan_percent=optional_number(fan, 0, 100))
    if not gpus:
        raise ValueError('No GPUs returned')
    return gpus


def read_gpus():
    return parse_gpus(command([
        '/usr/bin/nvidia-smi',
        '--query-gpu=index,uuid,pci.bus_id,temperature.gpu,clocks.current.graphics,fan.speed',
        '--format=csv,noheader,nounits']))


def parse_locks(output):
    locks = {}
    index = None
    for line in output.splitlines():
        m = re.fullmatch(r'\s*DEVICE #(\d+):\s*', line)
        if m:
            index = int(m[1])
            if index in locks:
                raise ValueError('Duplicate nvtool device')
            locks[index] = None
        m = re.fullmatch(r'\s*GPU CLOCKS LOCKED:\s*(\d+) MHz\s*', line)
        if m and index is not None:
            if locks[index] is not None:
                raise ValueError('Duplicate nvtool clock readback')
            locks[index] = int(m[1])
    if not locks:
        raise ValueError('No device lock information from nvtool')
    return locks


def read_locks(gpus):
    """Validate the index-to-UUID mapping around nvtool's index-only readback."""
    locks = parse_locks(command([shutil.which('nvtool') or '/hive/sbin/nvtool',
                                 '-q', '--clocksinfo']))
    after = read_gpus()
    if {u: (g['index'], g['bus']) for u, g in gpus.items()} != {
            u: (g['index'], g['bus']) for u, g in after.items()}:
        raise ValueError('GPU mapping changed during clock readback')
    return {u: locks.get(g['index']) for u, g in gpus.items()}


def array_value(values, index):
    parts = values.split()
    return int(parts[min(index, len(parts) - 1)]) if parts else 0


def hive_baselines(gpus, oc_path=HIVE_OC, detect_path=HIVE_DETECT, bus_path=HIVE_BUS):
    oc = shell_values(bounded_read(oc_path).decode(), {'CLOCK', 'LCLOCK', 'OHGODAPILL_ARGS',
                                                        'OHGODAPILL_ENABLED',
                                                        'OHGODAPILL_START_TIMEOUT'})
    if Path(bus_path).exists():
        bus_text = bounded_read(bus_path).decode()
        if bus_text.strip() and any(not x.lstrip().startswith('#') and x.strip()
                                    for x in bus_text.splitlines()):
            raise ValueError('Nonempty busid.conf requires supported bus overrides')
    devices = json.loads(bounded_read(detect_path, 1048576))
    if not isinstance(devices, list) or any(not isinstance(d, dict) for d in devices):
        raise ValueError('Invalid Hive GPU detection data')
    order = [bus_key(d['busid']) for d in devices
             if d.get('brand') == 'nvidia' or d.get('vendor') == 'NVIDIA']
    if len(set(order)) != len(order):
        raise ValueError('Duplicate Hive PCI address')
    positions = {bus: i for i, bus in enumerate(order)}
    result = {}
    for uuid, gpu in gpus.items():
        bus = bus_key(gpu['bus'])
        if bus not in positions:
            raise ValueError(f'GPU not in Hive mapping: {uuid}')
        i = positions[bus]
        core = array_value(oc.get('CLOCK', ''), i)
        lock = array_value(oc.get('LCLOCK', ''), i)
        if core >= 500:
            extra = oc.get('OHGODAPILL_ARGS', '')
            offset = (int(extra) if not oc.get('OHGODAPILL_ENABLED') and
                      not oc.get('OHGODAPILL_START_TIMEOUT') and
                      re.fullmatch(r'-?\d+', extra) else 0)
            lock = core + offset
        if lock > 5000:
            raise ValueError(f'Hive lock for {uuid} exceeds the supported 5000 MHz')
        result[uuid] = lock if lock >= 500 else None
    return result


def case_cooling(case_path, now):
    case = json.loads(bounded_read(case_path))
    if not isinstance(case, dict) or type(case.get('updated_at')) not in (int, float):
        raise ValueError('Invalid case-fan timestamp')
    age = now - case['updated_at']
    if not 0 <= age <= 20:
        raise ValueError('Case-fan status is stale')
    maximum = case['applied_config']['max_pwm_percent']
    pwm = case['pwm_actual_percent']
    if not (type(maximum) in (int, float) and type(pwm) in (int, float) and
            0 < maximum <= 100 and 0 <= pwm <= 100):
        raise ValueError('Invalid case PWM limits')
    if 'pwm_mode' in case and (
            type(case['pwm_mode']) is not int or case['pwm_mode'] not in (0, 1)
            or (case['pwm_mode'] == 0 and pwm < 98)):
        raise ValueError('Case PWM is not confirmed in manual or full-speed mode')
    case_error = case.get('error') or case.get('config_error')
    if case_error and not (case.get('mode') == 'emergency' and pwm >= 98):
        raise ValueError('Case-fan error without confirmed emergency PWM')
    return dict(case_pwm=pwm, case_max=maximum,
                case_at_max=pwm >= maximum - FAN_TOLERANCE_PERCENT, case_error=case_error)


def gpu_fan_max(fan_path):
    fans = shell_values(bounded_read(fan_path).decode(), {'ENABLED', 'MAX_FAN', 'CUSTOM_MODE'})
    if fans.get('CUSTOM_MODE', '') not in ('', '0'):
        raise ValueError('Hive AutoFan custom limits cannot be mapped reliably to GPUs')
    gpu_max = int(fans.get('MAX_FAN', '0'))
    if fans.get('ENABLED') != '1' or not 1 <= gpu_max <= 100:
        raise ValueError('Hive AutoFan must have a valid enabled maximum')
    return gpu_max


def cooling_limits(case_path=CASE_STATUS, fan_path=HIVE_FANS, wall_time=None):
    now = time.time() if wall_time is None else wall_time
    try:
        limits = case_cooling(case_path, now)
    except (OSError, ValueError, TypeError, KeyError, OverflowError, RecursionError) as exc:
        limits = dict(case_pwm=None, case_max=None, case_at_max=False, case_error=str(exc)[:180])
    try:
        gpu_max = gpu_fan_max(fan_path)
    except (OSError, ValueError) as exc:
        return dict(limits, gpu_max=None, gpu_fan_threshold_percent=None, gpu_error=str(exc)[:180])
    return dict(limits, gpu_max=gpu_max, gpu_fan_threshold_percent=gpu_max - FAN_TOLERANCE_PERCENT,
                gpu_error=None)


@dataclass
class GPUState:
    current: int | None = None  # requested restriction, not the workload clock
    baseline: int | None = None
    emergency: bool = False
    hot_since: float | None = None
    reserve_since: float | None = None
    cool_since: float | None = None
    signature: tuple | None = None
    case_max: float | None = None
    gpu_max: int | None = None
    reduction_rule: str | None = None
    pending_reduction: int | None = None

    def persistent(self):
        return dict(current=self.current, baseline=self.baseline, emergency=self.emergency)

    def reset_timers(self):
        self.hot_since = self.reserve_since = self.cool_since = None
        self.reduction_rule = None


def valid_lock(value):
    return type(value) is int and 500 <= value <= 10000


def decide(state, settings, baseline, locked, temperature, cooling_at_max, now,
           allow_growth=True, *, case_at_max=False, case_max=None, gpu_max=None,
           actual_core=None):
    """Return (desired lock or None, mode, wait seconds). No hardware I/O."""
    state.reduction_rule = None
    signature = (baseline, settings)
    if state.signature != signature:
        state.reset_timers()
        state.signature = signature
    if state.case_max != case_max:
        state.hot_since = state.reserve_since = None
        state.case_max = case_max
    if state.gpu_max != gpu_max:
        state.hot_since = None
        state.gpu_max = gpu_max
    state.baseline = baseline
    if baseline is None or baseline < settings.min_core_mhz:
        state.reset_timers()
        return None, 'paused_hive', None
    if not valid_lock(locked):
        state.reset_timers()
        if temperature is not None and temperature >= settings.emergency_c:
            state.emergency = True
            state.current = min(settings.min_core_mhz, state.current or settings.min_core_mhz)
        if (state.emergency and state.current is not None and actual_core is not None and
                actual_core > state.current + CLOCK_TOLERANCE):
            # The workload clock permits a downward emergency attempt, never a recovery.
            return state.current, 'emergency', None
        return None, 'sensor_error', None
    ceiling = baseline
    if state.current is None:
        state.current = min(locked, ceiling)
    elif locked < state.current - CLOCK_TOLERANCE:
        # Respect externally reduced clocks; never immediately raise them.
        state.current = locked
        state.reset_timers()
    state.current = min(state.current, ceiling)
    if temperature is None:
        state.reset_timers()
        return state.current, 'sensor_error', None
    if temperature >= settings.emergency_c:
        state.emergency = True
        state.reset_timers()
        state.current = min(state.current, settings.min_core_mhz)
        return state.current, 'emergency', None
    if state.emergency:
        state.hot_since = state.reserve_since = None
        if temperature <= settings.cool_c and allow_growth:
            if state.cool_since is None:
                state.cool_since = now
            wait = max(0, settings.cool_hold_seconds - (now - state.cool_since))
            if wait == 0:
                state.emergency = False
                state.current = min(ceiling, state.current + settings.step_mhz)
                state.cool_since = now
                return state.current, 'recovering', settings.cool_hold_seconds
            return state.current, 'emergency_cooling', math.ceil(wait)
        state.cool_since = None
        return state.current, 'emergency', None
    regular_hot = temperature >= settings.hot_c and cooling_at_max
    reserve_hot = temperature >= RESERVE_TEMP_C and case_at_max
    if regular_hot or reserve_hot:
        state.cool_since = None
        if state.current <= settings.min_core_mhz:
            state.hot_since = state.reserve_since = None
            return state.current, 'minimum', None
        if regular_hot:
            if state.hot_since is None:
                state.hot_since = now
        else:
            state.hot_since = None
        if reserve_hot:
            if state.reserve_since is None:
                state.reserve_since = now
        else:
            state.reserve_since = None
        regular_wait = (state.hot_since + settings.hot_hold_seconds - now
                        if regular_hot else math.inf)
        reserve_wait = (state.reserve_since + RESERVE_HOLD_SECONDS - now
                        if reserve_hot else math.inf)
        # If both matured in this poll, only the reserve step is applied.
        use_reserve = reserve_wait <= 0 or (regular_wait > 0 and reserve_wait <= regular_wait)
        state.reduction_rule = 'reserve' if use_reserve else 'regular'
        wait = reserve_wait if use_reserve else regular_wait
        if wait > 0:
            return state.current, 'waiting_hot', math.ceil(wait)
        step = RESERVE_STEP_MHZ if use_reserve else settings.step_mhz
        state.current = max(settings.min_core_mhz, state.current - step)
        state.hot_since = now if regular_hot else None
        state.reserve_since = now if reserve_hot else None
        next_wait = min(settings.hot_hold_seconds if regular_hot else math.inf,
                        RESERVE_HOLD_SECONDS if reserve_hot else math.inf)
        return state.current, 'reduced', next_wait

    mode, wait = 'normal', None
    if temperature <= settings.cool_c and state.current < ceiling and allow_growth:
        state.hot_since = state.reserve_since = None
        if state.cool_since is None:
            state.cool_since = now
        wait = max(0, settings.cool_hold_seconds - (now - state.cool_since))
        mode = 'waiting_cool'
        if wait == 0:
            state.current = min(ceiling, state.current + settings.step_mhz)
            state.cool_since = now
            mode, wait = 'recovering', settings.cool_hold_seconds
    else:
        state.reset_timers()
        mode = ('waiting_fans' if temperature >= settings.hot_c and not cooling_at_max
                else 'holding' if state.current < ceiling else 'normal')
    return state.current, mode, math.ceil(wait) if wait is not None else None


def notify(message):
    address = os.environ.get('NOTIFY_SOCKET')
    if address:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect('\0' + address[1:] if address.startswith('@') else address)
            sock.sendall(message.encode())


def restore_states(cfg, path=STATE, *, errors=None):
    states = {uuid: GPUState() for uuid in cfg.uuids}
    try:
        data = json.loads(bounded_read(path))
        if (not isinstance(data, dict) or type(data.get('version')) is not int
                or data['version'] != 1 or not isinstance(data.get('gpu'), dict)):
            raise ValueError('Invalid persistent core state')
    except FileNotFoundError:
        return states
    except (OSError, ValueError, RecursionError) as exc:
        if errors is None:
            raise
        errors.append(f'Persistent core state unreadable: {str(exc)[:160]}')
        return states
    for uuid in cfg.uuids:
        item = data['gpu'].get(uuid, {})
        if isinstance(item, dict):
            current, baseline = item.get('current'), item.get('baseline')
            emergency = item.get('emergency', False)
            if (all(v is None or valid_lock(v) for v in (current, baseline)) and
                    type(emergency) is bool):
                states[uuid] = GPUState(current, baseline, emergency)
                continue
        message = f'Invalid persistent state for {uuid}'
        if errors is None:
            raise ValueError(message)
        errors.append(message)
    return states


def persistent_data(states):
    return {'version': 1, 'gpu': {u: s.persistent() for u, s in states.items()}}


def apply_clock(uuid, desired):
    if not UUID_RE.fullmatch(uuid) or type(desired) is not int or not 500 <= desired <= 5000:
        raise ValueError('Invalid clock command')
    command(['/usr/bin/nvidia-smi', '-i', uuid, '--lock-gpu-clocks', str(desired)])
    snapshot = read_gpus()
    locks = read_locks(snapshot)
    actual = locks.get(uuid)
    if not valid_lock(actual) or abs(actual - desired) > CLOCK_TOLERANCE:
        raise RuntimeError(f'Lock not confirmed: requested={desired}, readback={actual}')
    return actual


def verify_recovery(uuid, sampled_gpus, baseline, observed, settings):
    """Recheck the live HiveOS ceiling and clock before allowing a recovery write."""
    live_gpus = read_gpus()
    if (set(live_gpus) != set(sampled_gpus) or
            live_gpus[uuid]['bus'] != sampled_gpus[uuid]['bus']):
        raise ValueError('GPU inventory changed before clock recovery')
    live_lock = read_locks(live_gpus).get(uuid)
    live_gpu = live_gpus[uuid]
    if (not valid_lock(live_lock) or live_lock < observed - CLOCK_TOLERANCE or
            live_gpu['temperature_c'] is None or live_gpu['temperature_c'] > settings.cool_c or
            live_gpu['fan_percent'] is None):
        raise ValueError('Clock or thermal conditions changed before clock recovery')
    if hive_baselines({uuid: live_gpu}).get(uuid) != baseline:
        raise ValueError('Hive clock profile changed before clock recovery')
    return live_lock


def run(cfg, config_path=CONFIG, state_path=STATE, runtime=RUNTIME):
    runtime = Path(runtime)
    runtime.mkdir(parents=True, exist_ok=True)
    lock_file = (runtime / 'controller.lock').open('a')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    restore_errors = []
    states = restore_states(cfg, state_path, errors=restore_errors)
    if restore_errors:
        LOG.error('Clock recovery disabled until persistent state is repaired and service restarted: %s',
                  '; '.join(restore_errors))
    saved = persistent_data(states)
    persistence_error = None
    last_modes = {}
    last_sample = None
    notify('READY=1')
    try:
        while not stop.is_set():
            started = time.monotonic()
            if last_sample is not None and started - last_sample > MAX_SAMPLE_GAP_SECONDS:
                for state in states.values():
                    state.reset_timers()
            last_sample = started
            config_error = None
            try:
                candidate = load_config(config_path)
                if candidate.uuids != cfg.uuids:
                    raise ValueError('Changing rig UUIDs requires stopping the service')
                cfg = candidate
            except (OSError, ValueError, TypeError) as exc:
                config_error = str(exc)[:220]
            errors = restore_errors.copy()
            gpus, locks, baselines, limits = {}, {}, {}, None
            try:
                gpus = read_gpus()
                locks = read_locks(gpus)
                if set(gpus) != set(cfg.uuids):
                    errors.append('GPU inventory differs from configured UUIDs; reconfigure after hardware changes')
            except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
                errors.append(f'GPU readings: {str(exc)[:180]}')
            try:
                baselines = hive_baselines(gpus)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f'HiveOC: {str(exc)[:180]}')
            try:
                limits = cooling_limits()
            except (OSError, ValueError, TypeError, KeyError) as exc:
                errors.append(f'Cooling: {str(exc)[:180]}')
            case_at_max = bool(limits and limits['case_at_max'])
            case_max = limits.get('case_max') if limits else None
            gpu_max = limits.get('gpu_max') if limits else None
            gpu_threshold = limits.get('gpu_fan_threshold_percent') if limits else None
            if limits and case_max is None and limits.get('case_error'):
                errors.append(f"Case cooling: {limits['case_error']}")
            if limits and limits.get('gpu_error'):
                errors.append(f"Hive AutoFan: {limits['gpu_error']}")
            rows = {}
            for uuid in cfg.uuids:
                notify('WATCHDOG=1')
                gpu = gpus.get(uuid, {})
                state = states[uuid]
                settings = cfg.for_gpu(uuid)
                temp = gpu.get('temperature_c')
                observed = locks.get(uuid)
                actual_core = gpu.get('actual_core_mhz')
                fan = gpu.get('fan_percent')
                cooling_max = (case_at_max and gpu_threshold is not None and fan is not None and
                               fan >= gpu_threshold)
                # Invalid cooling data blocks recovery, but never the 75 C emergency.
                baseline = baselines.get(uuid, state.baseline)
                if uuid not in baselines and baseline is None and temp is not None and \
                        temp >= settings.emergency_c:
                    # A failed HiveOC read must not prevent a downward emergency action.
                    baseline = observed
                previous_target = state.current
                previous_emergency = state.emergency
                desired, mode, wait = decide(state, settings, baseline, observed,
                                             temp, cooling_max, started,
                                             allow_growth=not errors and not config_error
                                             and fan is not None and not persistence_error,
                                             case_at_max=case_at_max, case_max=case_max, gpu_max=gpu_max,
                                             actual_core=actual_core)
                if (desired is not None and valid_lock(observed) and desired < observed and
                        mode in ('reduced', 'emergency') and desired != previous_target):
                    # Even a last step smaller than the readback tolerance needs a write.
                    state.pending_reduction = desired
                elif state.pending_reduction != desired or (
                        desired is not None and valid_lock(observed) and observed <= desired):
                    state.pending_reduction = None
                error = None
                emergency_without_lock = mode == 'emergency' and not valid_lock(observed)
                recovery = mode == 'recovering' and desired != previous_target
                should_write = valid_lock(observed) and desired is not None and (
                    abs(desired - observed) > CLOCK_TOLERANCE or
                    state.pending_reduction == desired or (recovery and desired > observed))
                if should_write and desired > observed:
                    try:
                        observed = verify_recovery(uuid, gpus, baseline, observed, settings)
                        should_write = (desired > observed or
                                        observed - desired > CLOCK_TOLERANCE)
                    except (OSError, ValueError, TypeError, KeyError,
                            subprocess.SubprocessError, RuntimeError) as exc:
                        state.current = previous_target if previous_target is not None else observed
                        state.emergency = previous_emergency
                        state.reset_timers()
                        desired, mode, wait = state.current, 'apply_error', None
                        error = str(exc)[:220]
                        should_write = False
                growth_write = should_write and desired > observed
                prior_target = previous_target if previous_target is not None else observed
                changed = persistent_data(states)
                if growth_write:
                    # A restart during recovery must restore the earlier, lower cap.
                    changed['gpu'][uuid].update(current=prior_target, emergency=previous_emergency)
                if not restore_errors and (growth_write or changed != saved or persistence_error):
                    try:
                        # Persist before writing the clock, but commit growth only after readback.
                        atomic_json(state_path, changed)
                        saved = changed
                        persistence_error = None
                    except OSError as exc:
                        persistence_error = f'State persistence failed: {str(exc)[:160]}'
                try:
                    if not stop.is_set() and (emergency_without_lock or (
                            should_write and (not persistence_error or desired < observed))):
                        observed = apply_clock(uuid, desired)
                        state.pending_reduction = None
                        LOG.info('%s lock=%sMHz readback=%sMHz mode=%s rule=%s', uuid, desired,
                                 observed, mode, state.reduction_rule)
                        if growth_write:
                            try:
                                changed = persistent_data(states)
                                atomic_json(state_path, changed)
                                saved = changed
                            except OSError as exc:
                                persistence_error = f'State persistence failed: {str(exc)[:160]}'
                                state.current, state.emergency = prior_target, previous_emergency
                                desired = prior_target
                                if observed > prior_target:
                                    state.pending_reduction = prior_target
                                    observed = apply_clock(uuid, prior_target)
                                    state.pending_reduction = None
                except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
                    if growth_write:
                        state.current, state.emergency = prior_target, previous_emergency
                        desired = state.current
                    error = str(exc)[:220]
                    state.reset_timers()
                    mode = 'apply_error'
                    wait = None
                if persistence_error:
                    if growth_write:
                        state.current, state.emergency = prior_target, previous_emergency
                        desired = state.current
                    error = error or persistence_error
                    mode = 'storage_error'
                    state.reset_timers()
                    wait = None
                if temp is None or not valid_lock(observed):
                    error = error or 'GPU temperature or locked clock unavailable'
                row = dict(gpu, uuid=uuid, baseline_mhz=baseline,
                           target_mhz=desired, locked_mhz=observed, mode=mode,
                           wait_seconds=wait, emergency=state.emergency,
                           cooling_at_max=cooling_max, reduction_rule=state.reduction_rule,
                           settings=asdict(settings), error=error)
                row['pause_reason'] = (('Фиксация ядра в HiveOS отключена' if baseline is None else
                                        f'Частота HiveOS {baseline} МГц ниже минимума '
                                        f'{settings.min_core_mhz} МГц')
                                       if mode == 'paused_hive' else None)
                rows[uuid] = row
                current_mode = (mode, state.reduction_rule, error)
                if last_modes.get(uuid) != current_mode:
                    LOG.info('%s temp=%s mode=%s wait=%s error=%s rule=%s',
                             uuid, temp, mode, wait, error, state.reduction_rule)
                    last_modes[uuid] = current_mode
            atomic_json(runtime / 'status.json', dict(updated_at=time.time(),
                        controller_pid=os.getpid(), config_hash=cfg.digest,
                        config_error=config_error, errors=errors, cooling=limits, gpu=rows))
            notify('WATCHDOG=1')
            stop.wait(max(0, POLL - (time.monotonic() - started)))
    finally:
        # Deliberately leave all clock restrictions in place on stop/restart.
        lock_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--check', action='store_true', help='validate configuration; no writes')
    parser.add_argument('--inspect', action='store_true', help='read rig inputs; no writes')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cfg = load_config(args.config)
    if args.check:
        print('Configuration OK:', len(cfg.uuids), 'GPUs; SHA256', cfg.digest)
    elif args.inspect:
        gpus = read_gpus()
        print(json.dumps(dict(gpu=gpus, locks=read_locks(gpus),
              baselines=hive_baselines(gpus), cooling=cooling_limits()), indent=2))
    else:
        run(cfg, args.config)


if __name__ == '__main__':
    main()
