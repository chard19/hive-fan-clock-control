"""Hardware-independent regression tests for variable inventories and PWM selection."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import configure
import core_control as core
import test_fancheck

control = configure.control
fancheck = test_fancheck.fancheck
SELECTOR = dict(hwmon_name='test_chip', device_path='/sys/devices/platform/test.0', pwm_channel=4)


def uuids(count):
    return [f'GPU-00000000-0000-0000-0000-{i:012x}' for i in range(1, count + 1)]


class InventoryTests(unittest.TestCase):
    def test_configurations_and_temperature_sets_for_variable_counts(self):
        for count in (1, 2, 3, 4, 8, 12, 16, 64):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as folder:
                ids = uuids(count)
                cfg, text = configure.make_configs(SELECTOR, ids)
                path = Path(folder) / 'core.toml'
                path.write_text(text)
                self.assertEqual(core.load_config(path).uuids, tuple(ids))
                output = '\n'.join(f'{u}, {60+i%8}' for i, u in enumerate(reversed(ids)))
                self.assertEqual(set(control.parse_temperatures(output, ids)), set(ids))
                if count > 1:
                    with self.assertRaises(ValueError):
                        control.parse_temperatures('\n'.join(output.splitlines()[:-1]), ids)
                self.assertEqual(len(cfg['expected_gpu_uuids']), count)

    def test_empty_duplicate_and_malformed_inventories_are_rejected(self):
        for ids in ([], [uuids(1)[0]] * 2, ['GPU-' + '-' * 36], ['GPU-test']):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                configure.make_configs(SELECTOR, ids)

    def test_many_cards_have_independent_timers_and_hive_ceilings(self):
        settings = core.Settings()
        states = {u: core.GPUState() for u in uuids(16)}
        for now in (0, 179, 180):
            for i, (uuid, state) in enumerate(reversed(list(states.items()))):
                hot = uuid == uuids(16)[0]
                ceiling = 2700 + uuids(16).index(uuid) * 10
                core.decide(state, settings, ceiling, ceiling, 68 if hot else 65, True, now)
        self.assertEqual(states[uuids(16)[0]].current, 2650)
        for i, uuid in enumerate(uuids(16)[1:], 1):
            self.assertEqual(states[uuid].current, 2700 + i * 10)

    def test_hardware_and_inventory_cannot_switch_during_live_reload(self):
        cfg, _ = configure.make_configs(SELECTOR, uuids(4))
        changed = copy.deepcopy(cfg)
        changed['controller']['pwm_channel'] = 3
        with self.assertRaises(ValueError):
            control.assert_live_change(cfg, changed)
        changed = dict(cfg, expected_gpu_uuids=uuids(3))
        with self.assertRaises(ValueError):
            control.assert_live_change(cfg, changed)
        control.assert_live_change(cfg, dict(cfg, expected_gpu_uuids=list(reversed(uuids(4)))))

    def test_distinct_pci_domains_do_not_alias(self):
        self.assertNotEqual(core.bus_key('0000:02:00.0'), core.bus_key('0001:02:00.0'))
        output = '\n'.join(f'{i}, {uuid}, {i:04x}:02:00.0, 65, 2700, 90'
                           for i, uuid in enumerate(uuids(2)))
        self.assertEqual(len(core.parse_gpus(output)), 2)

    def test_configuration_generation_has_no_hardware_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'generated'
            candidate = dict(SELECTOR, hwmon_path='/sys/class/hwmon/hwmon9')
            with patch.object(control, 'discover_controllers', return_value=[candidate]), \
                 patch.object(configure.subprocess, 'run', return_value=subprocess.CompletedProcess(
                     [], 0, '\n'.join(uuids(12)), '')), \
                 patch.object(control.Fan, 'set', side_effect=AssertionError('write')), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                args = ['--hwmon', candidate['hwmon_path'], '--pwm', '4', '--output', str(output)]
                self.assertEqual(configure.main(args), 0)
                self.assertEqual(core.load_config(output / 'cryptone-core-control.toml').uuids, tuple(uuids(12)))
                self.assertEqual(configure.main(args), 1)  # Never overwrite a previous setup.


class PWMTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.hwmon = self.root / 'hwmon9'
        self.hwmon.mkdir()
        (self.hwmon / 'name').write_text('test_chip\n')
        (self.hwmon / 'pwm4').write_text('150\n')
        (self.hwmon / 'pwm4_enable').write_text('2\n')
        (self.hwmon / 'pwm1').write_text('130\n')
        (self.hwmon / 'pwm1_enable').write_text('1\n')
        self.patch = patch.object(control, 'device_identity', return_value=SELECTOR['device_path'])
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.cfg, _ = configure.make_configs(SELECTOR, uuids(1))

    def test_discovery_and_state_do_not_write_and_ignore_channel_suffixes(self):
        (self.hwmon / 'pwm4_auto_point1_pwm').write_text('99')
        before = {p: p.read_bytes() for p in self.hwmon.iterdir()}
        candidates = control.discover_controllers(self.root)
        self.assertEqual({c['pwm_channel'] for c in candidates}, {1, 4})
        self.assertEqual(control.Fan(self.cfg, self.root).state()['pwm_raw'], 150)
        self.assertEqual(before, {p: p.read_bytes() for p in self.hwmon.iterdir()})

    def test_hwmon_renumbering_does_not_change_the_selected_channel(self):
        self.hwmon.rename(self.root / 'hwmon2')
        fan = control.Fan(self.cfg, self.root)
        fan.set(40, self.cfg)
        self.assertEqual(fan.state()['pwm_raw'], 102)
        self.assertEqual((fan.path / 'pwm1').read_text(), '130\n')

    def test_duplicate_or_missing_channels_fail_without_writes(self):
        duplicate = self.root / 'hwmon8'
        duplicate.mkdir()
        for p in self.hwmon.iterdir():
            (duplicate / p.name).write_bytes(p.read_bytes())
        with self.assertRaises(RuntimeError):
            control.Fan(self.cfg, self.root)
        (duplicate / 'pwm4_enable').unlink()
        (self.hwmon / 'pwm4_enable').unlink()
        with self.assertRaises(RuntimeError):
            control.Fan(self.cfg, self.root)
        self.assertEqual((self.hwmon / 'pwm4').read_text(), '150\n')

    def test_identity_change_after_open_refuses_write(self):
        fan = control.Fan(self.cfg, self.root)
        (self.hwmon / 'name').write_text('other_chip\n')
        with self.assertRaises(RuntimeError):
            fan.set(100)
        self.assertEqual((self.hwmon / 'pwm4').read_text(), '150\n')

    def test_full_duty_precedes_switch_to_manual_and_failed_readback_is_rejected(self):
        fan = control.Fan(self.cfg, self.root)
        writes = []
        original = Path.write_text
        def record(path, text, *args, **kwargs):
            writes.append((path.name, text.strip()))
            return original(path, text, *args, **kwargs)
        with patch.object(Path, 'write_text', record):
            fan.set(30, self.cfg)
        self.assertEqual(writes, [('pwm4', '255'), ('pwm4_enable', '1'), ('pwm4', '77')])
        with patch.object(fan, 'state', return_value={'pwm_raw': 0, 'pwm_mode': 1}):
            with self.assertRaises(RuntimeError):
                fan.set(50)


class StatusRegressions(test_fancheck.StatusFixture, unittest.TestCase):
    def test_warning_exit_code_and_json_agree_for_thermal_limit_and_emergency(self):
        for change, text in (({'target_unmet_at_limit': True}, 'целевая температура'),
                             ({'mode': 'emergency'}, 'Аварийное охлаждение')):
            with self.subTest(change=change):
                self.case.update(change)
                if self.case['mode'] == 'emergency':
                    self.hw['pwm_raw'] = 255
                self.write_status()
                rc, rendered = self.command()
                self.assertEqual(rc, 1)
                self.assertIn(text, rendered)
                rc, raw = self.command(['--json'])
                self.assertEqual(rc, 1)
                self.assertTrue(json.loads(raw)['case']['warnings'])

    def test_missing_card_is_visible_and_malformed_modes_do_not_crash(self):
        del self.core['gpu'][self.ids[-1]]
        self.core['gpu'][self.ids[0]]['mode'] = []
        self.write_status()
        rc, text = self.command()
        self.assertEqual(rc, 1)
        self.assertIn(f'[{self.ids[-1][-8:]}]: — °C', text)
        self.assertIn('ВНИМАНИЕ [Ядро]', text)
        self.assertIn('ошибка датчиков', text)

    def test_null_or_invalid_observation_never_looks_healthy(self):
        for key, val in (('temperature_c', None), ('fan_percent', 101), ('locked_mhz', 0),
                         ('actual_core_mhz', []), ('mode', 'sensor_error')):
            with self.subTest(key=key):
                saved = copy.deepcopy(self.core)
                self.core['gpu'][self.ids[0]][key] = val
                self.write_status()
                self.assertEqual(self.command()[0], 1)
                self.core = saved

    def test_actual_pwm_below_or_above_applied_limits_warns(self):
        for raw in (0, 50, 255):
            self.hw['pwm_raw'] = raw
            rc, text = self.command()
            self.assertEqual(rc, 1)
            self.assertIn('PWM вне применённых границ', text)
        self.hw['pwm_raw'] = 150
        self.assertEqual(self.command()[0], 0)

    def test_nonfinite_json_is_rejected_and_case_status_still_renders(self):
        self.core_path.write_text('{"updated_at": NaN}')
        rc, text = self.command()
        self.assertEqual(rc, 1)
        self.assertIn('Текущие значение FAN: 59%, 66°C', text)
        self.assertIn('ВНИМАНИЕ [Ядро]', text)
        report = json.loads(self.command(['--json'])[1])
        self.assertTrue(report['core']['warnings'])

    def test_pending_reload_uses_applied_limits(self):
        self.config.write_text(json.dumps(dict(self.cfg, min_pwm_percent=60, max_pwm_percent=70)))
        rc, text = self.command()
        self.assertEqual(rc, 1)
        self.assertNotIn('PWM вне применённых границ', text)

    def test_dynamic_status_inventory(self):
        for count in (1, 3, 8, 16):
            with self.subTest(count=count):
                original = copy.deepcopy(next(iter(self.core['gpu'].values())))
                ids = uuids(count)
                self.cfg['expected_gpu_uuids'] = ids
                self.config.write_text(json.dumps(self.cfg))
                self.case.update(applied_config=self.cfg, config_hash=control.config_hash(self.cfg))
                self.core['gpu'] = {u: dict(original, index=i) for i, u in enumerate(ids)}
                self.write_status()
                cfg_path = self.root / 'core.toml'
                cfg_path.write_text(configure.make_configs(SELECTOR, ids)[1])
                with patch.object(fancheck, 'CORE_CONFIG', cfg_path):
                    rc, text = self.command()
                    self.assertEqual(rc, 0)
                    self.assertEqual(text.count(']: '), count)


if __name__ == '__main__':
    unittest.main()
