import contextlib
import fcntl
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
loader = importlib.machinery.SourceFileLoader('fanset', str(HERE / 'fanset'))
spec = importlib.util.spec_from_loader(loader.name, loader)
fanset = importlib.util.module_from_spec(spec)
loader.exec_module(fanset)
spec = importlib.util.spec_from_file_location('control_fanset_tests', HERE / 'cryptone-fan-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class CommandTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'config.json'
        self.cfg = control.load_config(HERE / 'cryptone-fan-control.json')
        self.path.write_text(json.dumps(self.cfg))
        self.original = self.path.read_bytes()
        self.output = io.StringIO()
        self.patches = [patch.object(fanset, 'CONFIG_PATH', self.path),
                        patch.object(fanset, 'load_control', return_value=control),
                        patch.object(fanset.os, 'geteuid', return_value=0)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def command(self, argv, replies=None, acknowledgement=True):
        def applied(_, cfg):
            return ({'pwm_actual_percent': 45.9, 'mode': 'normal'}
                    if acknowledgement else None)
        with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output), \
                patch.object(fanset, 'wait_applied', side_effect=applied), \
                patch('builtins.input', side_effect=replies or []):
            return fanset.main(argv)

    def test_arguments_save_all_settings_and_preserve_private_fields(self):
        self.assertEqual(self.command(['70', '35', '60']), 0)
        saved = control.load_config(self.path)
        self.assertEqual((saved['target_c'], saved['min_pwm_percent'],
                          saved['max_pwm_percent'], saved['recovery_c']), (70, 35, 60, 72))
        self.assertEqual(saved['expected_gpu_uuids'], self.cfg['expected_gpu_uuids'])
        self.assertEqual(saved['critical_c'], 75)
        self.assertIn('Применено.', self.output.getvalue())
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o644)
        self.assertEqual(list(self.path.parent.glob('.config.json.*')), [])

    def test_interactive_defaults_and_confirmation(self):
        self.assertEqual(self.command([], ['', '', '', 'да']), 0)
        self.assertEqual(control.load_config(self.path), self.cfg)
        self.assertIn('Новые настройки', self.output.getvalue())

    def test_cancel_and_eof_leave_the_original_file(self):
        self.assertEqual(self.command([], ['66', '35', '70', 'нет']), 0)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.command([], ['66', EOFError()]), 130)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_invalid_inputs_never_write(self):
        for values in [('19', '30', '100'), ('71', '30', '100'), ('65', '29', '100'),
                       ('65', '55', '55'), ('65', '90', '80'), ('65', '30', '101'),
                       ('65.0', '30', '100'), ('65', '30%', '100'),
                       ('６５', '30', '100'), ('nan', '30', '100')]:
            with self.subTest(values=values):
                self.assertEqual(self.command(list(values)), 1)
                self.assertEqual(self.path.read_bytes(), self.original)

    def test_parallel_command_is_rejected_without_writing(self):
        with self.path.with_name('.cryptone-fan-control.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.assertEqual(self.command(['66', '35', '70']), 1)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_unconfirmed_save_is_reported_without_false_success(self):
        self.assertEqual(self.command(['66', '35', '70'], acknowledgement=False), 2)
        self.assertEqual(control.load_config(self.path)['target_c'], 66)
        self.assertNotIn('Применено.', self.output.getvalue())
        self.assertIn('применение не подтверждено', self.output.getvalue())

    def test_read_only_show_does_not_write_or_wait(self):
        with patch.object(fanset, 'read_status', return_value={}), \
                patch.object(fanset, 'service_info', return_value=('inactive', 0)):
            self.assertEqual(self.command(['--show']), 0)
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertIn('Сохранено:', self.output.getvalue())
        self.assertIn('inactive', self.output.getvalue())

    def test_acknowledgement_requires_matching_fresh_valid_status(self):
        valid = {'updated_at': 100, 'config_hash': control.config_hash(self.cfg),
                 'config_error': None, 'controller_pid': 123, 'applied_config': self.cfg,
                 'mode': 'normal', 'pwm_actual_percent': 55}
        invalid = dict(valid, config_error='broken config')
        with patch.object(fanset, 'read_status', side_effect=[invalid, valid]), \
                patch.object(fanset, 'service_info', return_value=('active', 123)), \
                patch.object(fanset.time, 'time', return_value=100), \
                patch.object(fanset.time, 'sleep'):
            self.assertEqual(fanset.wait_applied(control, self.cfg), valid)
        for status in (dict(valid, updated_at=50), dict(valid, config_hash='other'),
                       dict(valid, controller_pid=321), dict(valid, mode='failsafe'),
                       dict(valid, pwm_actual_percent=None), dict(valid, pwm_actual_percent=0),
                       dict(valid, mode='emergency'), dict(valid, applied_config={}), invalid):
            with patch.object(fanset, 'read_status', return_value=status), \
                    patch.object(fanset, 'service_info', return_value=('active', 123)), \
                    patch.object(fanset.time, 'time', return_value=100), \
                    patch.object(fanset.time, 'monotonic', side_effect=[0, 0, 16]), \
                    patch.object(fanset.time, 'sleep'):
                self.assertIsNone(fanset.wait_applied(control, self.cfg))

    def test_systemd_timeout_after_write_is_saved_but_unconfirmed(self):
        cfg = fanset.candidate_config(control, self.cfg, [66, 35, 70])
        status = {'updated_at': 100, 'config_hash': control.config_hash(cfg),
                  'applied_config': cfg, 'controller_pid': 123, 'mode': 'normal',
                  'pwm_actual_percent': 55}
        with contextlib.redirect_stdout(self.output), contextlib.redirect_stderr(self.output), \
                patch.object(fanset, 'read_status', return_value=status), \
                patch.object(fanset.subprocess, 'run', side_effect=subprocess.TimeoutExpired('systemctl', 1)), \
                patch.object(fanset.time, 'time', return_value=100), \
                patch.object(fanset.time, 'monotonic', side_effect=[0, 0, 16]), \
                patch.object(fanset.time, 'sleep'):
            self.assertEqual(fanset.main(['66', '35', '70']), 2)
        self.assertEqual(control.load_config(self.path), cfg)
        self.assertIn('применение не подтверждено', self.output.getvalue())

    def test_expanded_config_over_size_limit_does_not_replace_working_file(self):
        self.path.write_text(json.dumps(dict(self.cfg, notes=[0] * 10000), separators=(',', ':')))
        before = self.path.read_bytes()
        self.assertEqual(self.command(['66', '35', '70']), 1)
        self.assertEqual(self.path.read_bytes(), before)


class ReloadTests(unittest.TestCase):
    def setUp(self):
        self.cfg = control.load_config(HERE / 'cryptone-fan-control.json')

    def test_startup_and_live_bounds_keep_speed_within_limits(self):
        c = control.Controller(self.cfg)
        self.assertEqual(c.current, 55)
        c.current = 46
        c.last_adjust = 5
        c.reconfigure(dict(self.cfg, target_c=66, max_pwm_percent=90))
        self.assertEqual((c.current, c.last_adjust), (46, 5))
        c.reconfigure(dict(self.cfg, min_pwm_percent=50))
        self.assertEqual(c.current, 50)
        self.assertEqual(control.Controller(dict(self.cfg, min_pwm_percent=60)).current, 60)

    def test_reloading_does_not_clear_emergency(self):
        c = control.Controller(self.cfg)
        self.assertEqual(c.update(75, 0), 100)
        self.assertEqual(c.update(70, 5), 100)
        c.reconfigure(dict(self.cfg, target_c=70, recovery_c=72, max_pwm_percent=80))
        self.assertTrue(c.emergency)
        self.assertEqual(c.update(71, 10), 100)
        self.assertEqual(c.update(71, 15), 100)
        self.assertEqual(c.update(71, 20), 80)

    def test_competing_controller_and_failsafe_cannot_write_hardware(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(control, 'RUNTIME', Path(directory)), \
                patch.object(control, 'Fan') as fan_factory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(self.cfg))
            with control.controller_lock():
                for flag in ([], ['--failsafe']):
                    with self.subTest(flag=flag), \
                            patch('sys.argv', ['case-control', '--config', str(path), *flag]), \
                            self.assertRaises(BlockingIOError):
                        control.main()
            fan_factory.return_value.set.assert_not_called()

    def test_restart_preserves_emergency_and_starting_hot_waits_for_recovery(self):
        for prior_emergency, temperatures, expected in (
                (True, [69, 69, 69], [100, 100, 100, self.cfg['max_pwm_percent'], 100]),
                (False, [74, 70, 70, 70], [100, 100, 100, 100, self.cfg['max_pwm_percent'], 100])):
            with self.subTest(prior_emergency=prior_emergency), tempfile.TemporaryDirectory() as directory, \
                    patch.object(control, 'RUNTIME', Path(directory)):
                fan = Mock()
                fan.set.side_effect = lambda percent, cfg=None: {
                    'pwm_raw': control.pwm_value(percent, cfg), 'pwm_mode': 1}
                if prior_emergency:
                    control.write_status({'mode': 'emergency'})
                    with control.controller_lock():
                        control.apply_failsafe(fan)
                    self.assertTrue(control.previous_emergency())
                    fan.set.reset_mock()

                class Stop:
                    cycle = 0

                    def is_set(self):
                        return self.cycle >= len(temperatures)

                    def wait(self, seconds):
                        self.cycle += 1

                with patch.object(control.threading, 'Event', Stop), \
                        patch.object(control.signal, 'signal'), \
                        patch.object(control, 'notify'), \
                        patch.object(control, 'read_temperatures',
                                     side_effect=[dict.fromkeys(self.cfg['expected_gpu_uuids'], temp)
                                                  for temp in temperatures]):
                    control.run(fan, self.cfg)
                self.assertEqual([call.args[0] for call in fan.set.call_args_list], expected)

    def test_real_loop_reloads_and_fails_safe_on_invalid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.json'
            path.write_text(json.dumps(self.cfg))
            statuses, writes = [], []

            class Fan:
                def set(self, percent, cfg=None):
                    writes.append(percent)
                    return {'pwm_raw': control.pwm_value(percent, cfg), 'pwm_mode': 1}

            class Stop:
                cycle = 0

                def is_set(self):
                    return self.cycle >= 6

                def wait(self, seconds):
                    self.cycle += 1
                    if self.cycle == 1:
                        path.write_text(json.dumps(dict(self_cfg, target_c=66, min_pwm_percent=60)))
                    elif self.cycle == 2:
                        path.write_text('{broken')
                    elif self.cycle == 3:
                        path.write_text(json.dumps(self_cfg))

                def set(self):
                    pass

            self_cfg = self.cfg
            with patch.object(control, 'RUNTIME', Path(directory)), \
                    patch.object(control.threading, 'Event', Stop), \
                    patch.object(control.signal, 'signal'), \
                    patch.object(control, 'notify'), \
                    patch.object(control, 'write_status', side_effect=statuses.append), \
                    patch.object(control, 'read_temperatures',
                                 return_value={uuid: 65 for uuid in self.cfg['expected_gpu_uuids']}):
                control.run(Fan(), self.cfg, path)
            self.assertEqual([s['pwm_requested_percent'] for s in statuses], [55, 60, 100, 100, 100, 100])
            self.assertEqual(statuses[1]['applied_config']['target_c'], 66)
            self.assertTrue(statuses[2]['config_error'])
            self.assertEqual(statuses[2]['mode'], 'emergency')
            self.assertEqual(statuses[-1]['mode'], 'normal')
            self.assertEqual(writes[0], 100)
            self.assertEqual(writes[-1], 100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
