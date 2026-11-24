"""Installer transaction checks; all filesystem and service effects are isolated."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

import configure
import core_control as core
import install


class InstallationTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.build = self.root / 'build'
        self.build.mkdir()
        self.cfg, text = configure.make_configs(
            dict(hwmon_name='test', device_path='/sys/devices/platform/test.0', pwm_channel=3),
            ['GPU-00000000-0000-0000-0000-000000000001'])
        (self.build / 'cryptone-fan-control.json').write_text(json.dumps(self.cfg))
        (self.build / 'cryptone-core-control.toml').write_text(text)
        self.files = {source: (str(self.root / 'installed' / Path(destination).name), mode)
                      for source, (destination, mode) in install.FILES.items()}

    def test_mismatched_gpu_inventories_are_rejected_before_install(self):
        config = core.load_config(self.build / 'cryptone-core-control.toml')
        install.verify_pair(self.cfg, config)
        with self.assertRaises(ValueError):
            install.verify_pair(dict(self.cfg, expected_gpu_uuids=[]), config)

    def test_backup_copies_and_verification_precede_enable(self):
        events = []
        def command(args, **kwargs):
            events.append(args[:2])
            return subprocess.CompletedProcess(args, 0)
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {'pwm_raw': 140, 'pwm_mode': 2})), \
             patch.object(install.subprocess, 'run', side_effect=command), \
             patch.object(install, 'wait_ready', side_effect=lambda *args: events.append(['ready'])), \
             contextlib.redirect_stdout(io.StringIO()):
            install.install(self.build)
        self.assertLess(events.index(['ready']), events.index(['systemctl', 'enable']))
        backups = list(self.root.glob('cryptone-install-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].stat().st_mode & 0o777, 0o700)
        self.assertEqual(json.loads((backups[0] / 'baseline.json').read_text())['pwm_mode'], 2)
        for source, (destination, mode) in self.files.items():
            self.assertEqual(Path(destination).read_bytes(), (backups[0] / source).read_bytes())
            self.assertEqual(Path(destination).stat().st_mode & 0o777, mode)

    def test_start_failure_stops_services_and_requests_failsafe_without_enabling(self):
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            if args[:2] == ['systemctl', 'start']:
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args, 0)
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {})), \
             patch.object(install.subprocess, 'run', side_effect=command), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                install.install(self.build)
        self.assertFalse(any(c[:2] == ['systemctl', 'enable'] for c in calls))
        self.assertTrue(any(c[:2] == ['systemctl', 'stop'] for c in calls))
        self.assertTrue(any(c[-1] == '--failsafe' for c in calls))

    def test_exclusive_copy_does_not_overwrite_racing_file_or_start_services(self):
        destination = Path(next(iter(self.files.values()))[0])
        destination.parent.mkdir()
        destination.write_text('preserve existing source')
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {})), \
             patch.object(install.subprocess, 'run') as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(FileExistsError):
                install.install(self.build)
        self.assertEqual(destination.read_text(), 'preserve existing source')
        run.assert_not_called()

    def test_changed_configuration_between_preflight_and_copy_is_rejected(self):
        changed = dict(self.cfg, max_pwm_percent=90)
        (self.build / 'cryptone-fan-control.json').write_text(json.dumps(changed))
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {})), \
             patch.object(install.subprocess, 'run') as run:
            with self.assertRaisesRegex(ValueError, 'changed during preflight'):
                install.install(self.build)
        self.assertFalse((self.root / 'installed').exists())
        run.assert_not_called()

    def test_fresh_status_must_confirm_config_and_every_gpu(self):
        uuids = set(self.cfg['expected_gpu_uuids'])
        base = dict(controller_pid=10, updated_at=time.time(), config_hash='hash')
        fan = dict(base, mode='normal', gpu_temperatures=dict.fromkeys(uuids, 65))
        core_status = dict(base, gpu={u: {'mode': 'normal'} for u in uuids})
        self.assertTrue(install.healthy_status(fan, 10, 'hash', uuids, is_core=False))
        self.assertTrue(install.healthy_status(core_status, 10, 'hash', uuids, is_core=True))
        self.assertFalse(install.healthy_status(fan, 10, 'wrong', uuids, is_core=False))
        self.assertFalse(install.healthy_status(fan, 11, 'hash', uuids, is_core=False))
        self.assertFalse(install.healthy_status([], 10, 'hash', uuids, is_core=False))
        self.assertFalse(install.healthy_status(dict(core_status, gpu={}), 10, 'hash', uuids, is_core=True))
        row = next(iter(core_status['gpu'].values()))
        row['mode'] = 'paused_hive'
        self.assertFalse(install.healthy_status(core_status, 10, 'hash', uuids, is_core=True))

    def test_verify_failure_removes_only_files_created_by_installer(self):
        verify_error = subprocess.CalledProcessError(1, ['systemd-analyze', 'verify'])
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {})), \
             patch.object(install.subprocess, 'run', side_effect=verify_error), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                install.install(self.build)
        self.assertEqual(list((self.root / 'installed').iterdir()), [])
        self.assertTrue((self.build / 'cryptone-fan-control.json').exists())

    def test_start_interruption_reports_failed_stop_and_attempts_full_cooling(self):
        def command(args, **kwargs):
            if args[:2] == ['systemctl', 'start']:
                raise KeyboardInterrupt
            if args[:2] == ['systemctl', 'stop']:
                raise subprocess.CalledProcessError(1, args)
            return subprocess.CompletedProcess(args, 0)
        errors = io.StringIO()
        with patch.object(install, 'FILES', self.files), \
             patch.object(install, 'BACKUP_ROOT', self.root), \
             patch.object(install, 'preflight', return_value=(self.cfg, {})), \
             patch.object(install.subprocess, 'run', side_effect=command) as run, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(errors):
            with self.assertRaises(KeyboardInterrupt):
                install.install(self.build)
        self.assertIn('Check service and fan state', errors.getvalue())
        self.assertTrue(any(c.args[0][-1] == '--failsafe' for c in run.call_args_list))

    def test_config_generation_can_retry_after_incomplete_write(self):
        output = self.root / 'generated'
        original_open = Path.open
        def broken_open(path, *args, **kwargs):
            if path.parent == output and path.suffix == '.toml':
                raise OSError('simulated permission failure')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'open', broken_open), self.assertRaises(OSError):
            configure.write_configs(output, self.cfg, '# sample')
        self.assertFalse(output.exists())
        configure.write_configs(output, self.cfg, '# sample')
        self.assertTrue((output / 'cryptone-core-control.toml').exists())


if __name__ == '__main__':
    unittest.main()
