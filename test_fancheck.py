"""Combined status remains read-only and useful when either source fails."""
import contextlib
import importlib.machinery
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
loader = importlib.machinery.SourceFileLoader('fancheck', str(HERE / 'fancheck'))
spec = importlib.util.spec_from_loader(loader.name, loader)
fancheck = importlib.util.module_from_spec(spec)
loader.exec_module(fancheck)
SERVICE_STATUS = fancheck.service_status
spec = importlib.util.spec_from_file_location('fancheck_control', HERE / 'cryptone-fan-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class StatusFixture:
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.cfg = control.load_config(HERE / 'cryptone-fan-control.json')
        self.cfg.update(min_pwm_percent=30, target_c=65, max_pwm_percent=59)
        self.config = self.root / 'config.json'
        self.case_path = self.root / 'case.json'
        self.core_path = self.root / 'core.json'
        self.config.write_text(json.dumps(self.cfg))
        self.ids = self.cfg['expected_gpu_uuids']
        self.case = dict(updated_at=100, controller_pid=10, applied_config=self.cfg,
                         config_hash=control.config_hash(self.cfg), mode='normal',
                         hottest_c=66, error=None, config_error=None)
        self.core = dict(updated_at=100, controller_pid=20, config_error=None, errors=[],
                         cooling={'gpu_max': 90}, gpu={u: dict(
                             index=i, temperature_c=65-i, fan_percent=30+i,
                             baseline_mhz=2900, locked_mhz=2850, actual_core_mhz=2842,
                             mode='waiting_cool', wait_seconds=120, error=None)
                             for i,u in enumerate(self.ids)})
        self.write_status()
        self.hw = {'pwm_raw': 150, 'pwm_mode': 1, 'hwmon_path': '/sys/fake'}
        test = self
        class Fan:
            def __init__(self, cfg):
                self.cfg = cfg
            def state(self):
                return dict(test.hw)
            def set(self, *args):
                raise AssertionError('Status command must not write PWM')
        self.control = {'Fan': Fan, 'load_config': control.load_config,
                        'config_hash': control.config_hash, 'validate_config': control.validate_config}
        patches = [patch.object(fancheck, 'CONFIG_PATH', self.config),
                   patch.object(fancheck, 'CASE_STATUS', self.case_path),
                   patch.object(fancheck, 'CORE_STATUS', self.core_path),
                   patch.object(fancheck, 'CORE_CONFIG', HERE / 'cryptone-core-control.toml'),
                   patch.object(fancheck.runpy, 'run_path', return_value=self.control),
                   patch.object(fancheck.time, 'time', return_value=105),
                   patch.object(fancheck, 'service_status', side_effect=lambda service:
                       {'state':'active','pid':10 if service.startswith('cryptone-fan-') else 20})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def write_status(self):
        self.case_path.write_text(json.dumps(self.case))
        self.core_path.write_text(json.dumps(self.core))

    def command(self, args=None):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            rc = fancheck.main(args or [])
        return rc, output.getvalue()


class CombinedStatusTests(StatusFixture, unittest.TestCase):
    def test_healthy_combined_command_shows_settings_gpus_and_instructions(self):
        before = {p:p.read_bytes() for p in (self.config,self.case_path,self.core_path)}
        rc, text = self.command()
        self.assertEqual(rc, 0)
        self.assertTrue(text.startswith('Заданные параметры FAN : 30%; 65 °C; 59%\n'
                                        'Текущие значение FAN: 59%, 66°C\n\n'))
        self.assertNotIn('возраст показаний', text)
        self.assertNotIn('RPM', text)
        self.assertNotIn('сохранено', text)
        self.assertNotIn('применено', text)
        self.assertNotIn('active', text)
        self.assertEqual(text.count(']: '), 4)
        self.assertIn('HiveOS 2900; фиксация 2850; рабочая 2842 МГц', text)
        self.assertIn('ожидание повышения; осталось 120 с', text)
        self.assertIn('Максимум вентиляторов GPU: 90%', text)
        self.assertLess(text.index('GPU 0 ['), text.index('GPU 3 ['))
        self.assertNotIn('fanset ЦЕЛЬ МИН МАКС', text)
        self.assertIn('nano /etc/cryptone-core-control.toml', text)
        self.assertTrue(text.rstrip().endswith('Настройки перечитываются за 5 с.'))
        self.assertEqual(before, {p:p.read_bytes() for p in before})

    def test_json_is_one_combined_object_without_human_help(self):
        rc, text = self.command(['--json'])
        self.assertEqual(rc, 0)
        report = json.loads(text)
        self.assertEqual(set(report), {'case','core'})
        self.assertTrue(report['case']['settings_confirmed'])
        self.assertEqual(report['case']['pwm']['percent'], 58.8)
        self.assertEqual(len(report['core']['status']['gpu']), 4)

    def test_stale_future_or_other_process_never_confirms_applied_settings(self):
        for changes in ({'updated_at':50},{'updated_at':110},{'controller_pid':99}):
            with self.subTest(changes=changes):
                saved = dict(self.case)
                self.case.update(changes)
                self.write_status()
                rc,text=self.command()
                self.assertEqual(rc,1)
                self.assertIn('ВНИМАНИЕ [Корпус]:', text)
                self.assertIn('Текущие значение FAN: 59%, —°C',text)
                self.assertFalse(fancheck.collect()['case']['settings_confirmed'])
                self.assertIn('GPU 3 [',text)
                self.case=saved

    def test_saved_and_applied_values_are_distinguished_on_pending_reload(self):
        self.config.write_text(json.dumps(dict(self.cfg,max_pwm_percent=70)))
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('Заданные параметры FAN : 30%; 65 °C; 70%',text)
        self.assertNotIn('последнее применённое',text)
        self.assertIn('Применение сохранённых настроек fanset не подтверждено',text)

    def test_hash_mismatch_and_config_error_do_not_claim_confirmation(self):
        for changes in ({'config_hash':'other'},{'config_error':'bad JSON'}):
            with self.subTest(changes=changes):
                self.case.update(changes)
                self.write_status()
                report=fancheck.collect()
                self.assertFalse(report['case']['settings_confirmed'])

    def test_missing_or_broken_core_status_keeps_case_section_and_help(self):
        for raw in (None,'{broken','[]','{"gpu": [1, 2]}','{"gpu":{"one":null}}'):
            with self.subTest(raw=raw):
                if raw is None:
                    self.core_path.unlink()
                else:
                    self.core_path.write_text(raw)
                rc,text=self.command()
                self.assertEqual(rc,1)
                self.assertIn('Текущие значение FAN: 59%, 66°C',text)
                self.assertIn('ВНИМАНИЕ [Ядро]:',text)
                self.assertIn('Пример: fanset 65 30 100',text)

    def test_broken_case_config_or_missing_hardware_does_not_hide_gpus(self):
        self.config.write_text('{broken')
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('Настройки fanset недоступны',text)
        self.assertIn('GPU 3 [',text)
        self.config.write_text(json.dumps(self.cfg))
        with patch.object(fancheck,'read_pwm',side_effect=RuntimeError('missing sensor')):
            rc,text=self.command()
            self.assertEqual(rc,1)
            self.assertIn('missing sensor',text)
            self.assertIn('GPU 0 [',text)

    def test_bios_pwm_is_not_reported_as_current_duty(self):
        self.hw['pwm_mode']=2
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('Текущие значение FAN: —%, 66°C',text)
        self.assertIn('управляется BIOS',text)

    def test_emergency_full_speed_bypasses_normal_maximum_in_display(self):
        self.hw.update(pwm_mode=0,pwm_raw=7)
        self.case.update(mode='emergency',error='GPU unavailable')
        self.write_status()
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('Текущие значение FAN: 100%, 66°C',text)
        self.assertIn('Заданные параметры FAN : 30%; 65 °C; 59%',text)
        self.assertIn('100% независимо от обычного максимума',text)

    def test_per_gpu_error_pause_and_limit_warning_are_visible(self):
        self.core['gpu'][self.ids[0]].update(error='temperature unavailable',temperature_c=None)
        self.core['gpu'][self.ids[1]].update(mode='paused_hive',pause_reason='Hive lock disabled')
        self.case['target_unmet_at_limit']=True
        self.write_status()
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('temperature unavailable',text)
        self.assertIn('Причина: Hive lock disabled',text)
        self.assertIn('целевая температура не удерживается',text)

    def test_exact_requested_normal_template(self):
        self.hw['pwm_raw']=79
        self.case['hottest_c']=67
        temperatures=(65,63,67,61)
        fans=(32,30,31,30)
        for i,gpu in enumerate(self.core['gpu'].values()):
            gpu.update(temperature_c=temperatures[i],fan_percent=fans[i],baseline_mhz=3050,
                       locked_mhz=3050,actual_core_mhz=3045,mode='normal',wait_seconds=None)
        self.write_status()
        rc,text=self.command()
        expected='Заданные параметры FAN : 30%; 65 °C; 59%\nТекущие значение FAN: 31%, 67°C\n\n'
        for i,uuid in enumerate(self.ids):
            expected+=(f'GPU {i} [{uuid[-8:]}]: {temperatures[i]} °C; вентилятор {fans[i]}%\n'
                       '  HiveOS 3050; фиксация 3050; рабочая 3045 МГц\n  норма\n')
        expected+=('  Максимум вентиляторов GPU: 90%\n\nНастройка:\n'
                   '  Пример: fanset 65 30 100; fanset — пошагово; fanset --show — настройки.\n'
                   '  Предел частоты каждой GPU задаётся в HiveOS.\n'
                   '  Пороги, шаг и задержки ядра: nano /etc/cryptone-core-control.toml\n'
                   '  Настройки перечитываются за 5 с.\n')
        self.assertEqual((rc,text),(0,expected))

    def test_rounding_changes_only_text_not_json(self):
        for raw,expected in ((79,31),(80,31),(81,32),(150,59),(255,100)):
            with self.subTest(raw=raw):
                self.hw['pwm_raw']=raw
                rc,text=self.command()
                self.assertEqual(rc,1 if raw == 255 else 0)
                self.assertIn(f'Текущие значение FAN: {expected}%,',text)
                report=json.loads(self.command(['--json'])[1])
                self.assertEqual(report['case']['pwm']['percent'],round(raw*100/255,1))

    def test_warnings_are_compact_before_help_details_stay_in_json(self):
        self.core['errors']=['first problem','second problem']
        self.case['error']='case problem'
        self.write_status()
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertEqual(text.count('ВНИМАНИЕ [Ядро]:'),1)
        self.assertIn('first problem; подробности: fancheck --json',text)
        self.assertNotIn('second problem',text)
        self.assertLess(text.index('ВНИМАНИЕ [Ядро]:'),text.index('Настройка:'))
        self.assertEqual(json.loads(self.command(['--json'])[1])['core']['status']['errors'],
                         ['first problem','second problem'])

    def test_stale_gpu_readings_are_not_presented_as_current(self):
        self.core['updated_at']=50
        self.write_status()
        rc,text=self.command()
        self.assertEqual(rc,1)
        self.assertIn('°C; вентилятор —%',text)
        self.assertIn('HiveOS —; фиксация —; рабочая — МГц',text)
        self.assertIn('показания недоступны',text)
        self.assertNotIn('осталось 120 с',text)

    def test_service_query_only_uses_read_only_systemctl_show(self):
        # Bypass setUp's mock for this function; mock subprocess itself instead.
        with patch.object(fancheck.subprocess,'run',return_value=subprocess.CompletedProcess(
                [],0,'MainPID=10\nActiveState=active\n','')) as run:
            result=SERVICE_STATUS('cryptone-fan-control.service')
            self.assertEqual(result,{'state':'active','pid':10})
            self.assertEqual(run.call_args.args[0],['systemctl','show',
                'cryptone-fan-control.service','--property=ActiveState','--property=MainPID'])

if __name__ == '__main__':
    unittest.main()
