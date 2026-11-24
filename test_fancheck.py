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

    def test_reserve_status_shows_countdown_and_hive_fan_maximum(self):
        self.core['gpu'][self.ids[0]].update(
            mode='waiting_hot', wait_seconds=35, reduction_rule='reserve')
        self.core['gpu'][self.ids[1]].update(
            mode='reduced', wait_seconds=60, reduction_rule='reserve')
        self.core['gpu'][self.ids[2]].update(
            mode='waiting_hot', wait_seconds=87, reduction_rule='regular')
        self.core['cooling'].update(gpu_max=75, gpu_fan_threshold_percent=73)
        self.write_status()

        rc, text = self.command()
        self.assertEqual(rc, 0)
        self.assertIn('ожидание резервного снижения; осталось 35 с', text)
        self.assertIn('резервное снижение; осталось 60 с', text)
        self.assertIn('ожидание снижения; осталось 87 с', text)
        self.assertIn('Максимум вентиляторов GPU: 75%', text)

        rc, raw = self.command(['--json'])
        self.assertEqual(rc, 0)
        status = json.loads(raw)['core']['status']
        self.assertEqual(status['gpu'][self.ids[0]], self.core['gpu'][self.ids[0]])
        self.assertEqual(status['cooling'], self.core['cooling'])

    def test_autofan_error_keeps_case_fan_and_reserve_status(self):
        self.core['gpu'][self.ids[0]].update(
            mode='waiting_hot', wait_seconds=15, reduction_rule='reserve')
        self.core['cooling'].update(
            case_pwm=59, case_max=59, case_at_max=True,
            gpu_max=None, gpu_fan_threshold_percent=None)
        self.core['errors'] = ['AutoFan: configuration unavailable']
        self.write_status()

        rc, text = self.command()
        self.assertEqual(rc, 1)
        self.assertIn('ожидание резервного снижения; осталось 15 с', text)
        self.assertIn('Максимум вентиляторов GPU: —', text)
        self.assertIn('AutoFan: configuration unavailable', text)
        self.assertNotIn('Максимум вентиляторов GPU недоступен', text)

        rc, raw = self.command(['--json'])
        self.assertEqual(rc, 1)
        core = json.loads(raw)['core']
        self.assertEqual(core['status']['cooling'], self.core['cooling'])
        self.assertEqual(core['warnings'], self.core['errors'])

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

    def test_thermal_limit_is_a_warning_in_both_output_modes(self):
        self.case['target_unmet_at_limit'] = True
        self.write_status()
        self.assertEqual(self.command()[0], 1)
        rc, raw = self.command(['--json'])
        self.assertEqual(rc, 1)
        self.assertIn('целевая температура не удерживается',
                      json.loads(raw)['case']['warnings'][0])

    def test_emergency_warns_when_live_pwm_does_not_confirm_full_speed(self):
        self.case.update(mode='emergency')
        self.write_status()
        for mode, message in ((1, 'фактический PWM ниже 100%'),
                              (2, 'фактический PWM не подтверждён')):
            with self.subTest(mode=mode):
                self.hw['pwm_mode'] = mode
                rc, output = self.command()
                self.assertEqual(rc, 1)
                self.assertIn(f'Аварийный режим, {message}', output)
                self.assertNotIn('Аварийное охлаждение 100%', output)

    def test_corrupt_status_keeps_other_service_and_emits_strict_json(self):
        for field in ('1e9999', '9' * 400, '[' * 1100 + '0' + ']' * 1100, 'NaN',
                      '100,"updated_at":101'):
            with self.subTest(prefix=field[:15]):
                self.core_path.write_text('{"updated_at":' + field + '}')
                rc, raw = self.command(['--json'])
                self.assertEqual(rc, 1)
                report = json.loads(raw, parse_constant=lambda name: self.fail(name))
                self.assertFalse(report['core']['current'])
                self.assertTrue(report['case']['current'])
                self.assertEqual(report['case']['pwm']['percent'], 58.8)

    def test_malformed_gpu_row_and_duplicate_indices_keep_healthy_gpu_readings(self):
        self.core['gpu'][self.ids[0]].update(mode=[], wait_seconds=-1)
        self.core['gpu'][self.ids[1]] = None
        self.core['gpu'][self.ids[2]]['index'] = 3
        self.write_status()
        rc, output = self.command()
        self.assertEqual(rc, 1)
        self.assertIn(f"[{self.ids[1][-8:]}]: — °C", output)
        self.assertIn(f"GPU ? [{self.ids[3][-8:]}]: 62 °C", output)
        self.assertNotIn('осталось -1 с', output)
        report = json.loads(self.command(['--json'])[1])
        self.assertIn('Повторяющийся индекс GPU 3', report['core']['warnings'])
        self.assertEqual(report['core']['status']['gpu'][self.ids[3]]['temperature_c'], 62)

    def test_fresh_status_with_corrupt_pid_or_wait_does_not_look_healthy(self):
        self.core['gpu'][self.ids[0]]['wait_seconds'] = None
        self.case['controller_pid'] = 10.0
        self.write_status()
        rc, raw = self.command(['--json'])
        self.assertEqual(rc, 1)
        report = json.loads(raw)
        self.assertFalse(report['case']['current'])
        self.assertFalse(report['case']['settings_confirmed'])
        self.assertTrue(report['core']['current'])
        self.assertIn('wait_seconds недоступно', report['core']['warnings'][0])

    def test_unprintable_diagnostics_do_not_break_text_or_json_output(self):
        diagnostic = 'driver error\n\x1b\ud800'
        self.core['errors'] = [diagnostic, None]
        self.write_status()
        rc, output = self.command()
        self.assertEqual(rc, 1)
        self.assertIn(r'driver error\n\x1b\ud800', output)
        self.assertNotIn('\x1b', output)
        rc, output = self.command(['--json'])
        self.assertEqual(rc, 1)
        output.encode('utf-8')
        self.assertEqual(json.loads(output)['core']['warnings'],
                         [diagnostic, 'Список ошибок в статусе повреждён'])

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
