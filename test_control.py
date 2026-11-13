import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('control', HERE / 'cryptone-fan-control.py')
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = control.load_config(HERE / 'cryptone-fan-control.json')
        # Keep the original restricted-range safety scenarios as regressions.
        self.cfg['max_pwm_percent'] = 55
        self.ids = self.cfg['expected_gpu_uuids']

    def csv(self, values):
        return '\n'.join(f'{uuid}, {temp}' for uuid, temp in zip(self.ids, values))

    def test_missing_duplicate_and_invalid_gpus_fail_closed(self):
        good = self.csv([64, 65, 66, 65])
        self.assertEqual(max(control.parse_temperatures(good, self.ids).values()), 66)
        bad_outputs = [self.csv([64, 65, 66]), good + '\n' + good.splitlines()[0],
                       self.csv(['N/A', 65, 66, 65]), self.csv([-1, 65, 66, 65]),
                       self.csv([121, 65, 66, 65]), self.csv(['nan', 65, 66, 65]), '']
        for output in bad_outputs:
            with self.subTest(output=output), self.assertRaises(ValueError):
                control.parse_temperatures(output, self.ids)

    def test_replaced_gpu_is_not_silently_accepted(self):
        output = self.csv([65] * 4).replace(self.ids[0], 'GPU-unexpected')
        with self.assertRaises(ValueError):
            control.parse_temperatures(output, self.ids)

    def test_emergency_threshold_and_sustained_recovery(self):
        c = control.Controller(self.cfg)
        self.assertEqual(c.update(75, 0), 100)
        self.assertEqual(c.update(71, 5), 100)
        self.assertEqual(c.update(70, 10), 100)
        self.assertEqual(c.update(70, 15), 100)
        self.assertEqual(c.update(70, 20), 55)
        self.assertFalse(c.emergency)
        self.assertEqual(c.update(None, 25), 100)
        self.assertEqual(c.update(65, 30), 100)
        self.assertEqual(c.update(None, 35), 100)
        self.assertEqual(c.update(65, 40), 100)
        self.assertEqual(c.update(65, 45), 100)
        self.assertEqual(c.update(65, 50), 55)

    def test_limits_deadband_and_ramp_timing(self):
        c = control.Controller(self.cfg)
        self.assertEqual(c.update(60, 0), 55)
        self.assertEqual(c.update(60, 15), 55)
        self.assertEqual(c.update(60, 20), 54)
        c.current = 40
        self.assertEqual(c.update(68, 25), 40)
        self.assertEqual(c.update(68, 30), 42)
        for temp in (64, 65, 66):
            self.assertEqual(c.update(temp, 100), 42)
        for now in range(120, 1200, 20):
            c.update(20, now)
        self.assertEqual(c.current, 30)
        for now in range(1210, 1700, 10):
            c.update(74, now)
        self.assertEqual(c.current, 55)

    def test_pwm_quantization_stays_inside_approved_bounds(self):
        for percent in range(30, 56):
            duty = control.pwm_value(percent, self.cfg)
            self.assertGreaterEqual(duty * 100 / 255, 30)
            self.assertLessEqual(duty * 100 / 255, 55)
        self.assertEqual(control.pwm_value(100, self.cfg), 255)

    def test_bad_configuration_is_rejected(self):
        for key, value in [('min_pwm_percent', 0), ('max_pwm_percent', 101),
                           ('target_c', 75), ('poll_seconds', 20),
                           ('critical_c', float('nan')), ('expected_gpu_uuids', [])]:
            cfg = dict(self.cfg, **{key: value})
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(cfg))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    control.load_config(path)


if __name__ == '__main__':
    unittest.main(verbosity=2)
