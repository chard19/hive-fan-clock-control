"""Thermal and failure scenarios with fake time/hardware; no GPU writes."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import core_control as core

IDS = core.load_config(Path(__file__).with_name('cryptone-core-control.toml')).uuids
S = core.Settings()


class RegulationTests(unittest.TestCase):
    def step(self, state, now, temp=68, maxed=True, base=2700, locked=None, settings=S,
             growth=True, **cooling):
        if locked is None:
            locked = state.current if state.current is not None else 2700
        return core.decide(state, settings, base, locked, temp, maxed, now, growth, **cooling)

    def test_three_minutes_continuous_and_repeated_steps(self):
        state = core.GPUState()
        self.assertEqual(self.step(state, 0), (2700, 'waiting_hot', 180))
        self.assertEqual(self.step(state, 179)[0], 2700)
        self.assertEqual(self.step(state, 180)[0], 2650)
        self.assertEqual(self.step(state, 359)[0], 2650)
        self.assertEqual(self.step(state, 360)[0], 2600)

    def test_temperature_interruption_restarts_timer(self):
        state = core.GPUState()
        self.step(state, 0)
        self.step(state, 175, temp=67)
        self.assertEqual(self.step(state, 180)[2], 180)
        self.assertEqual(self.step(state, 359)[0], 2700)
        self.assertEqual(self.step(state, 360)[0], 2650)

    def test_fans_must_reach_configured_maximum_continuously(self):
        state = core.GPUState()
        self.assertEqual(self.step(state, 0, maxed=False)[1], 'waiting_fans')
        self.step(state, 100, maxed=True)
        self.step(state, 200, maxed=False)
        self.assertEqual(self.step(state, 280, maxed=True)[0], 2700)
        self.assertEqual(self.step(state, 460)[0], 2650)

    def test_reserve_starts_at_70_and_repeats_fixed_steps_at_60_and_120_seconds(self):
        state = core.GPUState()
        settings = replace(S, step_mhz=30)
        options = dict(maxed=False, case_at_max=True, settings=settings)
        self.assertEqual(self.step(state, 0, temp=69, **options),
                         (2700, 'waiting_fans', None))
        self.assertEqual(self.step(state, 120, temp=69, **options)[0], 2700)
        self.assertEqual(self.step(state, 125, temp=70, **options),
                         (2700, 'waiting_hot', 60))
        self.assertEqual(state.reduction_rule, 'reserve')
        self.assertEqual(self.step(state, 184, temp=70, **options)[0], 2700)
        self.assertEqual(self.step(state, 185, temp=70, **options)[0], 2650)
        self.assertEqual(state.reduction_rule, 'reserve')
        self.assertEqual(self.step(state, 244, temp=70, **options)[0], 2650)
        self.assertEqual(self.step(state, 245, temp=70, **options)[0], 2600)

    def test_reserve_does_not_cross_floor_or_reuse_time_after_cooling(self):
        state = core.GPUState(current=2420)
        options = dict(maxed=False, case_at_max=True)
        self.step(state, 0, temp=70, **options)
        self.assertEqual(self.step(state, 60, temp=70, **options)[0], 2400)
        self.assertEqual(self.step(state, 120, temp=70, **options)[0], 2400)
        self.assertEqual(self.step(state, 125, temp=65, **options)[0], 2400)
        self.assertEqual(self.step(state, 304, temp=65, **options)[0], 2400)
        self.assertEqual(self.step(state, 305, temp=65, **options)[0], 2450)

    def test_hot_cards_keep_independent_reserve_start_times(self):
        first, second = core.GPUState(), core.GPUState()
        options = dict(maxed=False, case_at_max=True)
        for now in (0, 30, 60):
            self.step(second, now, temp=69 if now == 0 else 70, **options)
            self.step(first, now, temp=70, **options)
        self.assertEqual((first.current, second.current), (2650, 2700))
        self.assertEqual(self.step(second, 90, temp=70, **options)[0], 2650)
        self.assertEqual(self.step(first, 90, temp=70, **options)[0], 2650)

    def test_reserve_restarts_after_temperature_or_case_telemetry_interruption(self):
        for temp, case_at_max in ((69, True), (None, True), (70, False)):
            with self.subTest(temperature=temp, case_at_max=case_at_max):
                state = core.GPUState()
                self.step(state, 0, temp=70, maxed=False, case_at_max=True)
                self.step(state, 55, temp=temp, maxed=False, case_at_max=case_at_max)
                self.assertEqual(self.step(state, 60, temp=70, maxed=False,
                                           case_at_max=True)[2], 60)
                self.assertEqual(self.step(state, 120, temp=70, maxed=False,
                                           case_at_max=True)[0], 2650)

    def test_parallel_delays_apply_one_step_and_restart_both_timers(self):
        # A distinct regular step identifies which rule won, including when both are due.
        settings = replace(S, step_mhz=30)
        for reserve_start, due, expected_clock, rule in (
                (110, 170, 2650, 'reserve'),
                (120, 180, 2650, 'reserve'),
                (160, 180, 2670, 'regular')):
            with self.subTest(reserve_start=reserve_start, rule=rule):
                state = core.GPUState()
                options = dict(case_at_max=True, settings=settings)
                self.step(state, 0, temp=68, **options)
                self.step(state, reserve_start, temp=70, **options)
                self.assertEqual(self.step(state, due - 1, temp=70, **options)[0], 2700)
                self.assertEqual(self.step(state, due, temp=70, **options)[0], expected_clock)
                self.assertEqual(state.reduction_rule, rule)
                self.assertEqual(self.step(state, due + 59, temp=70, **options)[0], expected_clock)
                self.assertEqual(self.step(state, due + 60, temp=70, **options)[0],
                                 expected_clock - 50)

    def test_emergency_outranks_a_reserve_step_due_at_the_same_poll(self):
        state = core.GPUState()
        self.step(state, 0, temp=70, maxed=False, case_at_max=True)
        self.assertEqual(self.step(state, 60, temp=75, maxed=False, case_at_max=True),
                         (2400, 'emergency', None))

    def test_case_maximum_change_restarts_both_delays_even_if_both_remain_at_maximum(self):
        for temp, hold in ((68, 180), (70, 60)):
            with self.subTest(temperature=temp):
                state = core.GPUState()
                options = dict(temp=temp, case_at_max=True, gpu_max=90)
                self.step(state, 0, case_max=70, **options)
                self.assertEqual(self.step(state, hold - 5, case_max=75, **options)[2], hold)
                self.assertEqual(self.step(state, hold, case_max=75, **options)[0], 2700)
                self.assertEqual(self.step(state, 2 * hold - 5, case_max=75, **options)[0], 2650)

    def test_gpu_maximum_change_restarts_only_regular_delay(self):
        normal, reserve = core.GPUState(), core.GPUState()
        self.step(normal, 0, case_at_max=True, gpu_max=90)
        self.assertEqual(self.step(normal, 175, case_at_max=True, gpu_max=75)[2], 180)
        self.assertEqual(self.step(normal, 180, case_at_max=True, gpu_max=75)[0], 2700)
        self.assertEqual(self.step(normal, 355, case_at_max=True, gpu_max=75)[0], 2650)
        self.step(reserve, 0, temp=70, case_at_max=True, gpu_max=90)
        self.assertEqual(self.step(reserve, 55, temp=70, case_at_max=True, gpu_max=75)[2], 5)
        self.assertEqual(self.step(reserve, 60, temp=70, case_at_max=True, gpu_max=75)[0], 2650)

    def test_cards_are_independent_and_uuid_survives_reordering(self):
        states = {u: core.GPUState() for u in IDS}
        for now in (0, 180):
            for uuid in reversed(IDS):
                self.step(states[uuid], now, temp=68 if uuid == IDS[0] else 65)
        self.assertEqual([states[u].current for u in IDS], [2650, 2700, 2700, 2700])

    def test_recovery_requires_three_minutes_at_65(self):
        state = core.GPUState(current=2500)
        self.assertEqual(self.step(state, 0, temp=65)[2], 180)
        self.step(state, 175, temp=66)
        self.step(state, 180, temp=65)
        self.assertEqual(self.step(state, 359, temp=65)[0], 2500)
        self.assertEqual(self.step(state, 360, temp=65)[0], 2550)
        self.assertEqual(self.step(state, 540, temp=65)[0], 2600)

    def test_never_exceeds_hive_ceiling_including_above_old_cap(self):
        for base in (2400, 2420, 2700, 2850, 2900, 3000):
            state = core.GPUState(current=2650)
            for now in range(0, 2200, 180):
                self.step(state, now, temp=64, base=base)
                self.assertLessEqual(state.current, base)
            self.assertEqual(state.current, base)

    def test_each_card_recovers_to_its_own_hive_ceiling(self):
        baselines = dict(zip(IDS, (2900, 3000, 2750, 2850)))
        states = {u: core.GPUState(current=2700) for u in IDS}
        for now in range(0, 1800, 180):
            for uuid in reversed(IDS):
                self.step(states[uuid], now, temp=64, base=baselines[uuid])
                self.assertLessEqual(states[uuid].current, baselines[uuid])
        self.assertEqual({u: s.current for u, s in states.items()}, baselines)

    def test_hive_raise_above_2850_keeps_reduction_and_restarts_hold(self):
        state = core.GPUState(current=2800)
        self.step(state, 0, temp=64, base=2850)
        self.assertEqual(self.step(state, 175, temp=64, base=2900, locked=2900),
                         (2800, 'waiting_cool', 180))
        self.assertEqual(self.step(state, 350, temp=64, base=2900)[0], 2800)
        self.assertEqual(self.step(state, 355, temp=64, base=2900)[0], 2850)
        self.assertEqual(self.step(state, 535, temp=64, base=2900)[0], 2900)
        self.assertEqual(self.step(state, 540, temp=67, base=2750)[0], 2750)

    def test_partial_final_recovery_step_stops_at_hive_ceiling(self):
        state = core.GPUState(current=2850)
        self.step(state, 0, temp=64, base=2890)
        self.assertEqual(self.step(state, 180, temp=64, base=2890)[0], 2890)

    def test_higher_hive_ceiling_preserves_emergency_and_recovery(self):
        state = core.GPUState(current=2900)
        self.assertEqual(self.step(state, 0, temp=75, maxed=False, base=2900)[0], 2400)
        self.assertTrue(state.emergency)
        self.assertEqual(self.step(state, 180, temp=66, base=3000)[0], 2400)
        self.assertTrue(state.emergency)
        self.step(state, 200, temp=65, base=3000)
        self.assertEqual(self.step(state, 379, temp=65, base=3000)[0], 2400)
        self.assertEqual(self.step(state, 380, temp=65, base=3000)[0], 2450)
        for now in range(560, 2600, 180):
            self.step(state, now, temp=65, base=3000)
        self.assertEqual(state.current, 3000)

    def test_lower_bound_and_nondivisible_last_step(self):
        state = core.GPUState(current=2420)
        self.step(state, 0)
        self.assertEqual(self.step(state, 180)[0], 2400)
        self.assertEqual(self.step(state, 360)[1], 'minimum')

    def test_deadband_holds_restriction(self):
        state = core.GPUState(current=2600)
        self.assertEqual(self.step(state, 0, temp=66), (2600, 'holding', None))
        self.assertEqual(self.step(state, 500, temp=67)[0], 2600)

    def test_75_immediately_bypasses_fan_gate_and_timer(self):
        state = core.GPUState()
        self.assertEqual(self.step(state, 0, temp=75, maxed=False, growth=False),
                         (2400, 'emergency', None))
        self.assertTrue(state.emergency)
        self.assertEqual(self.step(state, 180, temp=74)[0], 2400)

    def test_emergency_exit_requires_continuous_cooling(self):
        state = core.GPUState()
        self.step(state, 0, temp=75)
        self.step(state, 10, temp=65)
        self.step(state, 100, temp=66)
        self.step(state, 200, temp=65)
        self.assertEqual(self.step(state, 379, temp=65)[0], 2400)
        self.assertEqual(self.step(state, 380, temp=65)[0], 2450)
        self.assertFalse(state.emergency)

    def test_emergency_config_change_does_not_raise_or_clear_latch(self):
        state = core.GPUState()
        self.step(state, 0, temp=75)
        newer = replace(S, min_core_mhz=2500)
        self.assertEqual(self.step(state, 20, temp=70, settings=newer)[0], 2400)
        self.assertTrue(state.emergency)
        self.assertEqual(self.step(state, 30, temp=76, settings=newer)[0], 2400)

    def test_hive_change_keeps_restriction_and_restarts_recovery(self):
        state = core.GPUState(current=2600)
        self.step(state, 0, temp=65)
        self.assertEqual(self.step(state, 170, temp=65, base=2850, locked=2850)[0], 2600)
        self.assertEqual(self.step(state, 180, temp=65, base=2850)[0], 2600)
        self.assertEqual(self.step(state, 350, temp=65, base=2850)[0], 2650)
        self.assertEqual(self.step(state, 355, base=2500)[0], 2500)

    def test_offset_only_hive_reapply_does_not_release_restriction(self):
        state = core.GPUState(current=2500)
        self.assertEqual(self.step(state, 0, temp=67, locked=2700)[0], 2500)

    def test_external_lower_clock_respected(self):
        state = core.GPUState(current=2600)
        self.assertEqual(self.step(state, 0, temp=67, locked=2450)[0], 2450)

    def test_hive_disabled_or_below_min_is_paused(self):
        for base in (None, 2300):
            state = core.GPUState(current=2600)
            self.assertEqual(self.step(state, 0, base=base)[1], 'paused_hive')
            self.assertEqual(state.current, 2600)

    def test_missing_temperature_or_locked_clock_prevents_growth(self):
        for temp, lock in ((None, 2500), (65, None), (65, 0)):
            state = core.GPUState(current=2500)
            target = core.decide(state, S, 2700, lock, temp, True, 0)[0]
            self.assertEqual(target, 2500 if lock == 2500 else None)
            self.assertEqual(state.current, 2500)
        state = core.GPUState(current=2500)
        self.step(state, 0, temp=65)
        self.assertEqual(self.step(state, 180, temp=65, growth=False)[0], 2500)
        self.assertEqual(self.step(state, 200, temp=65)[2], 180)

    def test_missing_temperature_reasserts_previous_restriction_after_hive_reset(self):
        state = core.GPUState(current=2500)
        self.assertEqual(self.step(state, 0, temp=None, locked=2700)[0], 2500)

    def test_missing_lock_still_latches_emergency_but_never_raises_clock(self):
        state = core.GPUState()
        self.assertEqual(core.decide(state, S, 2700, None, 75, False, 0, actual_core=2200),
                         (None, 'sensor_error', None))
        self.assertTrue(state.emergency)
        self.assertEqual(state.current, 2400)
        self.assertEqual(core.decide(state, S, 2700, 0, 73, False, 5, actual_core=2800),
                         (2400, 'emergency', None))
        self.assertEqual(core.decide(state, S, None, 0, 75, False, 10, actual_core=2800),
                         (None, 'paused_hive', None))

    def test_restore_reduced_and_emergency_state_on_reboot(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'state.json'
            original = {u: core.GPUState(current=2400 if i == 0 else 2700,
                           baseline=2700, emergency=i == 0) for i, u in enumerate(IDS)}
            core.atomic_json(p, core.persistent_data(original))
            restored = core.restore_states(core.load_config(
                Path(__file__).with_name('cryptone-core-control.toml')), p)
            self.assertEqual(self.step(restored[IDS[0]], 0, locked=2700, temp=66)[0], 2400)
            self.assertTrue(restored[IDS[0]].emergency)


class InputTests(unittest.TestCase):
    def test_validation_rejects_entire_invalid_override(self):
        original = Path(__file__).with_name('cryptone-core-control.toml').read_text()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'config.toml'
            p.write_text(original + f'\n[gpu."{IDS[0]}"]\nstep_mhz=0\n')
            with self.assertRaises(ValueError):
                core.load_config(p)
            p.write_text(original + f'\n[gpu."{IDS[0]}"]\nstep_mhz=60\n')
            cfg = core.load_config(p)
            self.assertEqual(cfg.for_gpu(IDS[0]).step_mhz, 60)
            self.assertEqual(cfg.for_gpu(IDS[1]).step_mhz, 50)

    def test_invalid_types_temperatures_and_ranges(self):
        for overrides in ({'step_mhz': True}, {'cool_c': 68}, {'min_core_mhz': 499},
                          {'min_core_mhz': 5001},
                          {'hot_hold_seconds': 0}, {'misspelled': 1}, {'emergency_c': 90}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                core.Settings.parse(overrides)

    def test_old_maximum_is_rejected_in_defaults_and_per_gpu(self):
        original = Path(__file__).with_name('cryptone-core-control.toml').read_text()
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / 'config.toml'
            for text in (original.replace('[defaults]', '[defaults]\nmax_core_mhz=2850', 1),
                         original + f'\n[gpu."{IDS[0]}"]\nmax_core_mhz=2800\n'):
                p.write_text(text)
                with self.assertRaisesRegex(ValueError, 'maximum comes only from HiveOS'):
                    core.load_config(p)

    def test_hive_bus_order_and_padding_match_stock_oc(self):
        gpus = {IDS[0]: {'bus': '02:00.0'}, IDS[1]: {'bus': '04:00.0'}}
        with tempfile.TemporaryDirectory() as folder:
            oc, detect, bus = (Path(folder) / n for n in ('oc', 'detect', 'bus'))
            oc.write_text('CLOCK="450 450"\nLCLOCK="2700 2850"\n')
            detect.write_text(json.dumps([{'busid': '04:00.0', 'brand': 'nvidia'},
                                          {'busid': '02:00.0', 'brand': 'nvidia'}]))
            result = core.hive_baselines(gpus, oc, detect, bus)
            self.assertEqual(result, {IDS[0]: 2850, IDS[1]: 2700})
            oc.write_text('CLOCK="450"\nLCLOCK="2750"\n')
            self.assertEqual(set(core.hive_baselines(gpus, oc, detect, bus).values()), {2750})
            oc.write_text('CLOCK="2800"\nLCLOCK="2700"\n')
            self.assertEqual(set(core.hive_baselines(gpus, oc, detect, bus).values()), {2800})

    def test_no_shell_execution_from_config(self):
        for text in ('LCLOCK="$(touch /tmp/no)"', 'LCLOCK="`id`"', 'LCLOCK=2700;id'):
            with self.assertRaises(ValueError):
                core.shell_values(text, {'LCLOCK'})

    def test_bus_formats_normalized(self):
        self.assertEqual(core.bus_key('00000000:02:00.0'), '0000:02:00.0')
        self.assertEqual(core.bus_key('0000:02:00.0'), '0000:02:00.0')
        self.assertNotEqual(core.bus_key('0000:02:00.0'), core.bus_key('0001:02:00.0'))

    def test_hive_lock_above_driver_command_range_is_an_input_error(self):
        gpus = {IDS[0]: {'bus': '02:00.0'}}
        with tempfile.TemporaryDirectory() as folder:
            oc, detect, bus = (Path(folder) / n for n in ('oc', 'detect', 'bus'))
            detect.write_text(json.dumps([{'busid': '02:00.0', 'brand': 'nvidia'}]))
            for value in ('LCLOCK=5001', 'CLOCK=5100', 'CLOCK=5000\nOHGODAPILL_ARGS="1"'):
                oc.write_text(value)
                with self.assertRaisesRegex(ValueError, 'exceeds the supported'):
                    core.hive_baselines(gpus, oc, detect, bus)

    def test_case_status_cannot_claim_automatic_mode_is_at_max(self):
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / 'case.json'
            for mode, speed in ((2, 70), (0, 70), (True, 70), (0, 100), (1, 70)):
                with self.subTest(mode=mode, speed=speed):
                    f.write_text(json.dumps(dict(updated_at=100, pwm_actual_percent=speed,
                                      applied_config={'max_pwm_percent': 70}, pwm_mode=mode)))
                    if type(mode) is int and (mode == 1 or speed == 100):
                        self.assertTrue(core.case_cooling(f, 105)['case_at_max'])
                    else:
                        with self.assertRaisesRegex(ValueError, 'not confirmed'):
                            core.case_cooling(f, 105)

    def test_non_object_core_state_does_not_escape_validation(self):
        cfg = core.load_config(Path(__file__).with_name('cryptone-core-control.toml'))
        with tempfile.TemporaryDirectory() as folder:
            f = Path(folder) / 'state.json'
            for state in ([], {'version': True, 'gpu': {}}, {'version': 1, 'gpu': {IDS[0]: None}}):
                f.write_text(json.dumps(state))
                with self.assertRaisesRegex(ValueError, 'Invalid persistent'):
                    core.restore_states(cfg, f)

    def test_invalid_gpu_sensor_does_not_discard_other_cards(self):
        output = (f'0, {IDS[0]}, 00000000:02:00.0, N/A, 2692, 90\n'
                  f'1, {IDS[1]}, 00000000:04:00.0, 68, 2692, 80\n')
        parsed = core.parse_gpus(output)
        self.assertIsNone(parsed[IDS[0]]['temperature_c'])
        self.assertEqual(parsed[IDS[1]]['temperature_c'], 68)

    def test_configured_cooling_limits_not_100_percent(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 53.1,
                            'applied_config': {'max_pwm_percent': 55}}))
            fan.write_text('ENABLED=1\nMAX_FAN=80\n')
            limits = core.cooling_limits(case, fan, wall_time=105)
            self.assertTrue(limits['case_at_max'])
            self.assertEqual(limits['gpu_max'], 80)
            self.assertEqual(limits['gpu_fan_threshold_percent'], 78)
            # Editing the actual Hive setting is reflected on the very next read.
            fan.write_text('ENABLED=1\nMAX_FAN=70\n')
            self.assertEqual(core.cooling_limits(case, fan, wall_time=105)['gpu_max'], 70)
            stale = core.cooling_limits(case, fan, wall_time=121)
            self.assertFalse(stale['case_at_max'])
            self.assertTrue(stale['case_error'])
            self.assertEqual(stale['gpu_max'], 70)

    def test_bad_or_individual_autofan_settings_keep_fresh_case_readings(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 70,
                            'applied_config': {'max_pwm_percent': 70}}))
            for config in ('ENABLED=0\nMAX_FAN=75\n', 'ENABLED=1\nMAX_FAN=invalid\n',
                           'ENABLED=1\nMAX_FAN=75\nCUSTOM_MODE=1\n'):
                with self.subTest(config=config):
                    fan.write_text(config)
                    limits = core.cooling_limits(case, fan, wall_time=105)
                    self.assertTrue(limits['case_at_max'])
                    self.assertEqual(limits['case_max'], 70)
                    self.assertIsNone(limits['gpu_fan_threshold_percent'])
                    self.assertTrue(limits['gpu_error'])

    def test_missing_card_in_case_controller_keeps_other_cards_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 100,
                            'applied_config': {'max_pwm_percent': 55},
                            'mode': 'emergency', 'error': 'Expected GPU set mismatch'}))
            fan.write_text('ENABLED=1\nMAX_FAN=80\n')
            self.assertTrue(core.cooling_limits(case, fan, wall_time=105)['case_at_max'])

    def test_actual_workload_clock_not_used_as_lock_readback(self):
        locks = core.parse_locks('DEVICE #0:\n GPU CLOCKS LOCKED: 2700 MHz\n'
                                ' MEM CLOCKS LOCKED: 7001 MHz\n')
        self.assertEqual(locks[0], 2700)

    def test_index_readback_rejects_changed_gpu_identity(self):
        before = {IDS[0]: {'index': 0, 'bus': '02:00.0'}}
        after = {IDS[1]: {'index': 0, 'bus': '04:00.0'}}
        with patch.object(core, 'command', return_value='DEVICE #0:\n GPU CLOCKS LOCKED: 2700 MHz\n'), \
             patch.object(core, 'read_gpus', return_value=after):
            with self.assertRaises(ValueError):
                core.read_locks(before)

    def test_only_graphics_lock_command_and_verified_uuid(self):
        commands = []
        with patch.object(core, 'command', side_effect=lambda args: commands.append(args) or ''), \
             patch.object(core, 'read_gpus', return_value={}), \
             patch.object(core, 'read_locks', return_value={IDS[0]: 2650}):
            self.assertEqual(core.apply_clock(IDS[0], 2650), 2650)
        self.assertEqual(commands, [['/usr/bin/nvidia-smi', '-i', IDS[0],
                                     '--lock-gpu-clocks', '2650']])

    def test_readback_mismatch_is_not_success(self):
        with patch.object(core, 'command', return_value=''), \
             patch.object(core, 'read_gpus', return_value={}), \
             patch.object(core, 'read_locks', return_value={IDS[0]: 2700}):
            with self.assertRaises(RuntimeError):
                core.apply_clock(IDS[0], 2650)

    def test_invalid_readback_cannot_verify_minimum_supported_lock(self):
        with patch.object(core, 'command'), patch.object(core, 'read_gpus'), \
             patch.object(core, 'read_locks', return_value={IDS[0]: 499}):
            with self.assertRaisesRegex(RuntimeError, 'Lock not confirmed'):
                core.apply_clock(IDS[0], 500)


class DaemonTests(unittest.TestCase):
    def exercise(self, folder, after_poll=None, bad_storage=False, bad_hive=False,
                 recover=False, initial_clock=2700, hive_clocks=None,
                 first_telemetry=None, cooling=None, polls=4, fail_first_apply=False,
                 hive_read=None, monotonic=None, before_apply=None, bad_growth_commit=False,
                 stored_state=None):
        path = Path(folder) / 'config.toml'
        text = Path(__file__).with_name('cryptone-core-control.toml').read_text()
        path.write_text(text.replace('seconds = 180', 'seconds = 5'))
        cfg = core.load_config(path)
        state_path = Path(folder) / 'state.json'
        locks = dict.fromkeys(IDS, initial_clock)
        if recover:
            locks[IDS[0]] = 2500
            core.atomic_json(Path(folder) / 'state.json', core.persistent_data(
                {u: core.GPUState(current=locks[u], baseline=2700) for u in IDS}))
        if stored_state is not None:
            state_path.write_text(stored_state)
        calls = []
        class Event:
            count = 0
            def is_set(self):
                return self.count >= polls
            def set(self):
                self.count = polls
            def wait(self, seconds):
                self.count += 1
                if after_poll:
                    after_poll(self.count, path)
        event = Event()
        original_write = core.atomic_json
        def apply(uuid, target):
            if before_apply:
                before_apply(uuid, target, state_path)
            calls.append((uuid, target))
            if fail_first_apply and len(calls) == 1:
                raise RuntimeError('simulated clock command failure')
            locks[uuid] = target
            return target
        def write(p, data):
            if bad_storage and Path(p).name == 'state.json':
                raise OSError('simulated disk full')
            if Path(p) == state_path and bad_growth_commit and calls and data['gpu'][IDS[0]]['current'] == 2550:
                raise OSError('simulated disk full after hardware confirmation')
            original_write(p, data)
        gpus = {u: dict(index=i, bus=f'{i+2:02x}:00.0', temperature_c=
                        (65 if recover else 75 if bad_hive else 68) if i == 0 else 65,
                        actual_core_mhz=2692, fan_percent=80) for i, u in enumerate(IDS)}
        if first_telemetry is not None:
            gpus[IDS[0]].update(first_telemetry)
        with patch.object(core.threading, 'Event', return_value=event), \
                 patch.object(core.signal, 'signal'), \
                 patch.object(core, 'notify'), \
                 patch.object(core.time, 'monotonic', side_effect=lambda: (
                     monotonic(event.count) if monotonic else event.count * 5)), \
                 patch.object(core, 'read_gpus', return_value=gpus), \
                 patch.object(core, 'read_locks', side_effect=lambda _: dict(locks)), \
                 patch.object(core, 'hive_baselines', side_effect=(hive_read or (
                                   ValueError('bad HiveOC') if bad_hive else
                                   lambda _: hive_clocks or dict.fromkeys(IDS, 2700)))), \
                 patch.object(core, 'cooling_limits', side_effect=cooling, return_value=dict(
                     case_pwm=55, case_max=55, gpu_max=80, case_at_max=True,
                     gpu_fan_threshold_percent=78, case_error=None, gpu_error=None)), \
                 patch.object(core, 'apply_clock', side_effect=apply), \
                 patch.object(core, 'atomic_json', side_effect=write):
            core.run(cfg, path, state_path, Path(folder) / 'runtime')
        return calls, json.loads((Path(folder) / 'runtime/status.json').read_text())

    def test_polling_applies_one_card_and_reports_confirmed_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder)
        self.assertEqual(calls, [(IDS[0], 2650), (IDS[0], 2600), (IDS[0], 2550)])
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2550)
        self.assertEqual(status['gpu'][IDS[1]]['locked_mhz'], 2700)

    def test_running_daemon_uses_lower_autofan_limit_on_next_poll(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 70,
                            'applied_config': {'max_pwm_percent': 70}}))
            fan.write_text('ENABLED=1\nMAX_FAN=90\n')
            def edit(count, _path):
                if count == 2:
                    fan.write_text('ENABLED=1\nMAX_FAN=75\n')
            reader = core.cooling_limits
            calls, status = self.exercise(folder, after_poll=edit, first_telemetry={'fan_percent': 73},
                                          cooling=lambda: reader(case, fan, wall_time=105))
        self.assertEqual(calls, [(IDS[0], 2650)])
        self.assertEqual(status['gpu'][IDS[0]]['reduction_rule'], 'regular')
        self.assertEqual(status['cooling']['gpu_max'], 75)
        self.assertEqual(status['cooling']['gpu_fan_threshold_percent'], 73)

    def test_running_daemon_reserve_works_without_autofan_or_gpu_fan_readings(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 70,
                            'applied_config': {'max_pwm_percent': 70}}))
            fan.write_text('ENABLED=0\nMAX_FAN=90\n')
            reader = core.cooling_limits
            calls, status = self.exercise(folder, polls=13,
                                          first_telemetry={'temperature_c': 70, 'fan_percent': None},
                                          cooling=lambda: reader(case, fan, wall_time=105))
        self.assertEqual(calls, [(IDS[0], 2650)])
        self.assertEqual(status['gpu'][IDS[0]]['reduction_rule'], 'reserve')
        self.assertEqual(status['gpu'][IDS[1]]['locked_mhz'], 2700)
        self.assertTrue(status['errors'])

    def test_running_daemon_does_not_treat_stale_full_case_pwm_as_permission_to_reduce(self):
        with tempfile.TemporaryDirectory() as folder:
            case, fan = (Path(folder) / n for n in ('case', 'fan'))
            case.write_text(json.dumps({'updated_at': 100, 'pwm_actual_percent': 70,
                            'applied_config': {'max_pwm_percent': 70}}))
            fan.write_text('ENABLED=1\nMAX_FAN=75\n')
            reader = core.cooling_limits
            calls, status = self.exercise(folder, polls=13, first_telemetry={'temperature_c': 70},
                                          cooling=lambda: reader(case, fan, wall_time=121))
        self.assertEqual(calls, [])
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2700)
        self.assertIsNone(status['gpu'][IDS[0]]['reduction_rule'])
        self.assertTrue(status['errors'])

    def test_running_daemon_applies_final_reserve_step_inside_readback_tolerance(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=26, initial_clock=2410,
                                          hive_clocks=dict.fromkeys(IDS, 2410),
                                          first_telemetry={'temperature_c': 70, 'fan_percent': 73})
        self.assertEqual(calls, [(IDS[0], 2400)])
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2400)

    def test_running_daemon_retries_unconfirmed_small_reserve_step(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=26, initial_clock=2410,
                                          hive_clocks=dict.fromkeys(IDS, 2410),
                                          first_telemetry={'temperature_c': 70, 'fan_percent': 73},
                                          fail_first_apply=True)
        self.assertEqual(calls, [(IDS[0], 2400), (IDS[0], 2400)])
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2400)

    def test_no_reserve_step_from_unobserved_interval_during_poll_stall(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=14,
                                          monotonic=lambda i: i * 5 + (60 if i >= 10 else 0),
                                          first_telemetry={'temperature_c': 70, 'fan_percent': 73})
        self.assertEqual(calls, [])
        self.assertEqual(status['gpu'][IDS[0]]['wait_seconds'], 45)

    def test_daemon_emergency_can_reduce_from_known_high_workload_with_no_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, initial_clock=0,
                                          first_telemetry={'temperature_c': 75})
        self.assertEqual(calls, [(IDS[0], 2400)])
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2400)

    def test_recheck_prevents_growth_when_hive_ceiling_changes_before_write(self):
        def hive_read(gpus):
            return dict.fromkeys(gpus, 2400 if len(gpus) == 1 else 2700)

        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=2, recover=True, hive_read=hive_read)
            saved = json.loads((Path(folder) / 'state.json').read_text())
        self.assertEqual(calls, [])
        self.assertEqual(saved['gpu'][IDS[0]]['current'], 2500)
        self.assertIn('Hive clock profile changed', status['gpu'][IDS[0]]['error'])

    def test_failed_recovery_keeps_the_conservative_cap_on_disk(self):
        def before_apply(uuid, target, path):
            self.assertEqual(json.loads(path.read_text())['gpu'][uuid]['current'], 2500)
            self.assertEqual(target, 2550)

        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=2, recover=True, fail_first_apply=True,
                                          before_apply=before_apply)
            saved = json.loads((Path(folder) / 'state.json').read_text())
        self.assertEqual(calls, [(IDS[0], 2550)])
        self.assertEqual(saved['gpu'][IDS[0]]['current'], 2500)
        self.assertEqual(status['gpu'][IDS[0]]['target_mhz'], 2500)
        self.assertEqual(status['gpu'][IDS[0]]['mode'], 'apply_error')

    def test_post_write_disk_error_restores_earlier_hardware_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, polls=3, recover=True, bad_growth_commit=True)
            saved = json.loads((Path(folder) / 'state.json').read_text())
        self.assertEqual(calls, [(IDS[0], 2550), (IDS[0], 2500)])
        self.assertEqual(saved['gpu'][IDS[0]]['current'], 2500)
        self.assertEqual(status['gpu'][IDS[0]]['locked_mhz'], 2500)

    def test_corrupt_saved_state_preserves_good_rows_and_protects_other_cards(self):
        stored = json.dumps({'version': 1, 'gpu': {IDS[0]: None,
                            IDS[1]: {'current': 2500, 'baseline': 2700}}})
        with tempfile.TemporaryDirectory() as folder:
            with self.assertLogs(core.LOG, level='ERROR'):
                calls, status = self.exercise(folder, stored_state=stored,
                                              first_telemetry={'temperature_c': 75})
            self.assertEqual((Path(folder) / 'state.json').read_text(), stored)
        self.assertEqual(calls, [(IDS[0], 2400), (IDS[1], 2500)])
        self.assertEqual(status['gpu'][IDS[0]]['mode'], 'emergency')
        self.assertIn('Invalid persistent state', status['errors'][0])

    def test_daemon_follows_individual_hive_limits_above_and_below_old_cap(self):
        baselines = dict(zip(IDS, (2900, 3000, 2750, 2850)))
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, recover=True, initial_clock=2850,
                                          hive_clocks=baselines)
        self.assertEqual([v for u, v in calls if u == IDS[1]], [2900, 2950, 3000])
        self.assertEqual([v for u, v in calls if u == IDS[2]], [2750])
        self.assertEqual(status['gpu'][IDS[1]]['locked_mhz'], 3000)
        self.assertEqual(status['gpu'][IDS[2]]['locked_mhz'], 2750)
        self.assertFalse(status['errors'])
        self.assertIsNone(status['config_error'])

    def test_live_reload_then_invalid_config_keeps_last_valid_settings(self):
        def edit(count, path):
            if count == 1:
                path.write_text(path.read_text().replace('step_mhz = 50', 'step_mhz = 60'))
            if count == 2:
                path.write_text(path.read_text().replace('step_mhz = 60', 'step_mhz = 0'))
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, after_poll=edit)
        self.assertEqual(calls, [(IDS[0], 2640), (IDS[0], 2580)])
        self.assertIsNotNone(status['config_error'])
        self.assertEqual(status['gpu'][IDS[0]]['settings']['step_mhz'], 60)

    def test_failed_persistence_never_raises_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, bad_storage=True, recover=True)
        self.assertEqual(calls, [])
        self.assertEqual(status['gpu'][IDS[0]]['mode'], 'storage_error')

    def test_disk_full_does_not_block_75_degree_emergency(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, bad_storage=True, bad_hive=True)
        self.assertEqual(calls, [(IDS[0], 2400)])
        self.assertTrue(status['gpu'][IDS[0]]['emergency'])

    def test_unreadable_hive_oc_does_not_block_emergency_reduction(self):
        with tempfile.TemporaryDirectory() as folder:
            calls, status = self.exercise(folder, bad_hive=True)
        self.assertEqual(calls, [(IDS[0], 2400)])
        self.assertTrue(status['gpu'][IDS[0]]['emergency'])


if __name__ == '__main__':
    unittest.main()
