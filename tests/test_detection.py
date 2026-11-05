"""
   Copyright (c) 2026 InfAI (CC SES)

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import tempfile
import unittest

from tests._harness import at, feed, new_operator, transitions
from tests._room import Room

# every scenario opens the window at minute 120 unless it says otherwise
OPEN_AT = 120


class DetectionTestCase(unittest.TestCase):
    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.data_path = data_dir.name
        self.operator, _ = new_operator(self.data_path)

    def run_scenario(self, room, minutes, **events):
        return feed(self.operator, room.readings(minutes, **events))


class TestWinterAiring(DetectionTestCase):
    """0 °C and 80 % outside: humidity falls within minutes, temperature only slowly."""

    def setUp(self):
        super().setUp()
        self.outputs = self.run_scenario(Room(0, 80), 240, openings=[(OPEN_AT, 10)])

    def test_reports_the_window_open_within_5_minutes(self):
        minute, is_open = transitions(self.outputs)[0]

        self.assertTrue(is_open)
        self.assertLessEqual(minute, OPEN_AT + 5)

    def test_reports_the_window_closed_within_10_minutes_of_closing(self):
        minute, is_open = transitions(self.outputs)[-1]

        self.assertFalse(is_open)
        self.assertGreaterEqual(minute, OPEN_AT + 10)
        self.assertLessEqual(minute, OPEN_AT + 10 + 10)

    def test_reports_one_opening_and_one_closing(self):
        self.assertEqual([is_open for _, is_open in transitions(self.outputs)], [True, False])


class TestLongAiring(DetectionTestCase):
    """A tilted window: humidity and temperature settle at a lower level without recovering."""

    def readings(self):
        return Room(0, 80, tau_humidity_open=20, tau_temperature_open=200).readings(360, openings=[(OPEN_AT, 120)])

    def test_stays_open_while_the_window_is_tilted(self):
        changes = transitions(feed(self.operator, self.readings()))

        self.assertEqual([is_open for _, is_open in changes], [True, False])
        self.assertGreaterEqual(changes[-1][0], OPEN_AT + 120)

    def test_counts_the_window_closed_after_max_open_minutes(self):
        operator, _ = new_operator(self.data_path + "/limited", max_open_minutes="60")

        changes = transitions(feed(operator, self.readings()))

        opened, closed = changes[0][0], changes[1][0]
        self.assertEqual(closed - opened, 60)


class TestFastClosing(DetectionTestCase):
    def test_the_temperature_drop_of_the_last_airing_does_not_reopen_the_window(self):
        # humidity returns so fast that the closing shows while the room is still cold
        outputs = self.run_scenario(Room(0, 80, humidity=80, tau_humidity_recovery=3), 240, openings=[(OPEN_AT, 10)])

        self.assertEqual([is_open for _, is_open in transitions(outputs)], [True, False])

    def test_the_humidity_drop_of_the_last_airing_does_not_reopen_the_window(self):
        # the temperature shows the closing while humidity is still far below its level from before the airing
        outputs = self.run_scenario(Room(-5, 80, tau_temperature_open=30), 240, openings=[(OPEN_AT, 5)])

        self.assertEqual([is_open for _, is_open in transitions(outputs)], [True, False])


class TestRepeatedAiring(DetectionTestCase):
    def test_detects_a_second_airing_10_minutes_after_the_first(self):
        second = OPEN_AT + 20
        outputs = self.run_scenario(Room(0, 80), 300, openings=[(OPEN_AT, 10), (second, 10)])

        changes = transitions(outputs)
        self.assertEqual([is_open for _, is_open in changes], [True, False, True, False])
        self.assertLessEqual(changes[2][0], second + 5)


class TestMildWeatherAiring(DetectionTestCase):
    def test_detects_the_opening_by_the_temperature_drop_when_humidity_hardly_changes(self):
        # 12 °C and 85 % outside hold about as much water as the room: only the temperature shows the airing
        outputs = self.run_scenario(Room(12, 85), 240, openings=[(OPEN_AT, 15)])

        minute, is_open = transitions(outputs)[0]
        self.assertTrue(is_open)
        self.assertLessEqual(minute, OPEN_AT + 15)


class TestNoAiring(DetectionTestCase):
    def test_sensor_noise_opens_no_window(self):
        outputs = self.run_scenario(Room(0, 80), 8 * 60)

        self.assertEqual(transitions(outputs), [])

    def test_a_shower_opens_no_window(self):
        # 15 minutes of shower: humidity and temperature rise, then the room's ventilation removes the moisture
        outputs = self.run_scenario(Room(0, 80), 240, moisture_sources=[(OPEN_AT, 15, 0.3, 0.05)])

        self.assertEqual(transitions(outputs), [])

    def test_no_temperature_readings_open_no_window(self):
        for minute, _, humidity in Room(0, 80).readings(240, openings=[(OPEN_AT, 10)]):
            self.operator.run(data={"Humidity": humidity}, selector=None, device_id="device:1", timestamp=at(minute))

        self.assertFalse(self.operator.window_open)


class TestHumidityRebound(DetectionTestCase):
    def rebounds(self, outputs):
        return [minute for minute, output in outputs if output["humidity_too_fast_too_high"]]

    def test_cold_air_right_after_closing_is_no_rebound(self):
        # mild, saturated air outside: the cooled room shows a high relative humidity, but holds little water
        outputs = self.run_scenario(Room(12, 100), 240, openings=[(OPEN_AT, 15)])

        self.assertTrue(transitions(outputs), "the scenario must open the window")
        self.assertEqual(self.rebounds(outputs), [])

    def test_reports_a_rebound_when_moisture_returns_fast(self):
        # a humid room whose walls give the moisture back within minutes after airing
        outputs = self.run_scenario(Room(0, 80, humidity=80, tau_humidity_recovery=3), 240, openings=[(OPEN_AT, 10)])

        self.assertTrue(self.rebounds(outputs))


class TestRestart(DetectionTestCase):
    def test_an_open_window_stays_open_across_a_restart(self):
        readings = Room(0, 80).readings(240, openings=[(OPEN_AT, 10)])
        feed(self.operator, readings[:OPEN_AT + 8])
        self.assertTrue(self.operator.window_open, "the window must be open before the restart")
        self.operator.stop()

        restarted, _ = new_operator(self.data_path)

        self.assertTrue(restarted.window_open)


class TestBaselineAfterAiring(DetectionTestCase):
    def test_baseline_does_not_reopen_the_window_while_humidity_recovers(self):
        # 26 h give the moving mean its baseline before the airing
        start = 26 * 60
        outputs = self.run_scenario(Room(0, 80), start + 240, openings=[(start, 10)])

        self.assertIsNotNone(self.operator.movingMean_feature.current_mean, "the baseline must be active")
        self.assertEqual([is_open for _, is_open in transitions(outputs)], [True, False])


if __name__ == "__main__":
    unittest.main()
