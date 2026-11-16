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

import os
import random
import tempfile
import unittest

import pandas as pd

from main import CustomConfig
from tests._room import Room
from tools import calibrate

T0 = pd.Timestamp("2026-01-15T08:00:00Z")

# every scenario opens the window at minute 120
OPEN_AT = 120


def walk_with_mixed_gaps(minutes=3000, step_size=0.2, pattern=(1, 1, 1, 1, 1, 1, 3, 12), seed=1):
    """A series drifting by step_size per minute, read at gaps that repeat `pattern`, the way a sensor reporting
    on change leaves them."""
    rnd = random.Random(seed)
    value, walk = 10.0, [10.0]
    for _ in range(minutes):
        value += rnd.gauss(0, step_size)
        walk.append(value)
    times, values, minute, step = [], [], 0, 0
    while minute < len(walk):
        times.append(T0 + pd.Timedelta(minute, "min"))
        values.append(walk[minute])
        minute += pattern[step % len(pattern)]
        step += 1
    return times, values


class CalibrateTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = directory.name

    def history(self, rows):
        """(time, Temperature, Humidity) rows as the CSV the calibrator reads, read back in."""
        path = os.path.join(self.directory, "history.csv")
        pd.DataFrame(rows, columns=["time", "Temperature", "Humidity"]).to_csv(path, index=False)
        return calibrate.read_history(path)

    def room_history(self, room, minutes, **events):
        return self.history([(T0 + pd.Timedelta(minute, "min"), temperature, humidity)
                             for minute, temperature, humidity in room.readings(minutes, **events)])

    def drifting_history(self, minutes=3000, step_size=0.05, seed=1):
        """A history at a steady temperature whose humidity level wanders, as a room's does over hours."""
        rnd = random.Random(seed)
        value, rows = 50.0, []
        for minute in range(minutes):
            value += rnd.gauss(0, step_size)
            rows.append((T0 + pd.Timedelta(minute, "min"), 21.0, value))
        return self.history(rows)


class TestReadingTheHistory(CalibrateTestCase):
    def test_holds_a_reading_until_the_next_one_arrives(self):
        history = self.history([(T0, 21.0, 50.0), (T0 + pd.Timedelta(1, "min"), 21.4, None)])

        self.assertEqual(len(history), 2)
        self.assertEqual(history["Humidity"].iloc[1], 50.0)

    def test_drops_a_reading_older_than_the_operator_holds_it(self):
        history = self.history([(T0, 21.0, 50.0), (T0 + pd.Timedelta(200, "min"), 21.4, None)])

        self.assertEqual(len(history), 1)

    def test_refuses_a_history_without_the_operators_column_names(self):
        path = os.path.join(self.directory, "wrong.csv")
        pd.DataFrame({"time": [T0], "temp": [21.0]}).to_csv(path, index=False)

        with self.assertRaises(ValueError):
            calibrate.read_history(path)


class TestScaleBySpan(CalibrateTestCase):
    """Report 1: the scale the operator decides on, measured over the spans a drop is measured across."""

    def test_the_scale_grows_with_the_span(self):
        report = calibrate.scale_by_span(self.drifting_history(), spans=(10.0, 30.0))

        over_10, over_30 = report["scale"]
        self.assertGreater(over_30 / over_10, 1.4)
        self.assertLess(over_30 / over_10, 2.3)

    def test_an_hour_of_its_own_carries_a_scale_too(self):
        # what a deployed operator has; the hours too short for the span are counted rather than left out
        report = calibrate.scale_by_span(self.drifting_history(), spans=(30.0,))

        self.assertGreater(report["hours_measured"].iloc[0], 40)
        self.assertGreater(report["median_scale_per_hour"].iloc[0], 0)

    def test_a_room_whose_humidity_never_moves_has_no_scale(self):
        still = self.history([(T0 + pd.Timedelta(minute, "min"), 21.0, 50.0) for minute in range(600)])

        report = calibrate.scale_by_span(still, spans=(30.0,))

        self.assertEqual(report["scale"].iloc[0], 0.0)


class TestScaleByGap(CalibrateTestCase):
    """Report 1b, the diagnostic that refuted normalising a change by the square root of its own gap."""

    def setUp(self):
        super().setUp()
        self.report = calibrate.scale_by_gap(*walk_with_mixed_gaps())

    def test_the_scale_per_square_root_of_a_minute_is_the_same_across_gap_lengths(self):
        scales = self.report["scale_per_sqrt_min"]

        self.assertGreater(len(scales), 2, "the pattern must fill at least three ranges of gap lengths")
        self.assertLess(scales.max() / scales.min(), 1.5)

    def test_the_raw_scale_of_a_drifting_series_grows_with_the_square_root_of_the_gap(self):
        """A series that really does move while nobody reads it. A room read by a sensor that reports on change
        does not: there the exponent comes out near zero, which is what refuted the normalisation."""
        self.assertAlmostEqual(calibrate.gap_exponent(self.report), 0.5, delta=0.1)


class TestFindingAirings(CalibrateTestCase):
    def airings(self, room, minutes=360, **events):
        return calibrate.find_airings(calibrate.minute_grid(self.room_history(room, minutes, **events)))

    def test_finds_the_airing_of_a_winter_day_where_it_began(self):
        airings = self.airings(Room(0, 80), openings=[(OPEN_AT, 10)])

        self.assertEqual(len(airings), 1)
        self.assertAlmostEqual((airings["start"].iloc[0] - T0).total_seconds() / 60, OPEN_AT, delta=5)

    def test_finds_no_airing_in_a_quiet_day(self):
        self.assertEqual(len(self.airings(Room(0, 80))), 0)

    def test_a_shower_ending_is_no_airing(self):
        # humidity falls just as far afterwards, but back to below where it started rather than up to it
        self.assertEqual(len(self.airings(Room(0, 80), moisture_sources=[(OPEN_AT, 15, 0.3, 0.05)])), 0)


class TestSweep(CalibrateTestCase):
    def setUp(self):
        super().setUp()
        self.config = CustomConfig({})
        history = self.room_history(Room(0, 80), 360, openings=[(OPEN_AT, 10)])
        self.statistics = calibrate.replay(history, self.config)
        self.airings = calibrate.find_airings(calibrate.minute_grid(history))

    def test_a_larger_factor_never_raises_the_number_of_alarms(self):
        report = calibrate.sweep(self.statistics, self.airings, [0.5, 1, 2, 4, 8, 20], drop_min=0.0)

        self.assertEqual(list(report["alarms"]), sorted(report["alarms"], reverse=True))

    def test_catches_the_airing_at_the_factor_the_operator_ships_with(self):
        report = calibrate.sweep(self.statistics, self.airings,
                                 [self.config.ah_drop_std_factor], self.config.ah_drop_min)

        self.assertEqual(report["airings_caught"].iloc[0], 1)
        self.assertLessEqual(report["median_latency_min"].iloc[0], 10)

    def test_recommends_no_factor_when_none_of_the_candidates_catches_enough(self):
        report = calibrate.sweep(self.statistics, self.airings, [1000.0], self.config.ah_drop_min)

        catching, _ = calibrate.recommend(report, target_recall=0.8, false_alarm_budget=1.0)
        self.assertIsNone(catching)


if __name__ == "__main__":
    unittest.main()
