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

import unittest

import pandas as pd

from algo import utils

T0 = pd.Timestamp("2026-01-15T08:00:00Z")


def window(values, start=T0, step_seconds=60):
    return [{"timestamp": start + pd.Timedelta(i * step_seconds, "s"), "value": v} for i, v in enumerate(values)]


class TestSlope(unittest.TestCase):
    def test_slope_of_a_linear_rise_is_its_rate_per_minute(self):
        rising = window([50 + 0.1 * i for i in range(10)])

        self.assertAlmostEqual(utils.compute_n_min_slope(rising, 10), 0.1)

    def test_slope_covers_only_the_last_n_minutes(self):
        flat_then_rising = window([50] * 20 + [50 + 0.2 * i for i in range(1, 11)])

        self.assertAlmostEqual(utils.compute_n_min_slope(flat_then_rising, 10), 0.2)


class TestMinuteResampling(unittest.TestCase):
    def test_interpolates_linearly_at_every_full_minute(self):
        readings = [{"timestamp": T0 + pd.Timedelta(30, "s"), "value": 10.0},
                    {"timestamp": T0 + pd.Timedelta(150, "s"), "value": 14.0}]

        grid = utils.minute_resampling(readings)

        self.assertEqual([e["timestamp"] for e in grid], [T0 + pd.Timedelta(1, "min"), T0 + pd.Timedelta(2, "min")])
        self.assertEqual([round(e["value"], 6) for e in grid], [11.0, 13.0])

    def test_a_later_reading_with_the_same_timestamp_replaces_the_earlier_one(self):
        readings = [{"timestamp": T0, "value": 10.0}, {"timestamp": T0, "value": 12.0},
                    {"timestamp": T0 + pd.Timedelta(2, "min"), "value": 12.0}]

        grid = utils.minute_resampling(readings)

        self.assertEqual([e["value"] for e in grid], [12.0, 12.0, 12.0])


if __name__ == "__main__":
    unittest.main()
