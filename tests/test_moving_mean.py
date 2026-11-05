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

import datetime
import random
import tempfile
import unittest

from algo.moving_mean_feature import MovingMeanFeature

T0 = datetime.datetime(2026, 1, 15, tzinfo=datetime.timezone.utc)
READINGS_PER_PERIOD = 72  # one every 5 minutes for 6 h


def new_feature(data_path, parameters=None):
    return MovingMeanFeature(data_path, "detections.pickle", "dismissed.pickle", "pending.pickle",
                             "waiting.pickle", model_parameters=parameters)


def readings(period_means, noise=0.5, seed=1):
    """(timestamp, value) for each 6 h period, then one more that closes the last period."""
    rnd = random.Random(seed)
    timestamp, result = T0, []
    for mean in period_means:
        for _ in range(READINGS_PER_PERIOD):
            result.append((timestamp, mean + rnd.gauss(0, noise)))
            timestamp += datetime.timedelta(minutes=5)
    result.append((timestamp, period_means[-1]))
    return result


def feed(feature, values):
    for timestamp, value in values:
        feature.update_stats(timestamp, value, False)


class MovingMeanTestCase(unittest.TestCase):
    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.data_path = data_dir.name


class TestBaselineSpread(MovingMeanTestCase):
    def test_includes_the_daily_cycle_between_periods(self):
        feature = new_feature(self.data_path)

        feed(feature, readings([45, 50, 55, 50]))

        # within-period noise is 0.5; the period means alone spread by about 3
        self.assertGreater(feature.current_std, 2.5)


class TestBaselinePersistence(MovingMeanTestCase):
    def test_baseline_survives_a_restart(self):
        feature = new_feature(self.data_path)
        feed(feature, readings([45, 50, 55, 50]))
        feature.stop()

        restarted = new_feature(self.data_path)

        self.assertAlmostEqual(restarted.current_mean, feature.current_mean)

    def test_continues_after_a_crash_without_counting_a_period_twice(self):
        values = readings([45, 50, 55, 50, 48, 52])
        uninterrupted_dir = tempfile.TemporaryDirectory()
        self.addCleanup(uninterrupted_dir.cleanup)
        uninterrupted = new_feature(uninterrupted_dir.name)
        feed(uninterrupted, values)
        just_after_a_period_end = 3 * READINGS_PER_PERIOD + 1

        feed(new_feature(self.data_path), values[:just_after_a_period_end])  # no stop(): the process died
        restarted = new_feature(self.data_path)
        feed(restarted, values[just_after_a_period_end:])

        self.assertAlmostEqual(restarted.current_mean, uninterrupted.current_mean)
        self.assertAlmostEqual(restarted.current_std, uninterrupted.current_std)


if __name__ == "__main__":
    unittest.main()
