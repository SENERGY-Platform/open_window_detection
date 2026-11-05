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
import math
import tempfile
import unittest
from unittest import mock

import pandas as pd
from operator_lib.util.config import MissingConfigValueError
from operator_lib.util.persistence import save

from algo import utils
from tests._harness import at, feed, history_frame, new_operator, transitions
from tests._room import Room

# a day and one period give the moving mean its baseline; the operator reads the init phase and an hour on top
HISTORY_MINUTES = 27 * 60
INIT_PHASE = {"init_phase_length": "26", "init_phase_level": "h"}


def temporary_directory(test):
    data_dir = tempfile.TemporaryDirectory()
    test.addCleanup(data_dir.cleanup)
    return data_dir.name


class TestWarmStart(unittest.TestCase):
    """A fresh deployment of a room with 27 h of history, aired 15 minutes after the start."""

    @classmethod
    def setUpClass(cls):
        cls.data_dir = tempfile.TemporaryDirectory()
        cls.airing = HISTORY_MINUTES + 15
        readings = Room(0, 80).readings(HISTORY_MINUTES + 120, openings=[(cls.airing, 10)])
        history, live = readings[:HISTORY_MINUTES], readings[HISTORY_MINUTES:]
        cls.reader = mock.Mock(return_value=[history_frame(history)])
        cls.operator, cls.producer = new_operator(cls.data_dir.name, history_reader=cls.reader, **INIT_PHASE)
        cls.messages_at_start = list(cls.producer.messages)
        cls.outputs = feed(cls.operator, live)

    @classmethod
    def tearDownClass(cls):
        cls.data_dir.cleanup()

    def test_reads_the_history_of_the_init_phase_and_an_hour(self):
        self.reader.assert_called_once_with(datetime.timedelta(hours=27))

    def test_the_init_phase_is_over_at_the_start(self):
        self.assertEqual(self.messages_at_start, [], "no init phase message")
        self.assertEqual(self.outputs[0][1]["initial_phase"], "")

    def test_the_baseline_is_active_at_the_start(self):
        self.assertIsNotNone(self.operator.movingMean_feature.current_mean)

    def test_detects_an_airing_15_minutes_after_the_start(self):
        # without history, the short-term detection needs 40 minutes of readings first
        minute, is_open = transitions(self.outputs)[0]

        self.assertTrue(is_open)
        self.assertLessEqual(minute, self.airing + 5)


class TestHistoryInputs(unittest.TestCase):
    def setUp(self):
        self.readings = Room(0, 80).readings(90)

    def start(self, frames):
        operator, _ = new_operator(temporary_directory(self), history_reader=mock.Mock(return_value=frames))
        return operator

    def test_combines_inputs_read_from_separate_topics(self):
        # the database reader gives UTC-naive times, timescale-wrapper UTC-aware ones
        humidity = history_frame(self.readings, inputs=("Humidity",))
        humidity["time"] = humidity["time"].dt.tz_localize("UTC")

        operator = self.start([history_frame(self.readings, inputs=("Temperature",)), humidity])

        minute, temperature, relative_humidity = self.readings[-1]
        self.assertEqual(operator.sliding_window_ah[-1]["timestamp"], at(minute))
        self.assertAlmostEqual(operator.sliding_window_ah[-1]["value"],
                               utils.absolute_humidity(temperature, relative_humidity))

    def test_skips_missing_and_invalid_values(self):
        frame = history_frame(self.readings)
        frame.loc[len(frame) - 1, "Humidity"] = math.nan
        frame.loc[len(frame) - 2, "Humidity"] = 150.0

        operator = self.start([frame])

        minute, _, humidity = self.readings[-3]
        self.assertEqual(operator.latest_humidity, {"timestamp": at(minute), "value": humidity})


class TestUpdate(unittest.TestCase):
    def test_the_first_start_of_this_version_replays_the_history(self):
        # a former version left its temperature window, which ends where the history ends, and its first data time;
        # the history of the last 30 minutes, too short for the window to drop the former entries on its own
        data_path = temporary_directory(self)
        readings = Room(0, 80).readings(90)[60:]
        save(data_path, "sliding_window_temp.pickle", [{"timestamp": at(minute), "value": 30.0} for minute in range(80, 90)])
        save(data_path, "first_data_time.pickle", pd.Timestamp(at(-7 * 24 * 60)))

        operator, producer = new_operator(data_path, history_reader=mock.Mock(return_value=[history_frame(readings)]),
                                          init_phase_length="2", init_phase_level="d")

        times = [entry["timestamp"] for entry in operator.sliding_window_temp]
        self.assertEqual(times, sorted(times))
        self.assertNotIn(30.0, [entry["value"] for entry in operator.sliding_window_temp])
        self.assertTrue(operator.sliding_window_ah)
        self.assertEqual(producer.messages, [], "the init phase ended with the former version")


class TestKafkaAfterHistory(unittest.TestCase):
    def test_ignores_messages_kafka_delivers_again_from_within_the_history(self):
        readings = Room(0, 80).readings(90)
        operator, _ = new_operator(temporary_directory(self), history_reader=mock.Mock(return_value=[history_frame(readings)]))
        window = list(operator.sliding_window_ah)

        minute, temperature, humidity = readings[-1]
        output = operator.run(data={"Temperature": temperature, "Humidity": humidity}, selector=None,
                              device_id="device:1", timestamp=at(minute))

        self.assertIsNone(output)
        self.assertEqual(operator.sliding_window_ah, window)


class TestNotificationAfterHistory(unittest.TestCase):
    def test_a_change_within_the_history_is_not_notified(self):
        # the window opened 5 minutes before the start
        readings = Room(0, 80).readings(185, openings=[(175, 30)])
        history, live = readings[:180], readings[180:]
        operator, _ = new_operator(temporary_directory(self), history_reader=mock.Mock(return_value=[history_frame(history)]),
                                   init_phase_length="2", init_phase_level="h", notification_mode="always")
        self.assertTrue(operator.window_open, "the window must be open at the start")

        outputs = feed(operator, live)

        self.assertEqual([output["notification"] for _, output in outputs], [None] * len(outputs))


class TestWithoutHistory(unittest.TestCase):
    def test_starts_without_history_when_no_timescale_is_configured(self):
        reader = mock.Mock(side_effect=MissingConfigValueError("cannot read history"))
        readings = Room(0, 80).readings(10)

        with self.assertLogs("operator", level="WARNING") as logs:
            operator, _ = new_operator(temporary_directory(self), history_reader=reader)
        outputs = feed(operator, readings)

        self.assertTrue(any("without history" in line for line in logs.output))
        without_history, _ = new_operator(temporary_directory(self))
        self.assertEqual(outputs, feed(without_history, readings))

    def test_reads_no_history_after_a_restart(self):
        data_path = temporary_directory(self)
        operator, _ = new_operator(data_path)
        feed(operator, Room(0, 80).readings(10))
        operator.stop()
        reader = mock.Mock(return_value=[])

        new_operator(data_path, history_reader=reader)

        reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
