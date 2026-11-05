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
import tempfile
import unittest
from unittest import mock

from operator_lib.manifest import build_manifest
from operator_lib.util import OperatorConfig, create_filter_handler

import main
from main import CustomConfig, Operator
from tests._harness import MockKafkaProducer, at, feed, new_operator, no_history, transitions
from tests._room import Room

OPERATOR_CONFIG = {
    "config": {},
    "inputTopics": [
        {
            "name": "climate_service",
            "filterType": "DeviceId",
            "filterValue": "device:1",
            "mappings": [
                {"dest": "Temperature", "source": "value.temperature"},
                {"dest": "Humidity", "source": "value.humidity"},
            ],
        }
    ],
}


class TestRunPath(unittest.TestCase):
    def test_run_accepts_the_arguments_the_operator_lib_passes(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        opr_config = OperatorConfig(OPERATOR_CONFIG)
        operator = Operator()
        with mock.patch.object(main, "provide_historic_frames_local", no_history):
            operator.init(
                kafka_consumer=None,
                kafka_producer=MockKafkaProducer(),
                filter_handler=create_filter_handler(opr_config.inputTopics, "pipeline:1", Operator.selectors),
                output_topic="output",
                pipeline_id="pipeline:1",
                operator_id="operator:1",
                config=CustomConfig({"data_path": data_dir.name}),
            )
        message = {"device_id": "device:1", "service_id": "climate:service",
                   "value": {"temperature": 21.0, "humidity": 50.0}}

        _, results = operator._OperatorBase__call_run(
            message, "device:1", datetime.datetime(2026, 1, 15, 8, tzinfo=datetime.timezone.utc))

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["window_open"])


class TestInvalidReadings(unittest.TestCase):
    def setUp(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        self.operator, _ = new_operator(data_dir.name)
        self.send({"Temperature": 21.0, "Humidity": 50.0}, 0)  # past the init phase reset

    def send(self, data, minute):
        return self.operator.run(data=data, selector=None, device_id="device:1", timestamp=at(minute))

    def test_ignores_a_humidity_reading_that_is_null(self):
        self.assertIsNone(self.send({"Humidity": None}, 1))

    def test_ignores_a_humidity_reading_that_is_not_a_number(self):
        self.assertIsNone(self.send({"Humidity": float("nan")}, 1))

    def test_ignores_a_humidity_reading_outside_0_to_100_percent(self):
        self.assertIsNone(self.send({"Humidity": 150.0}, 1))

    def test_ignores_a_temperature_reading_that_is_not_a_number(self):
        self.assertIsNone(self.send({"Temperature": "warm"}, 1))

    def test_keeps_reporting_after_an_invalid_reading(self):
        self.send({"Humidity": None}, 1)

        self.assertFalse(self.send({"Humidity": 50.0}, 2)["window_open"])


class TestInitPhase(unittest.TestCase):
    def test_first_init_message_of_a_fresh_deployment_names_a_real_end_time(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)

        _, producer = new_operator(data_dir.name, init_phase_length="2", init_phase_level="d")

        self.assertNotIn("NaT", producer.messages[0]["analytics"]["initial_phase"])


class TestConfig(unittest.TestCase):
    def test_reads_a_threshold_from_the_string_the_flow_engine_passes(self):
        self.assertEqual(CustomConfig({"ah_drop_min": "0.5"}).ah_drop_min, 0.5)

    def test_keeps_the_default_of_an_empty_threshold(self):
        self.assertEqual(CustomConfig({"ah_drop_min": ""}).ah_drop_min, 0.4)


class TestNotification(unittest.TestCase):
    def test_notifies_once_per_change_of_the_window_state(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        operator, _ = new_operator(data_dir.name, notification_mode="always")

        outputs = feed(operator, Room(0, 80).readings(240, openings=[(120, 10)]))

        changes = transitions(outputs)
        self.assertTrue(changes, "the scenario must open the window at least once")
        self.assertEqual(len([o for _, o in outputs if o.get("notification")]), len(changes))


class TestManifest(unittest.TestCase):
    def test_declares_the_inputs(self):
        self.assertEqual(build_manifest(Operator)["inputs"],
                         [{"name": "Temperature", "type": "float"}, {"name": "Humidity", "type": "float"}])

    def test_declares_every_output_the_operator_returns(self):
        data_dir = tempfile.TemporaryDirectory()
        self.addCleanup(data_dir.cleanup)
        operator, _ = new_operator(data_dir.name)

        outputs = feed(operator, Room(0, 80).readings(5))

        returned = {key for _, output in outputs for key in output}
        self.assertEqual(returned, set(Operator.outputs.names()))

    def test_declares_every_config_field_and_the_logger_level(self):
        declared = [entry["name"] for entry in build_manifest(Operator)["config_values"]]

        self.assertEqual(set(declared), set(vars(CustomConfig)["_own_fields"]))
        self.assertIn("logger_level", declared)


if __name__ == "__main__":
    unittest.main()
