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

import math
import random


def saturation_vapour_pressure(temperature):
    """hPa over water, Magnus formula with the WMO coefficients."""
    return 6.112 * math.exp(17.62 * temperature / (243.12 + temperature))


def absolute_humidity(temperature, relative_humidity):
    """g/m³"""
    return 216.7 * (relative_humidity / 100 * saturation_vapour_pressure(temperature)) / (273.15 + temperature)


def relative_humidity(temperature, absolute_humidity):
    """%"""
    return 100 * absolute_humidity * (273.15 + temperature) / 216.7 / saturation_vapour_pressure(temperature)


class Room:
    """A heated room with first-order air dynamics, for scenario tests only.

    While a window is open, air temperature and absolute humidity approach the outdoor values, humidity much
    faster than temperature because walls and furniture buffer heat far more than moisture. After closing,
    temperature recovers towards the setpoint and humidity towards its baseline. A moisture source (a shower,
    cooking) adds humidity that the room's ventilation removes again. Readings carry sensor noise and are
    rounded to 0.1 like the sensors' output.
    """

    def __init__(self, outdoor_temperature, outdoor_humidity, temperature=21.0, humidity=50.0,
                 tau_humidity_open=5, tau_temperature_open=60, tau_temperature_heating=25,
                 tau_humidity_recovery=90, tau_moisture_removal=20, seed=1):
        self.outdoor_temperature = outdoor_temperature
        self.outdoor_ah = absolute_humidity(outdoor_temperature, outdoor_humidity)
        self.setpoint = temperature
        self.baseline_ah = absolute_humidity(temperature, humidity)
        self.tau_humidity_open = tau_humidity_open
        self.tau_temperature_open = tau_temperature_open
        self.tau_temperature_heating = tau_temperature_heating
        self.tau_humidity_recovery = tau_humidity_recovery
        self.tau_moisture_removal = tau_moisture_removal
        self.random = random.Random(seed)

    def readings(self, minutes, openings=(), moisture_sources=()):
        """(minute, temperature, humidity) once a minute. openings are (start, duration) in minutes,
        moisture_sources (start, duration, g/m³ added per minute)."""
        temperature, airing_offset, moisture = self.setpoint, 0.0, 0.0
        result = []
        for minute in range(minutes):
            is_open = any(start <= minute < start + duration for start, duration in openings)
            if is_open:
                temperature += (self.outdoor_temperature - temperature) * _step(self.tau_temperature_open)
                airing_offset += (self.outdoor_ah - self.baseline_ah - airing_offset) * _step(self.tau_humidity_open)
            else:
                temperature += (self.setpoint - temperature) * _step(self.tau_temperature_heating)
                airing_offset -= airing_offset * _step(self.tau_humidity_recovery)
            moisture += sum(rate for start, duration, rate in moisture_sources if start <= minute < start + duration)
            moisture -= moisture * _step(self.tau_moisture_removal)
            ah = self.baseline_ah + airing_offset + moisture
            result.append((
                minute,
                round(temperature + self.random.gauss(0, 0.03), 1),
                round(min(relative_humidity(temperature, ah), 100.0) + self.random.gauss(0, 0.2), 1),
            ))
        return result


def _step(tau):
    return 1 - math.exp(-1 / tau)
