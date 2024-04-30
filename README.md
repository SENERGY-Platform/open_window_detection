# Open Window Detection
Detects whether a window is open by checking for step and short drops in the air humidity curve.

## Input 
| key                | type                                                 | description                                               | 
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     
| `Humidity`     | float | Humidity value in percentage |
| `Humidity_Time`     | string | Corresponding timestamp |



## Output 

| key                | type                                                 | description                                               | 
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     
| `[DEVICE_ID]`     | JSON string `{window_open: bool, timestamp: string}` | Whether the window is detected as open or not |


## Config options

| key                | type                                                 | description                                               | required |
|--------------------|------------------------------------------------------|-----------------------------------------------------------|----------|
|     |                                              |                      |     |
| `logger_level`     | string | `info`, `warning` (default), `error`, `critical`, `debug` | no |
| `data_path`     | string | Path to the mounted volume | no |
| `init_phase_length`     | number | Length of the initialization/training phase (default: 2) | no |
| `init_phase_level`     | string | Duration level of the initizalization phase (pandas timedelta string, e.g. `d` for days) | no |


