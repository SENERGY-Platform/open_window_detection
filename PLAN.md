# Plan: calibrating the detection instead of hand-setting it

The detection is three parallel paths — `drops_humid`, `drops_temp`, `moving_mean` — ORed into one decision, with
sixteen hand-set thresholds whose defaults the README itself traces to a room model rather than to field data. The
failure mode is structural: because the paths are ORed, their false alarms add up while their combined recall barely
improves, and sixteen thresholds are sixteen things that would have to be re-set per room.

The plan replaces the hand-set numbers with numbers learned from each room's own history, without labels, and then
narrows three paths to two with distinct jobs.

## Priorities this plan is written against

Set by the developer, 2026-11-06:

- A real opening must be detected quickly, above all in winter. This is the case to optimise.
- A tilted window may be detected only after a long time.
- It stays unsupervised. There is no window contact to train on, now or later.

## Step 1 — scale the drop by the room's own movement (done, tests green)

`algo/utils.py`, `algo/drops_and_risings_feature.py`, `tests/test_utils.py`, `tests/test_drops.py`.

`level_change_scale` estimates how far the room's level moves by itself **over the same span the drop is measured
across**: the readings go onto a one-minute grid with each reading held until the next one, and the scale is the
median change in magnitude between two grid points that far apart. `is_falling_unusually` requires the drop to exceed
`ah_drop_std_factor` times that scale.

What this fixed: `ah_drop_std_factor` was inert. The original code measured the spread around the trend of the
*minute-resampled* window, which is a straight line between two readings of an on-change sensor — a straight line has
no spread, so the estimate collapsed onto its floor and `ah_drop_min` was the only criterion left.

Two decisions inside it, both load-bearing:

- **Both sides of the comparison span the same time**, so nothing converts between spans. This replaced a first
  attempt that normalised each change between two readings by the square root of its own gap; real data refuted that
  (below), and no exponent fixes it, so the exponent is gone rather than re-fitted.
- **The level is held, not interpolated.** A reading holds until the next one, which is what an on-change sensor
  means. Interpolation turns a quiet half hour into a line of small changes and a jump into a gentle slope.
  `ah_drop_min` = 0.4 stays as an absolute floor; whether it can go is for step 2 to answer.

### What the real data said, 2026-11-16

Nine days of raw rows from two rooms, read through the Operator Lib reader (Aqara-01, 1312 rows; *Arbeitszimmer
Dani*, Bosch BTH-RM230Z, 8490 rows). The first attempt — change divided by the square root of its own gap — was
measured against both and does not hold:

| | exponent of scale ~ gap**x | readings in the 0–1 min bin |
|---|---|---|
| Aqara-01 | 0.23 | 729 of 1312 (median gap 0.000 min) |
| Bosch Dani | −0.02 | 5754 of 8490 (median gap 0.002 min) |

A random walk would give 0.50. Two causes, both real and both against the assumption:

- zigbee2mqtt republishes the whole state on every attribute change, so one event arrives as several rows within the
  same second. Two thirds of all rows on the Bosch. Between two of them absolute humidity does not move, and dividing
  that non-change by the square root of a non-gap dragged the median onto the floor — the feature was inert on
  exactly these devices.
- For an on-change sensor a long gap means the room **held still**, not that it had longer to move. On the Bosch the
  10–20 and 20–60 minute bins are exactly zero. Gap length is anti-correlated with movement, which is the opposite of
  what a random walk assumes.

Both go away on a held grid: rows of one message collapse into one level, and a quiet gap contributes the zero change
it is. Measured as a regression test in `tests/test_utils.py`.

**Known limit, and it is the reason `ah_scale` is in *Open decisions*.** The scale is now well defined however
sparsely a sensor reports — a held grid needs readings, not many readings — but the operator holds one hour, which is
about two spans of half an hour. Measured on a drifting series, ten hours of readings put the growth between two
spans anywhere between 1.6 and 3.0 where it should be 2.0; two days settle it within a tenth. A scale worth trusting
comes from weeks, and the operator does not have weeks.

## Step 2 — learn the calibration from history, unsupervised (written, run over real data outstanding)

`tools/calibrate.py`, `tests/test_calibrate.py`. A development tool, not in the image — the Dockerfile copies only
`main.py` and `algo/`.

The idea that makes this work without labels: an airing is unmistakable **in hindsight** — a deep fall of absolute
humidity that holds, with the temperature falling too, and both recovering afterwards. A deployed operator cannot
look forward; a calibrator can. Found that way, airings are pseudo-labels good enough to set a threshold against,
at high precision and mediocre recall. Mediocre recall is fine: the list has to be clean, not complete.

Reports: the room's own movement over the spans a drop is measured across (report 1, the scale itself, over the whole
history and over one hour at a time for comparison), the same between two messages per gap length (report 1b, the
diagnostic that refuted the square root and the place a new sensor would show the same), the airings, and a sweep over
`ah_drop_std_factor` giving per value the share of airings caught, the latency, and the false alarms per day outside
them. The sweep runs the operator's own code.

What it cannot tell you, and the README says so too: the pseudo-labels inherit their own blind spots — an airing in
mild, humid weather leaves no humidity signature for them to find either — so the result is calibrated on the winter
case, which is the stated priority but still a restriction. Tilted windows are barely in there at all.

**Outstanding.** The run over the nine days of real history has not been repeated since step 1 changed; the platform
token was rejected when it was tried. The previous run, on the refuted version, gave the Bosch room five airing
candidates (really about three, two pairs overlap), depth 1.6–2.7 g/m³ against 3.8–4.2 in the room model, and an
upper bound of 1.67 false alarms a day at the shipped defaults. Those numbers describe the old measurement and are
kept only as the order of magnitude to expect.

## Step 3 — hysteresis, and two channels instead of three paths

Not started. In this order, because the sweep of step 2 is what says whether the thresholds are near right.

1. **Hysteresis and a minimum open duration.** Opening needs 0.4 g/m³, closing already counts at a 0.3 g/m³ recovery,
   with no minimum duration — a fluctuation below the detection threshold ends the state immediately. Closing should
   require a *fraction of the fall that opened it*, relative to the observed event rather than absolute, plus a
   minimum open time. Two knobs fewer, and the flutter goes away structurally rather than by tuning.
2. **Two channels with distinct jobs.** A fast one on the gradient, which may be aggressive because its job is
   detecting real openings within minutes; a slow one on the distance to the baseline, which may be conservative
   because its job is tilted windows and long airings, where latency does not matter. Today's three ORed paths make
   both impossible at once. The latency to beat: the previous calibrator run put the median at 5–11 minutes, and the
   40-minute minimum span in `compute_front_end_measures` plus the 5-minute end window is a floor under it that no
   parameter removes.
3. **A separate output for the slow channel.** "Detected 3 minutes ago, uncertain" and "has been open for two hours,
   certain" are different statements. Putting both in `window_open` throws that away downstream.
4. **`feature` into the output.** It is already written into `detections` but not into `OpenWindowOutputs`, so no
   error rate can be attributed to a mechanism. Three lines, and a precondition for judging anything after step 2.

Also in this step, smaller and independent:

- `is_falling_extreme_last2min` compares the last two raw readings with no time check at all, despite its name. With
  on-change reporting the gap varies, so "0.9 g/m³ between two readings" is sometimes a collapse and sometimes half
  an hour of normal decline — and with republished rows the two readings are often the same message twice.
- `check_for_fast_risings` is called with `self.temperature_before_opening or temperature` (`main.py:307`). Where no
  reference temperature was ever set — the `drops_temp` and `moving_mean` paths — the fallback is the current, cold
  temperature, which is exactly the false alarm the check's own docstring says it prevents. Skipping the check
  without a reference is the cleaner fix.
- `_humidity_rebound_detected` is not persisted, unlike the baseline state, so a restart within the 30 minutes
  reports again.
- `held_reading` uses `abs(...)`, so a future-dated reading counts as fresh. During the replay, where Timescale and
  Kafka times meet, that is not theoretical.

## Step 4 — common mode across the rooms

Not started, and an architecture decision rather than a parameter change.

An open window lowers absolute humidity in **one** room. Weather, heating, cooking and showering move several at
once. With nine climate sensors in one flat, the median of the other rooms is an unsupervised estimate of that common
mode, and subtracting it replaces a good part of the missing outdoor measurement. The data is already there.

It breaks the operator model: one operator per room does not see the others. The clean shape is an upstream operator
that produces a common-mode topic from all rooms, which the room operators read as a third input. That is worth
doing, but not before steps 1 to 3 have been measured.

## Open decisions for the developer

- **`ah_scale` as a config field.** The calibrator hands the room's scale in, measured over weeks, instead of the
  operator estimating it from an hour that is two spans long. Three lines in `CustomConfig`, but it widens the config
  surface and the manifest, so it is a decision rather than a detail. It is also the more honest shape — a learned
  number that is visible and versioned in the deployment, rather than one that silently changes on restart. Step 1's
  own measurement stays as the fallback for a room nobody has calibrated.
- **The history read at startup is tied to the init phase**, currently just over two days. That is far too little for
  a scale estimate, which needs weeks. Either decouple the two, or pull the calibration out of the operator
  altogether and pass it in — which is the same decision as above.
- **The config comments in `main.py` still describe the old meaning** of `ah_drop_min` and `ah_drop_std_factor`
  (lines 83–84). Left alone because the write tool replaces whole files and 465 lines is a poor trade for two
  comments; it belongs in the same change as `ah_scale`.
- **Outdoor temperature as a third input** is probably the single largest lever, and it is not in any step above. In
  mild, humid weather the humidity contrast disappears and only the least robust path is left.

## Measuring any of this

There is one window contact on the platform with usable history, *Kontakt Fenster Arbeitszimmer*, and its labels are
thinner than they look: a 6.07-day gap leaves 2.9 usable days containing about three openings. `value.contact` is
1 = closed, 0 = open (measured, not assumed).

Three events are not a recall figure. What those 2.9 days *do* support, and support well, is a false-alarm rate: 99 %
of the time the window is shut, so every detection outside the three openings counts. Which of the two work rooms
the contact belongs to is still unresolved, and with three events a wrong room makes the result worthless rather than
merely noisy.

Label-free measures are better value here and run over weeks instead of days: detections per day and room, the
distribution of open durations, time-of-day distribution, and self-consistency — a real airing is followed by a
measurable recovery, and a detection without that aftermath is suspect.
