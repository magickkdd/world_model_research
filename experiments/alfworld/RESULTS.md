# ALFWorld Pilot Results

## Paired Comparison

Ten paired tasks completed with a 30-step per-episode limit. The baseline succeeded on 7/10 tasks, averaging 17.7 steps. Simulator-assisted succeeded on 10/10, averaging 6.9 steps. It used fewer steps on 8 tasks, tied on 2, and used more steps on none; mean savings were 10.8 steps per task.

The baseline used 177 API calls (124,040 prompt tokens; 17,699 completion tokens). Simulator-assisted used 126 calls (179,066 prompt tokens; 17,449 completion tokens).

The batch compared baseline with simulator-assisted candidate rollouts. It did not run the language-WM arm, so these results show the value of simulator-provided candidate outcomes in this setting, not language-WM prediction accuracy. The sample is ten tasks and does not establish performance on the full benchmark.

## Stop Boundary

The raw result file contains ten complete pairs plus one baseline-only record for task 2436. That extra baseline episode completed just before the runner received the stop signal; it has no simulator-assisted counterpart and is excluded from the paired metrics.

The raw JSON and trace are under `outputs/api_wm_assisted/`.
