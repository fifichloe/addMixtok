# Generation score and U ablations — interim result

Same frozen generator, 50,000 images, seed 4, 8+8 rounds, stage-2 CFG 3.0, matched mean 2D-token count.

| Variant | gFID | Mean generated 2D tokens |
| --- | ---: | ---: |
| Main | 1.8654 | 106.38386 |
| U=0 | 1.9410 | 106.38386 |

Gain, explicit White, and explicit CAM ablations are still running. The original bitwise ADM base-feature comparison failed on minor cross-GPU floating-point differences; the independent audit checked exact IDs/1D tokens and route replay, and bounded the maximum ADM feature difference to 3e-4. See the summary JSON and aggregation script for details.
