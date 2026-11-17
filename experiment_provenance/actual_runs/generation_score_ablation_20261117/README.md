# Generation score and U ablations

The MaskGIT generator is frozen in all rows. Each score ablation re-trains the base-only generation router, the spatial stage, and the U stage. The U=0 row uses the main router without its U shift.

Thresholds were calibrated from the same 50,000 generated 1D token sequences to match the main run's average K. Calibration used scores only, before gFID.

| Variant | gFID | Mean generated 2D tokens | Delta gFID |
| --- | ---: | ---: | ---: |
| main | 1.8654 | 106.3839 | +0.0000 |
| U_zero | 1.9410 | 106.3839 | +0.0756 |
| without_gain | 1.9074 | 106.3839 | +0.0420 |
| without_white_all_explicit | 1.9155 | 106.3838 | +0.0501 |
| without_cam_all_explicit | 1.8855 | 106.3838 | +0.0201 |

All generated image IDs and 1D tokens match the main run exactly. The independent ADM audit allows only bounded cross-GPU floating-point drift in the base feature arrays (maximum absolute difference below 3e-4); each summary records the observed drift and an independent FP64 gFID check.

Run configs, stage reports, policy receipts, and 50k summary JSON are archived in this directory. ImageNet images, large feature dumps, and checkpoints are excluded.
