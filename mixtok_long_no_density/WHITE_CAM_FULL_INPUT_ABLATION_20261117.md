# White and CAM full-input reconstruction ablations (50k images)

These controls use the accepted density-free Long reconstruction codec and trained heads. The scorer weights are frozen. Every arm has mean local budget K=96. For dynamic budgets, a scalar threshold is calibrated from scores only to match mean K=96; the fixed control uses K=96 for every image. No FID result is used in calibration.

| Budget | Scorer | rFID ↓ | PSNR ↑ | ΔrFID from full |
|---|---|---:|---:|---:|
| Dynamic | Full | 1.000763 | 19.4012 | — |
| Dynamic | Remove explicit White everywhere | 1.003671 | 19.3404 | +0.002908 |
| Dynamic | Remove explicit CAM everywhere | 1.140766 | 19.6149 | +0.140003 |
| Fixed K=96 | Full | 1.035990 | 19.3266 | — |
| Fixed K=96 | Remove explicit White everywhere | 1.044037 | 19.2691 | +0.008047 |
| Fixed K=96 | Remove explicit CAM everywhere | 1.202601 | 19.5625 | +0.166611 |

“Remove White” deletes the direct 0.5× whitened cosine term and neutralizes the whitened cosine input to the gain head using its training mean. “Remove CAM” deletes the direct 0.75× CAM term and neutralizes the raw and standardized CAM channels in both the gain head and the image-utility regressor. Generic cosine and other semantic features remain in the model. This is an inference-time component ablation; the heads were not retrained.

The earlier `without_white` and `without_cam` rows removed only the direct score term. Their dynamic rFID values were 1.002403 and 1.131262; their fixed rFID values were 1.042281 and 1.200785. Those rows are valid **direct-term** controls but should not be described as removal of all explicit component inputs.

The new full-policy masks were replayed independently, and the full arm exactly reproduces the accepted mask for every validation image. All six rFID values were independently recomputed with centered FP64 moments and a separate matrix square-root method; the maximum absolute discrepancy is below 2e-12. The CAM removal worsens rFID while improving PSNR, consistent with different metric preferences rather than a computation error.
