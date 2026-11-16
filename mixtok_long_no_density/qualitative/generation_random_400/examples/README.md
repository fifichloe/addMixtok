# Random generation examples (400 images)

These 400 class-conditional samples use the accepted density-free Long MixTok generator, the learned dynamic router, 8+8 sampling rounds, and CFG 4.5/3.0. Each comparison shows the 1D base, the selected 2D locations on the base, and the final generation. White cells at 55% opacity with thin dark outlines mark selected locations. The underlying base and final panels are unchanged.

A fixed random draw chose 400 distinct classes uniformly from ImageNet IDs 0..999 before sampling. One image was generated for each class; all 400 are exported in draw order, without filtering by image quality, K, scores or base-to-final difference. The seed and exact class draw are recorded in `manifest.json` and `class_ids.npy`. These examples are separate from the 50,000-image gFID evaluation.

Browse the 400 full-resolution triptychs in `comparisons/` or twenty 20-image sheets in `contact_sheets/`. Original 256x256 `base/` and `final/` PNGs, actual 16x16 `masks/`, `samples.csv`, and `manifest.json` support figure selection and auditing. K counts only the local 2D tokens. These images are qualitative examples, not additional gFID evaluations.
