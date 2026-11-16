# Generation comparison candidates

These 200 class-conditional samples use the accepted density-free Long MixTok generator, the learned dynamic router, 8+8 sampling rounds, and CFG 4.5/3.0. Each comparison shows the 1D base, the selected 2D locations on the base, and the final generation. White cells at 55% opacity with thin dark outlines mark selected locations. The underlying base and final panels are unchanged.

The source pool contains one seed-4 sample for each of 1000 ImageNet classes. Its eight-image batches follow the full-50k evaluation RNG protocol; the first batch was verified bitwise against its saved images, tokens, and mask. The 200 candidates were selected for large RGB base-to-final change within four K ranges, with quotas 77/4/4/115. This is for browsing visible refinements and is not a random or representative quality sample. A large change does not by itself establish better visual quality.

Browse the 200 full-resolution triptychs in `comparisons/` or ten 20-image sheets in `contact_sheets/`. Original 256x256 `base/` and `final/` PNGs, actual 16x16 `masks/`, `samples.csv`, and `manifest.json` support figure selection and auditing. K counts only the local 2D tokens. These images are qualitative examples, not additional gFID evaluations.
