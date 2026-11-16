# Reconstruction comparison candidates

These 200 ImageNet validation examples show ground truth, 1D base, selected 2D locations, and the MixTok reconstruction. They are deliberately selected for a visible 2D refinement effect, so they are qualitative examples rather than a representative random sample. The exact validation index, class, token count, per-image metrics, and PNG hash are in `manifest.json`.

The outputs use the density-free Long reconstruction scorer and the released MixTok reconstruction checkpoint identified by SHA256 in `manifest.json`. Panels include center crops of the ImageNet validation images; no model weights or standalone dataset files are included.

Browse the 200 candidates in ten contact sheets:

| | | | | |
|---|---|---|---|---|
| [1–20](contact_sheets/contact_01.jpg) | [21–40](contact_sheets/contact_02.jpg) | [41–60](contact_sheets/contact_03.jpg) | [61–80](contact_sheets/contact_04.jpg) | [81–100](contact_sheets/contact_05.jpg) |
| [101–120](contact_sheets/contact_06.jpg) | [121–140](contact_sheets/contact_07.jpg) | [141–160](contact_sheets/contact_08.jpg) | [161–180](contact_sheets/contact_09.jpg) | [181–200](contact_sheets/contact_10.jpg) |

Each numbered PNG is the full-resolution four-panel comparison. The ordering prioritizes visible measured improvement and limits each ImageNet class to two examples. Figure 5 in the paper compares *generated* samples and needs a separate selection pool.
