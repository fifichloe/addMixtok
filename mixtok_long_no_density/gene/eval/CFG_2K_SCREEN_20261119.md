# MixTok Long: 2k-image second-stage CFG screen

Current density-free Long checkpoint (SHA-256 prefix `04a2f5e061fa`), seed 4, 8+8 sampling rounds, and first-stage CFG 4.5. Each row uses the same first 2,000 generated image IDs from its completed 50k run, with two images per ImageNet class. ADM gFID was recomputed from the saved Inception pool features; checkpoint, sampling setup, feature hashes, 1D tokens, selected positions, and per-image K were checked for consistency.

| Second-stage CFG | 2k gFID ↓ | 50k gFID ↓ |
|---:|---:|---:|
| 2.50 | 18.6154 | 1.9147 |
| 2.75 | 18.6495 | 1.8814 |
| 3.00 | 18.4067 | **1.8654** |
| 3.25 | 18.4112 | 1.8794 |
| 3.50 | **18.3886** | 1.9116 |

The 2k subset averages 106.892 generated 2D tokens per image (138.892 total content tokens). CFG 3.5 is marginally best on 2k, by 0.0181 gFID over CFG 3.0. CFG 3.0 is best on the complete 50k evaluation. Use the 50k figures for reported results; the 2k values are screening diagnostics and are not directly comparable in magnitude to 50k gFID.
