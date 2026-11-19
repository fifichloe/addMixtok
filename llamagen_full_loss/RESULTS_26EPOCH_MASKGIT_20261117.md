# Pure LlamaGen 26-epoch control: measured results

The tokenizer is a full-grid LlamaGen VQ-16 model trained for 20+6 epochs with the matched MixTok reconstruction losses. It uses 256 local content tokens, no 1D branch, and no spatial routing. The reported reconstruction uses its EMA weights. A full-grid Halton MaskGIT generator was then trained from the official pretrained generator initialization for 20,000 updates on codes extracted with this tokenizer. The generator and tokenizer checkpoint SHA256 values are recorded in the JSON files in this directory.

## ImageNet 256 × 256 validation reconstruction

The 50,000-image validation evaluation gives **rFID 1.2178118362** and **PSNR 21.2104079755 dB**. It uses the official LlamaGen center crop, the ordered ImageNet validation set, and the same ADM Inception metric convention as the generation evaluation. See [reconstruction_50k.json](reconstruction_50k.json) for model identity and measurement details.

## Class-conditional generation

CFG was selected with 2,000 generated images at 16 sampling rounds and seed 4: CFG 0.5 gave gFID 19.5685, CFG 0.7 gave 18.7971, and CFG 1.0 gave 19.8633. The selected CFG **0.7** was then held fixed for all formal evaluations. Each formal result uses **50,000 generated images, seed 4**, and the same 256-token tokenizer and generator checkpoint.

| Sampling rounds | gFID (50k) |
| ---: | ---: |
| 12 | 4.1637042191 |
| 16 | **4.0060684723** |
| 24 | 4.2675420838 |
| 32 | 4.5151650571 |

The 24-round value is higher than the 16-round value under the common CFG 0.7. We report the measured values without retuning CFG separately for each round count. The independent centered-FP64 FID audit of each completed run agrees with the ADM result to within 0.000015. Per-run summaries are in `generation_{rounds}_rounds_50k.json`.

## Additional 16-round CFG comparison on 50k images

The same generator, tokenizer, 16 sampling rounds, and seed 4 were used for each CFG setting. All three results are complete 50,000-image ADM evaluations.

| CFG | gFID (50k) |
| ---: | ---: |
| 0.5 | **3.9766847640** |
| 0.7 | 4.0060684723 |
| 1.0 | 6.2466501434 |

CFG 0.5 is the lowest measured setting in this three-point 50k comparison. The round-count table above retains CFG 0.7, which was selected by the earlier 2k scan; these are separate evaluation choices. The independent FP64 audit differs from each ADM value by less than 0.000021. The new run summaries are [`generation_16_rounds_cfg0p5_50k.json`](generation_16_rounds_cfg0p5_50k.json) and [`generation_16_rounds_cfg1p0_50k.json`](generation_16_rounds_cfg1p0_50k.json).
