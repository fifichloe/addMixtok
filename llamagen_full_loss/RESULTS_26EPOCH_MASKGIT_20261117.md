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
| 32 | Running; result will be added after completion |

The 24-round value is higher than the 16-round value under the common CFG 0.7. We report the measured values without retuning CFG separately for each round count. The independent centered-FP64 FID audit of each completed run agrees with the ADM result to within 0.000015. Per-run summaries are in `generation_{rounds}_rounds_50k.json`.
