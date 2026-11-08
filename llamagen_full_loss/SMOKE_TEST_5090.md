# RTX 5090 smoke test (2026-11-08)

Six RTX 5090 GPUs were used to test the memory selector with all reconstruction losses and GAN enabled. The selector measured 70.7% memory at per-GPU batch 8, 86.9% at batch 10, 94.1% at batch 11, and 99.1% at batch 12; batch 16 ran out of memory. It correctly chose batch 10 for the 92% limit. The gap between 86.9% and 94.1% is due to the integer batch size.

A one-epoch, 256-image smoke run finished and wrote `latest.pt`. Re-running the launcher resumed from that checkpoint at the next epoch without repeating the probe. A separate ten-epoch, 128-image smoke run reached epoch 10 and produced exactly two model checkpoints: rolling `latest.pt` and `epoch_10.pt`. The final `latest.pt` contained model, EMA, discriminator, and both optimizer states. A 128-image evaluator smoke run loaded the EMA checkpoint and completed FID/PSNR computation.

These short runs verify execution and checkpoint behavior only. Their metrics are not scientific results; the full ImageNet 20-epoch H200 experiment remains to be run.
