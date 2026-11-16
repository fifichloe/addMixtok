"""ALIT 64+64 EMA reconstruction and image loading for a fresh ALIT scorer."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
VERSION4 = Path('/home/heyefei/lichenge/Mixture-of-Tokenizer/version4')
ALIT_ROOT = VERSION4 / 'third_party/alit'
LLAMAGEN = Path('/home/heyefei/lichenge/LlamaGen')
SEMANTIC = Path('/tmp/mixtok_light_backbones_20261111/efficientnet_b0_rwightman-7f5810bc.pth')
WEIGHTS = Path('/tmp/mixtok_alit20_20261116')


class ImagePaths(Dataset):
    def __init__(self, paths):
        sys.path.insert(0, str(LLAMAGEN))
        from dataset.augmentation import center_crop_arr
        self.paths = list(map(Path, paths))
        self.transform = transforms.Compose([
            transforms.Lambda(lambda pil: center_crop_arr(pil, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as image:
            return self.transform(image.convert('RGB'))


def load_alit(device='cuda'):
    sys.path.insert(0, str(VERSION4))
    from experiments.alit_vqgan_mix.eval_mot_fixed96 import load_eval_model
    args = argparse.Namespace(
        alit_root=str(ALIT_ROOT),
        vqgan_ckpt=str(WEIGHTS / 'official/vqgan.ckpt'),
        alit_ckpt=str(WEIGHTS / 'official/alit_vqgan_small_quantized_latent.pth'),
        ckpt=str(WEIGHTS / 'alit64_vqgan_mix_fixed64_l1_3_h200_20epoch/epoch_0020.pt'),
        router_hidden_dim=128, router_depth=3, router_detach_inputs=True,
        alit_tokens=64, refine_tokens=64, use_model_ema=True,
    )
    model, step, epoch, source = load_eval_model(args, torch.device(device))
    assert step == 66720 and int(epoch) == 20, (step, epoch)
    return model, dict(step=step, epoch=epoch, source_alit_state=source)
