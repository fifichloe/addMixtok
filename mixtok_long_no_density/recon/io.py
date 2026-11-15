"""Unchanged codec/ADM image loading functions, configured by the caller."""
import hashlib
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], crop):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.Lambda(lambda image: crop(image, 256)),
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))

def load_codec(args, device):
    source = args.reconstruction_root.resolve()
    sys.path.insert(0, str(source))
    import eval_titok_llamagen_mix_metrics_router_f2d_e2e_dynamic as ev
    from train_titok_llamagen_decoder_adapt_router_f2d_e2e_dynamic import DynamicBudgetRouter
    from train_titok_llamagen_recon import load_llamagen_vq, load_titok

    ev.add_path(str(args.llamagen_root.resolve()))
    from dataset.augmentation import center_crop_arr

    titok = load_titok(str(args.titok_root), str(args.titok_config),
                       str(args.titok_ckpt), device)
    vq = load_llamagen_vq(str(args.llamagen_root), str(args.llamagen_ckpt),
                          device, 16384, 8)
    model = ev.TiTokLlamaGenStage2(
        titok, vq, lg_latent_channels=256, head_channels=256,
        head_mode="feature", codebook_size=16384,
        codebook_temperature=1.0,
    ).to(device)
    model.router = DynamicBudgetRouter(
        latent_channels=256, hidden_dim=128, depth=3,
        target_ratio=0.375, min_ratio=0.0, max_ratio=1.0,
        detach_inputs=True,
    ).to(device)
    step, epoch, copied, _, _ = ev.load_trainable_params(
        model, str(args.checkpoint), require_latent_decoder=True,
        use_model_ema=True,
    )
    if copied != 692:
        raise RuntimeError(f"Expected 692 released EMA parameters, got {copied}")
    model.eval().requires_grad_(False)
    return model, ev, center_crop_arr, {"step": step, "epoch": epoch, "copied": copied}

def feature_tensor(extractor, image_uint8: torch.Tensor) -> np.ndarray:
    features = extractor(image_uint8)
    if isinstance(features, (list, tuple)):
        features = features[0]
    result = features.detach().float().cpu().numpy()
    if result.ndim != 2 or result.shape[1] != 2048 or not np.isfinite(result).all():
        raise ValueError(f"Unexpected Inception features: {result.shape}")
    return result
