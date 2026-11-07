"""Full-grid, two-view code cache from the independently trained tokenizer."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

N_IMAGES = 1_281_167
N_VIEWS = 2
K = 256


class FullGridCodes(Dataset):
    def __init__(self, root: Path, expected_tokenizer_sha256: str):
        self.root = Path(root)
        meta = json.loads((self.root / 'meta.json').read_text())
        run = json.loads((self.root / 'run_config.json').read_text())
        if (meta.get('completed') is not True or meta.get('num_samples') != N_IMAGES
                or meta.get('num_aug') != N_VIEWS
                or meta.get('mot_checkpoint_sha256') != expected_tokenizer_sha256
                or run.get('format') != 'paired_full256_codes_v1'
                or run.get('mot_checkpoint_sha256') != expected_tokenizer_sha256):
            raise RuntimeError('full-grid code cache provenance differs')
        self.codes = np.load(self.root / 'llamagen_codes.npy', mmap_mode='r', allow_pickle=False)
        self.labels = np.load(self.root / 'labels.npy', mmap_mode='r', allow_pickle=False)
        self.written = np.load(self.root / 'written.npy', mmap_mode='r', allow_pickle=False)
        if (self.codes.shape != (N_IMAGES, N_VIEWS, K) or self.codes.dtype != np.uint16
                or self.labels.shape != (N_IMAGES,) or self.labels.dtype != np.uint16
                or self.written.shape != (N_IMAGES,) or self.written.dtype != np.uint8
                or not bool(np.all(self.written == 1))):
            raise RuntimeError('full-grid code cache is incomplete or malformed')
        self.k_values = np.full(N_IMAGES * N_VIEWS, K, dtype=np.uint16)
        self.indices = torch.arange(K, dtype=torch.long)

    def __len__(self):
        return N_IMAGES * N_VIEWS

    def __getitem__(self, flat):
        image, view = divmod(int(flat), N_VIEWS)
        return dict(z2d=torch.from_numpy(np.array(self.codes[image, view], dtype=np.int64)),
                    label=torch.tensor(int(self.labels[image]), dtype=torch.long),
                    route_indices=self.indices,
                    route_valid=torch.ones(K, dtype=torch.bool),
                    k=torch.tensor(K), source_index=torch.tensor(image),
                    augmentation=torch.tensor(view))


def collate_full_grid(rows):
    return {name: torch.stack([row[name] for row in rows]) for name in rows[0]}
