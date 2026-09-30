"""Latent dataset, batching and LR schedule for FreyaTTS fine-tuning.

Adapted from FreyaTTS training/pretrain.py (Apache-2.0,
https://github.com/freyavoiceai/FreyaTTS, commit 146d36c). Copied instead of
imported because that module pulls in `accelerate` and `yaml` at import time,
which a single-GPU fine-tune does not need. Behaviour is kept identical;
the only addition is that items keep their record id.
"""

import glob
import math
import os

import torch
from torch.utils.data import Dataset

FILL_ID = 0
UNK_ID = 1


def lr_at(step, warmup, total, base):
    """Linear warmup then cosine decay to 5% of the peak."""
    if step < warmup:
        return base * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.05 * base + 0.95 * base * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


class LatentDataset(Dataset):
    """Map-style dataset over precomputed latent shards, held in RAM as fp16."""

    def __init__(self, data_dir, char_to_id, max_frames):
        self.items = []
        self.ids = []
        for path in sorted(glob.glob(os.path.join(data_dir, "*.pt"))):
            for entry in torch.load(path, weights_only=False):
                latent = entry["latent"]
                n_frames = latent.shape[0]
                if n_frames < 8 or n_frames > max_frames:
                    continue
                text = str(entry.get("text", ""))[:300]
                ids = [char_to_id.get(ch, UNK_ID) for ch in text]
                if len(ids) < 1 or len(ids) > 250:
                    continue
                self.items.append((latent.half(), torch.tensor(ids, dtype=torch.long)))
                self.ids.append(entry.get("id"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(batch):
    """Pad latents and text ids to the batch maximum, with boolean masks."""
    max_t = max(latent.shape[0] for latent, _ in batch)
    max_l = max(ids.shape[0] for _, ids in batch)
    batch_size = len(batch)
    feat = batch[0][0].shape[1]

    latents = torch.zeros(batch_size, max_t, feat)
    text = torch.full((batch_size, max_l), FILL_ID, dtype=torch.long)
    frame_mask = torch.zeros(batch_size, max_t, dtype=torch.bool)
    char_mask = torch.zeros(batch_size, max_l, dtype=torch.bool)
    log_frames = torch.zeros(batch_size)

    for i, (latent, ids) in enumerate(batch):
        t = latent.shape[0]
        n = ids.shape[0]
        latents[i, :t] = latent.float()
        frame_mask[i, :t] = True
        text[i, :n] = ids
        char_mask[i, :n] = True
        log_frames[i] = math.log(t)

    return dict(lat=latents, text=text, fmask=frame_mask, cmask=char_mask, logT=log_frames)
