"""One-time download of pinned model files into models/.

This is the only step that touches the network. Inference runs fully offline
afterwards. Every LFS file is checked against its expected SHA-256.

Usage:
    python src/download_models.py
"""

import hashlib
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

import config


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ok = True
    for mf in config.MODEL_FILES:
        mf.local_dir.mkdir(parents=True, exist_ok=True)
        print(f"-> {mf.repo_id}@{mf.revision[:10]} / {mf.filename}")
        path = Path(hf_hub_download(
            repo_id=mf.repo_id,
            filename=mf.filename,
            revision=mf.revision,
            local_dir=mf.local_dir,
        ))
        if mf.sha256:
            actual = sha256_of(path)
            if actual != mf.sha256:
                print(f"   SHA-256 MISMATCH: expected {mf.sha256}, got {actual}")
                ok = False
                continue
            print("   SHA-256 OK")
        print(f"   saved: {path.relative_to(config.PROJECT_ROOT)} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
