"""Synthetic audio for dataset tests (no real recordings needed)."""

import numpy as np
import soundfile as sf


def speech_like(seconds, sr, amp=0.3, seed=0):
    """Harmonic tone with a syllable-rate envelope: loud, voiced, non-silent."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    f0 = 180 + 20 * rng.random()
    sig = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 6))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return (amp * sig / np.max(np.abs(sig)) * env).astype(np.float32)


def write_wav(path, speech_s=3.0, sr=48000, lead=0.2, trail=0.2, amp=0.3, channels=1,
              subtype="PCM_24", seed=0, clip=False, noise=1e-4):
    rng = np.random.default_rng(seed + 1000)
    x = np.concatenate([np.zeros(int(lead * sr)), speech_like(speech_s, sr, amp, seed), np.zeros(int(trail * sr))])
    x = x + rng.normal(0, noise, len(x))  # noise floor
    if clip:
        x = np.clip(x * 5, -1, 1)
    if channels == 2:
        x = np.stack([x, x], axis=1)
    sf.write(str(path), x.astype(np.float32), sr, subtype=subtype)
