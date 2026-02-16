"""
Experiment 2: Basis Function Selection Analysis

Analyzes which basis functions are selected for different instrument types and harmonic indices to identify patterns and validate the need for per-harmonic selection.

Data output to experiment_2_results.txt
"""

import numpy as np
from scipy.io import wavfile
import os
import sys
from collections import Counter

# Add parent directory to path to import synth module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synth import SynthEngine, SAMPLE_RATE

def analyze_sample(sample_path, sample_name):
    """Cllect basis function selections for a single sample"""
    print("\n" + "="*80)
    print(f"Analyzing: {sample_name}")
    print("="*80)

    synth = SynthEngine()
    sr, audio = wavfile.read(sample_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32) / 32768.0

    f0, K = synth.process_sample(audio, sr)

    print(f"\nProcessed: {sample_name}")
    print(f"  Fundamental: {f0:.1f} Hz")
    print(f"  Harmonics: K = {K}")

    attack_indices = synth.attack_indices
    decay_indices = synth.decay_indices
    sustain_indices = synth.sustain_indices
    release_indices = synth.release_indices

    print(f"\nPer-harmoninc Basis Function Selections:")
    print(f"{'H':>3} {'Freq':>8} {'ATK':>5} {'DEC':>5} {'SUS':>5} {'REL':>5}")
    print("-"*35)

    for i in range(K):
        freq = synth.harmonics[i]
        atk = attack_indices[i]
        dec = decay_indices[i]
        sus = sustain_indices[i]
        rel = release_indices[i]
        print(f"{i+1:>3} {freq:>8.1f} ATK{atk:>1} DEC{dec:>1} SUS{sus:>1} REL{rel:>1}")

    return {
        'name': sample_name,
        'f0': f0,
        'K': K,
        'harmonics': synth.harmonics.copy(),
        'attack': attack_indices.copy(),
        'decay': decay_indices.copy(),
        'sustain': sustain_indices.copy(),
        'release': release_indices.copy()
    }


if __name__ == "__main__":
    samples = [
        ("../samples/test_samples/bass-guitar-d.wav", "Bass Guitar D2"),
        ("../samples/test_samples/bass-guitar-g.wav", "Bass Guitar G2"),
        ("../samples/test_samples/bass-guitar-high-b.wav", "Bass Guitar B3"),
        ("../samples/test_samples/bass-guitar-low-b.wav", "Bass Guitar B1"),
        ("../samples/test_samples/bass-guitar-short-e.wav", "Bass Guitar E2"),
        ("../samples/test_samples/bass-guitar-very-high-d.wav", "Bass Guitar D4"),
        ("../samples/test_samples/clap-1.wav", "Clap"),
        ("../samples/test_samples/piano-3.wav", "Piano 3"),
        ("../samples/test_samples/piano-4.wav", "Piano 4"),
        ("../samples/test_samples/piano-high-g.wav", "Piano High G"),
        ("../samples/test_samples/piano-middle-c.wav", "Piano Middle C"),
        ("../samples/test_samples/violin-a-bowed.wav", "Violin A4"),
        ("../samples/test_samples/voice-1.wav", "Voice 1"),
        ("../samples/test_samples/voice-2.wav", "Voice 2")
    ]

    results = []

    for sample_path, sample_name in samples:
        if os.path.exists(sample_path):
            result = analyze_sample(sample_path, sample_name)
            results.append(result)
        else:
            print(f"\nWARNING: Sample not found: {sample_path}")

    with open("experiment_2_results.txt", "w") as f:
        for result in results:
            f.write(f"{result['name']}\n")
            f.write(f"f0={result['f0']:.1f}\n")
            f.write(f"K={result['K']}\n")
            for i in range(result['K']):
                f.write(f"{i+1},{result['harmonics'][i]:.1f},{result['attack'][i]},{result['decay'][i]},{result['sustain'][i]},{result['release'][i]}\n")
            f.write("\n")

    print("\n\nResults saved to experiment_2_results.txt")
