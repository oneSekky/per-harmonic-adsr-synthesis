"""
Experiment 1: Harmonic Count Sensitivity Analysis

Investigates reconstruction quality as a function of the number of harmonics (K).
Tests: Piano Middle C and Violin A4 samples.

For each K from 1 to K_max:
- Reconstruct signal using only K harmonics
- Compute MSE between reconstructed and original signal

Data output to experiment_1_results.txt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import os
import sys

# Add parent directory to path to import synth module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synth import SynthEngine, SAMPLE_RATE

def compute_signal_mse(original, reconstructed):
    """MSE between original and reconstructed signals."""
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    return np.mean((original - reconstructed) ** 2)

def run_harmonic_count_experiment(sample_path, sample_name):
    """Harmonic count sensitivity analysis on a single sample."""
    print("\n" + "="*80)
    print(f"HARMONIC COUNT SENSITIVITY: {sample_name}")
    print("="*80)

    sr, audio = wavfile.read(sample_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]  # TMake mono
    audio = audio.astype(np.float32) / 32768.0

    print(f"\nSample: {sample_name}")
    print(f"Duration: {len(audio)/sr:.3f}s")

    synth = SynthEngine()
    f0, K_max = synth.process_sample(audio, sr)

    print(f"\nFundamental: {f0:.1f} Hz")
    print(f"Maximum harmonics detected: K_max = {K_max}")

    audio_trimmed = synth._trim_audio(audio, sr)

    envelopes_full = synth._extract_envelopes(audio_trimmed, sr)
    for i in range(K_max):
        if np.max(envelopes_full[i, :]) > 0:
            envelopes_full[i, :] = envelopes_full[i, :] / np.max(envelopes_full[i, :])

    print(f"\nOptimizing basis functions for {K_max} harmonics...")
    synth.optimize_basis_functions(envelopes_full)

    ATK_full, DEC_full, SUS_full, REL_full = synth._extract_adsr_features(envelopes_full)
    transitions_full = synth._detect_transitions(envelopes_full, DEC_full)
    envelopes_recon_full = synth._reconstruct_envelopes(ATK_full, DEC_full, SUS_full, REL_full, transitions_full)

    results = []

    print(f"\nTesting K from 1 to {K_max}...")
    for K in range(1, K_max + 1):
        synth_k = SynthEngine()

        synth_k.f0 = synth.f0
        synth_k.K = K
        synth_k.harmonics = synth.harmonics[:K].copy()
        synth_k.amplitudes = synth.amplitudes[:K].copy()
        synth_k.phases = synth.phases[:K].copy()

        synth_k.envelopes_reconstructed = envelopes_recon_full[:K, :]

        duration = len(audio_trimmed) / sr
        audio_recon = synth_k.synthesize_frequency(synth.f0, duration=duration)

        mse = compute_signal_mse(audio_trimmed, audio_recon)

        results.append({
            'K': K,
            'MSE': mse
        })

        print(f"  K={K:2d}: MSE = {mse:.6f}")

    return {
        'sample': sample_name,
        'f0': f0,
        'K_max': K_max,
        'results': results
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

    all_results = []

    for sample_path, sample_name in samples:
        if os.path.exists(sample_path):
            result = run_harmonic_count_experiment(sample_path, sample_name)
            all_results.append(result)
        else:
            print(f"\nWARNING: Sample not found: {sample_path}")

    
    with open("experiment_1_results.txt", "w") as f:
        for result in all_results:
            f.write(f"{result['sample']}\n")
            f.write(f"f0={result['f0']:.1f}\n")
            f.write(f"K_max={result['K_max']}\n")
            for r in result['results']:
                f.write(f"{r['K']},{r['MSE']:.6f}\n")
            f.write("\n")

    print("Results saved: experiment_1_results.txt")
