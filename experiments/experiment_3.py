"""
Experiment 3: Most Common vs. Optimized Basis Function Comparison

Compares reconstruction quality between:
- Most common basis functions from Experiment 2: ATK0, DEC1, SUS3, REL1 for all harmonics
- Optimized basis functions: Per-harmonic selection

Processes all test samples and reports MSE improvement from optimization.

"""

import numpy as np
from scipy.io import wavfile
import os
import sys

# Add parent directory to path to import synth module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synth import SynthEngine, SAMPLE_RATE

def process_with_default_basis(audio, sr):
    """Process sample using most common basis functions from Experiment 2."""
    synth = SynthEngine()

    audio_trimmed = synth._trim_audio(audio, sr)
    f0, harmonics, amplitudes, phases = synth._analyze_harmonics(audio_trimmed, sr)

    synth.f0 = f0
    synth.harmonics = harmonics
    synth.amplitudes = amplitudes
    synth.phases = phases
    synth.K = len(harmonics)

    envelopes = synth._extract_envelopes(audio_trimmed, sr)

    for i in range(synth.K):
        if np.max(envelopes[i, :]) > 0:
            envelopes[i, :] = envelopes[i, :] / np.max(envelopes[i, :])

    synth.attack_indices = np.full(synth.K, 0, dtype=int)
    synth.decay_indices = np.full(synth.K, 1, dtype=int)
    synth.sustain_indices = np.full(synth.K, 3, dtype=int)
    synth.release_indices = np.full(synth.K, 1, dtype=int)

    ATK, DEC, SUS, REL = synth._extract_adsr_features(envelopes)

    transitions = synth._detect_transitions(envelopes, DEC)

    envelopes_recon = synth._reconstruct_envelopes(ATK, DEC, SUS, REL, transitions)

    N = envelopes.shape[1]
    errors = {
        'attack': 0.0,
        'decay': 0.0,
        'sustain': 0.0,
        'release': 0.0
    }

    for i in range(synth.K):
        ni_1, ni_2, ni_3 = transitions[i]

        if ni_1 > 0:
            errors['attack'] += np.mean((envelopes[i, :ni_1] - envelopes_recon[i, :ni_1])**2)

        if ni_2 > ni_1:
            errors['decay'] += np.mean((envelopes[i, ni_1:ni_2] - envelopes_recon[i, ni_1:ni_2])**2)

        if ni_3 > ni_2:
            errors['sustain'] += np.mean((envelopes[i, ni_2:ni_3] - envelopes_recon[i, ni_2:ni_3])**2)

        if N > ni_3:
            errors['release'] += np.mean((envelopes[i, ni_3:] - envelopes_recon[i, ni_3:])**2)

    for key in errors:
        errors[key] /= synth.K

    errors['total'] = (errors['attack'] + errors['decay'] + errors['sustain'] + errors['release']) / 4

    return f0, synth.K, errors

def process_with_optimized_basis(audio, sr):
    """Process sample using optimized per-harmonic basis functions."""
    synth = SynthEngine()
    f0, K = synth.process_sample(audio, sr)


    audio_trimmed = synth._trim_audio(audio, sr)
    envelopes = synth._extract_envelopes(audio_trimmed, sr)

    for i in range(K):
        if np.max(envelopes[i, :]) > 0:
            envelopes[i, :] = envelopes[i, :] / np.max(envelopes[i, :])

    ATK, DEC, SUS, REL = synth._extract_adsr_features(envelopes)

    transitions = synth._detect_transitions(envelopes, DEC)

    envelopes_recon = synth._reconstruct_envelopes(ATK, DEC, SUS, REL, transitions)

    N = envelopes.shape[1]
    errors = {
        'attack': 0.0,
        'decay': 0.0,
        'sustain': 0.0,
        'release': 0.0
    }

    for i in range(K):
        ni_1, ni_2, ni_3 = transitions[i]

        if ni_1 > 0:
            errors['attack'] += np.mean((envelopes[i, :ni_1] - envelopes_recon[i, :ni_1])**2)

        if ni_2 > ni_1:
            errors['decay'] += np.mean((envelopes[i, ni_1:ni_2] - envelopes_recon[i, ni_1:ni_2])**2)

        if ni_3 > ni_2:
            errors['sustain'] += np.mean((envelopes[i, ni_2:ni_3] - envelopes_recon[i, ni_2:ni_3])**2)

        if N > ni_3:
            errors['release'] += np.mean((envelopes[i, ni_3:] - envelopes_recon[i, ni_3:])**2)

    for key in errors:
        errors[key] /= K

    errors['total'] = (errors['attack'] + errors['decay'] + errors['sustain'] + errors['release']) / 4

    return f0, K, errors

def run_experiment(sample_path, sample_name):
    """Compare default vs optimized basis functions for a sample."""
    print("\n" + "="*80)
    print(f"EXPERIMENT 3: {sample_name}")
    print("="*80)

    sr, audio = wavfile.read(sample_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    audio = audio.astype(np.float32) / 32768.0

    print(f"\nSample: {sample_name}")
    print(f"Duration: {len(audio)/sr:.3f}s")

    print("\n" + "-"*80)
    print("METHOD 1: MOST COMMON BASIS (ATK0 DEC1 SUS3 REL1)")
    print("-"*80)
    f0_def, K_def, errors_def = process_with_default_basis(audio, sr)

    print(f"\nMost Common Basis Results:")
    print(f"  Fundamental: {f0_def:.1f} Hz")
    print(f"  Harmonics: K = {K_def}")
    print(f"  Average MSE per segment:")
    print(f"    Attack:  {errors_def['attack']:.6f}")
    print(f"    Decay:   {errors_def['decay']:.6f}")
    print(f"    Sustain: {errors_def['sustain']:.6f}")
    print(f"    Release: {errors_def['release']:.6f}")
    print(f"    Total:   {errors_def['total']:.6f}")

    print("\n" + "-"*80)
    print("METHOD 2: GREEDY SELECTED BASIS (Per-Harmonic Selection)")
    print("-"*80)
    f0_opt, K_opt, errors_opt = process_with_optimized_basis(audio, sr)

    print(f"\nGreedy Selected Basis Results:")
    print(f"  Fundamental: {f0_opt:.1f} Hz")
    print(f"  Harmonics: K = {K_opt}")
    print(f"  Average MSE per segment:")
    print(f"    Attack:  {errors_opt['attack']:.6f}")
    print(f"    Decay:   {errors_opt['decay']:.6f}")
    print(f"    Sustain: {errors_opt['sustain']:.6f}")
    print(f"    Release: {errors_opt['release']:.6f}")
    print(f"    Total:   {errors_opt['total']:.6f}")

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Metric':<20} {'Most Common':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-"*80)

    def calc_improvement(most_common, optimized):
        if most_common == 0:
            return 0.0
        return (1 - optimized/most_common) * 100

    print(f"{'Attack MSE':<20} {errors_def['attack']:>15.6f} {errors_opt['attack']:>15.6f} {calc_improvement(errors_def['attack'], errors_opt['attack']):>14.1f}%")
    print(f"{'Decay MSE':<20} {errors_def['decay']:>15.6f} {errors_opt['decay']:>15.6f} {calc_improvement(errors_def['decay'], errors_opt['decay']):>14.1f}%")
    print(f"{'Sustain MSE':<20} {errors_def['sustain']:>15.6f} {errors_opt['sustain']:>15.6f} {calc_improvement(errors_def['sustain'], errors_opt['sustain']):>14.1f}%")
    print(f"{'Release MSE':<20} {errors_def['release']:>15.6f} {errors_opt['release']:>15.6f} {calc_improvement(errors_def['release'], errors_opt['release']):>14.1f}%")
    print(f"{'Total MSE':<20} {errors_def['total']:>15.6f} {errors_opt['total']:>15.6f} {calc_improvement(errors_def['total'], errors_opt['total']):>14.1f}%")
    print("="*80)

    return {
        'sample': sample_name,
        'K': K_def,
        'f0': f0_def,
        'most_common': errors_def,
        'optimized': errors_opt
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
            result = run_experiment(sample_path, sample_name)
            results.append(result)
        else:
            print(f"\nWARNING: Sample not found: {sample_path}")

    with open("experiment_3_results.txt", "w") as f:
        for result in results:
            f.write(f"{result['sample']}\n")
            f.write(f"f0={result['f0']:.1f}\n")
            f.write(f"K={result['K']}\n")
            f.write(f"most_common_attack={result['most_common']['attack']:.6f}\n")
            f.write(f"most_common_decay={result['most_common']['decay']:.6f}\n")
            f.write(f"most_common_sustain={result['most_common']['sustain']:.6f}\n")
            f.write(f"most_common_release={result['most_common']['release']:.6f}\n")
            f.write(f"most_common_total={result['most_common']['total']:.6f}\n")
            f.write(f"optimized_attack={result['optimized']['attack']:.6f}\n")
            f.write(f"optimized_decay={result['optimized']['decay']:.6f}\n")
            f.write(f"optimized_sustain={result['optimized']['sustain']:.6f}\n")
            f.write(f"optimized_release={result['optimized']['release']:.6f}\n")
            f.write(f"optimized_total={result['optimized']['total']:.6f}\n")
            f.write("\n")

    print("\nResults saved to experiment_3_results.txt")
