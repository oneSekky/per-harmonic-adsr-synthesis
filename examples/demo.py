"""
Demo: Per-Harmonic ADSR Synthesis
==================================

This script demonstrates the core functionality of the synthesizer:
1. Load an audio sample
2. Analyze and extract harmonic parameters
3. Synthesize at a new pitch
4. Save the result

Usage:
    python examples/demo.py
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from synth import SynthEngine
from scipy.io import wavfile


def demo_pitch_shifting():
    """
    Demonstrates pitch-shifting a piano sample from C4 (262 Hz) to A5 (880 Hz)
    while preserving the original temporal envelope.
    """
    print("=" * 60)
    print("Per-Harmonic ADSR Synthesis - Demo")
    print("=" * 60)

    # Initialize the synthesis engine
    print("\n[1] Initializing synthesis engine...")
    synth = SynthEngine()

    # Path to sample (adjust if needed)
    sample_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'samples',
        'test_samples',
        'piano-middle-c.wav'
    )

    if not os.path.exists(sample_path):
        print(f"\n⚠️  Sample file not found: {sample_path}")
        print("Please ensure test samples are in samples/test_samples/")
        return

    # Load and analyze the sample
    print(f"\n[2] Loading sample: {os.path.basename(sample_path)}")
    synth.load_sample(sample_path)

    print(f"    ✓ Detected fundamental frequency: {synth.f0:.2f} Hz")
    print(f"    ✓ Number of harmonics: {synth.K}")
    print(f"    ✓ Harmonic frequencies: {[f'{h:.1f}' for h in synth.harmonics[:5]]}... Hz")

    # Synthesize at original pitch
    print(f"\n[3] Synthesizing at original pitch ({synth.f0:.2f} Hz)...")
    original_output = synth.play_note(synth.f0)

    # Synthesize at new pitch (A5 = 880 Hz, ~3.35x higher)
    target_freq = 880.0
    print(f"\n[4] Synthesizing at new pitch ({target_freq:.2f} Hz)...")
    print(f"    ✓ Pitch shift: {12 * np.log2(target_freq / synth.f0):.1f} semitones")
    shifted_output = synth.play_note(target_freq)

    # Save outputs
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'examples')
    os.makedirs(output_dir, exist_ok=True)

    original_path = os.path.join(output_dir, 'demo_original.wav')
    shifted_path = os.path.join(output_dir, 'demo_shifted.wav')

    print(f"\n[5] Saving outputs...")
    wavfile.write(original_path, 44100, original_output.astype('int16'))
    wavfile.write(shifted_path, 44100, shifted_output.astype('int16'))

    print(f"    ✓ Original saved to: {original_path}")
    print(f"    ✓ Shifted saved to: {shifted_path}")

    print("\n" + "=" * 60)
    print("✅ Demo complete!")
    print("=" * 60)
    print("\nKey observations:")
    print("  • The shifted sample maintains the original attack sharpness")
    print("  • Decay characteristics are preserved (no artificial speed-up)")
    print("  • Only the pitch changed, not the temporal envelope")
    print("\nCompare this to traditional pitch-shifting (e.g., changing")
    print("playback speed), which would artificially sharpen the attack")
    print("and accelerate the decay at higher pitches.")
    print("=" * 60)


def demo_harmonic_visualization():
    """
    Demonstrates harmonic analysis by showing individual harmonic contributions.
    """
    print("\n" + "=" * 60)
    print("Harmonic Analysis Demo")
    print("=" * 60)

    synth = SynthEngine()

    sample_path = os.path.join(
        os.path.dirname(__file__),
        '..',
        'samples',
        'test_samples',
        'bass-guitar-d.wav'
    )

    if not os.path.exists(sample_path):
        print(f"\n⚠️  Sample file not found")
        return

    print(f"\nAnalyzing: {os.path.basename(sample_path)}")
    synth.load_sample(sample_path)

    print(f"\nHarmonic breakdown:")
    print(f"{'Harmonic':<10} {'Frequency':<12} {'Amplitude':<12} {'Phase':<10}")
    print("-" * 50)

    for i in range(min(synth.K, 10)):  # Show first 10 harmonics
        harm_num = i + 1
        freq = synth.harmonics[i]
        amp = synth.amplitudes[i]
        phase = synth.phases[i]
        print(f"{harm_num:<10} {freq:<12.2f} {amp:<12.4f} {phase:<10.3f}")

    if synth.K > 10:
        print(f"... and {synth.K - 10} more harmonics")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Run the main demo
    demo_pitch_shifting()

    # Uncomment to see harmonic analysis
    # demo_harmonic_visualization()
