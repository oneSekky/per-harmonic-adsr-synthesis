"""
Harmonic Synthesizer - Main Entry Point

This program implements a parametric system for musicality replication.
Given a single audio sample, it extracts pitch-independent musical characteristics
and uses them to synthesize the same sound at arbitrary pitches.

Usage:
    python main.py                              # Open GUI
    python main.py path/to/sample.wav           # Load WAV file
    python main.py path/to/sample.mp3           # Load MP3 file
"""

import sys
from ui import HarmonicSynthApp

if __name__ == "__main__":
    initial_sample = None
    if len(sys.argv) > 1:
        initial_sample = sys.argv[1]
        print(f"Loading initial sample: {initial_sample}")

    app = HarmonicSynthApp(initial_sample_path=initial_sample)
    app.run()
