#!/usr/bin/env python3
"""
Cross-platform launcher for the Per-Harmonic ADSR Synthesizer

Usage:
    python run.py                    # Launch GUI
    python run.py demo              # Run demo example
    python run.py exp1              # Run experiment 1
    python run.py exp2              # Run experiment 2
    python run.py exp3              # Run experiment 3
    python run.py <sample.wav>      # Launch GUI with sample loaded
"""

import sys
import os
import subprocess


def main():
    # Determine what to run
    if len(sys.argv) == 1:
        # No arguments - launch GUI
        print("Launching Per-Harmonic ADSR Synthesizer GUI...")
        subprocess.run([sys.executable, "src/main.py"])

    elif sys.argv[1] == "demo":
        # Run demo
        print("Running demo...")
        subprocess.run([sys.executable, "examples/demo.py"])

    elif sys.argv[1] == "exp1":
        # Run experiment 1
        print("Running Experiment 1: Harmonic Count Sensitivity...")
        subprocess.run([sys.executable, "experiments/experiment_1.py"])

    elif sys.argv[1] == "exp2":
        # Run experiment 2
        print("Running Experiment 2: Basis Function Selection...")
        subprocess.run([sys.executable, "experiments/experiment_2.py"])

    elif sys.argv[1] == "exp3":
        # Run experiment 3
        print("Running Experiment 3: Optimization Comparison...")
        subprocess.run([sys.executable, "experiments/experiment_3.py"])

    elif sys.argv[1] in ["-h", "--help", "help"]:
        # Show help
        print(__doc__)

    else:
        # Assume it's a file path - launch GUI with that sample
        sample_path = sys.argv[1]
        if os.path.exists(sample_path):
            print(f"Launching GUI with sample: {sample_path}")
            subprocess.run([sys.executable, "src/main.py", sample_path])
        else:
            print(f"Error: File not found: {sample_path}")
            print("\nUsage: python run.py [demo|exp1|exp2|exp3|<sample.wav>]")
            sys.exit(1)


if __name__ == "__main__":
    main()
