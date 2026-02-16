# Per-Harmonic ADSR Synthesis: Pitch-Independent Timbre Reproduction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Ever notice how pitch-shifted audio sounds unnatural?** Traditional pitch-shifting destroys the temporal envelope—when you pitch up, the attack becomes unnaturally sharp; pitch down, and everything sounds sluggish. This synthesizer fixes that by decoupling pitch from timbre.

A parametric audio synthesizer that enables **pitch transposition without temporal distortion** through per-harmonic ADSR envelope decomposition and automated optimization.

## 🎯 Key Innovation

Traditional pitch-shifting methods couple playback speed with pitch (Nyquist's constraint). This project breaks that coupling through:

1. **Per-Harmonic Decomposition**: Separates audio into independent harmonic components using FIR filter banks
2. **Automated Envelope Extraction**: Each harmonic gets its own ADSR (Attack, Decay, Sustain, Release) envelope
3. **Greedy Optimization**: Automatically selects from 26 basis function configurations per harmonic
4. **Pitch-Independent Synthesis**: Reconstructs at any frequency while preserving original temporal characteristics

**Result**: Synthesize a piano sample at C4, then play it back at A5 with perfect temporal fidelity—the attack remains crisp, the decay natural, just as the original instrument would sound.

## 🔬 Technical Highlights

- **FIR Filter Bank** with Kaiser windowing (60 dB stopband attenuation) for harmonic isolation
- **Adaptive harmonic count** based on perceptual significance (up to 30 harmonics)
- **Per-harmonic optimization**: Each of K harmonics independently selects optimal basis functions
- **Phase-coherent reconstruction** for accurate waveform synthesis across arbitrary pitches

### Architecture Overview

```
Input Signal
    ↓
[FFT Analysis] → Fundamental frequency detection (f₀)
    ↓
[Filter Bank] → Harmonic isolation (K bandpass FIR filters)
    ↓
[Hilbert Transform] → Envelope extraction for each harmonic
    ↓
[ADSR Detection] → Transition point identification
    ↓
[Greedy Optimization] → Select best basis functions (26 configs/harmonic)
    ↓
[Parametric Model] → Store: {f₀, K, amplitudes, phases, envelopes}
    ↓
[Synthesis] → Reconstruct at arbitrary target frequency
```

## 📊 Experimental Results

From rigorous validation across 14 samples (bass guitar, piano, violin, percussion, voice):

### Experiment 1: Harmonic Count Sensitivity
- **2nd harmonic provides largest benefit**: ~17% MSE improvement
- **Diminishing returns**: Beyond K ≈ 0.5×K_max, minimal improvement
- **Clap sample** (K_max=30): Monotonic improvement with harmonic count
- **Violin A4** (K_max=16): Plateau at K≥6

### Experiment 2: Basis Function Selection
- **100% unanimous selection** of ATK0 and DEC1 across all 102 harmonics
- **Different harmonics require different basis functions** (validates per-harmonic approach)
- **SUS3** (50%) and **REL1** (82.4%) most common for sustain/release stages

### Experiment 3: Optimization Performance
- **Piano samples: 10-12% improvement** with per-harmonic optimization
- **Validates core hypothesis**: When harmonics evolve independently, per-harmonic modeling wins
- **Identifies greedy algorithm limitation**: Bass/violin samples show degradation due to transition point mismatch (algorithm weakness, not model weakness)

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/per-harmonic-adsr-synthesis.git
cd per-harmonic-adsr-synthesis

# Install dependencies
pip install -r requirements.txt

# Optional: For MP3 support
pip install pydub
# Also install ffmpeg: https://ffmpeg.org/download.html
```

### Basic Usage

#### Interactive GUI
```bash
python src/main.py
```

The GUI provides:
- **Virtual piano keyboard** (playable via mouse or keyboard)
- **Sample browser** with test samples (bass, piano, violin, percussion, voice)
- **Real-time synthesis** at any pitch
- **Frequency input box** for precise pitch control
- **Sample recording** (1s wait + 5s recording window)

#### Keyboard Mapping
```
Black keys: W E   T Y U   O P   ]  \
White keys: A S D F G H J K L ; ' Enter

Notable mappings:
  A = C4 (middle C, 261.63 Hz)
  H = A4 (440 Hz reference)
  K = C5 (523.25 Hz)
```

#### Programmatic Usage
```python
from src.synth import SynthEngine

# Initialize synthesizer
synth = SynthEngine()

# Load and analyze a sample
synth.load_sample('samples/test_samples/piano-middle-c.wav')

# Synthesize at a new pitch (e.g., 880 Hz = A5)
output_signal = synth.play_note(880.0)

# Save the result
import scipy.io.wavfile as wavfile
wavfile.write('piano_at_a5.wav', 44100, output_signal.astype('int16'))
```

### Run Experiments

```bash
# Experiment 1: Harmonic count sensitivity analysis
python experiments/experiment_1.py

# Experiment 2: Basis function selection patterns
python experiments/experiment_2.py

# Experiment 3: Optimization effectiveness comparison
python experiments/experiment_3.py
```

Results are saved to `experiments/results/`.

## 📁 Project Structure

```
per-harmonic-adsr-synthesis/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── main.py              # Entry point
│   ├── synth.py             # Core synthesis engine (49 KB)
│   └── ui.py                # GUI implementation (38 KB)
├── experiments/
│   ├── experiment_1.py      # Harmonic count analysis
│   ├── experiment_2.py      # Basis function selection
│   ├── experiment_3.py      # Optimization comparison
│   └── results/
│       ├── experiment_1_results.txt
│       ├── experiment_2_results.txt
│       └── experiment_3_results.txt
├── samples/
│   └── test_samples/        # 14 test audio files
│       ├── bass-guitar-*.wav
│       ├── piano-*.wav
│       ├── violin-*.wav
│       ├── clap-*.wav
│       └── voice-*.wav
├── paper/
│   └── DSP_Project_Writeup.pdf  # Full technical report (6 pages)
└── examples/
    └── demo.py              # Usage examples
```

## 🎓 Technical Details

### Signal Model

A periodic musical signal x[n] is represented as:

```
x[n] = Σ(i=0 to K) Aᵢ Êᵢ[n] sin(2πfᵢ n/fs + θᵢ)
```

where:
- `Aᵢ` = amplitude of i-th harmonic
- `fᵢ = i·f₀` = harmonic frequency
- `Êᵢ[n]` = reconstructed envelope
- `θᵢ` = phase offset
- `fs = 44.1 kHz` = sampling rate

### ADSR Basis Functions

Each stage offers multiple kernel options:

| Stage   | Options | Examples |
|---------|---------|----------|
| Attack  | 6       | 2-sample to 6-sample differentiators |
| Decay   | 6       | Symmetric/asymmetric central differences |
| Sustain | 7       | Moving averages (10-100ms) + windowed variants |
| Release | 7       | Applied to log-envelope, 2-20 sample kernels |

**Total configurations per harmonic**: 6 × 6 × 7 × 7 = **1,764 possible combinations**

The greedy sequential optimizer tests **26 configurations** efficiently by optimizing each stage independently.

### Harmonic Isolation: FIR Filters

Kaiser-windowed FIR filters provide:
- **Guaranteed stability** (unlike IIR filters at low frequencies)
- **60 dB stopband attenuation**
- **Adaptive passband**: Δf = min(f₀/4, 50 Hz)
- **Filter order**: 51-1001 taps (odd for Type I symmetry)
- **Zero-phase filtering**: Forward-backward filtering (filtfilt)

## 🎯 Use Cases

- **Digital Audio Workstations (DAWs)**: Artifact-free pitch correction
- **Music Production**: Realistic pitch transposition of sampled instruments
- **Sound Design**: Timbre morphing and envelope manipulation
- **Game Audio**: Dynamic pitch variation without temporal distortion
- **Audio Restoration**: Pitch correction while preserving performance characteristics

## 📈 Limitations & Future Work

### Current Limitations
1. **Monophonic signals only**: Polyphonic input causes harmonic overlap
2. **Harmonic sounds**: Inharmonic instruments (bells, stretched piano strings) not fully captured (10% tolerance provides some robustness)
3. **Static fundamental**: No vibrato or pitch glide support (requires time-varying f₀ tracking)
4. **Exponential release assumption**: Complex release profiles (bowed strings with after-ring) need adaptive models
5. **Greedy optimization**: Sequential optimization can produce local minima for highly heterogeneous harmonics

### Future Enhancements
- [ ] **Inharmonicity modeling**: f_i = i·f₀ + δᵢ for piano/bells
- [ ] **Time-varying fundamental tracking**: Support vibrato and pitch bends
- [ ] **Polyphonic source separation**: Handle chords and multiple simultaneous notes
- [ ] **Machine learning optimization**: Learn optimal basis functions per instrument class
- [ ] **Perceptual metrics**: Replace MSE with psychoacoustic-weighted error metrics
- [ ] **Real-time implementation**: Optimize for low-latency DSP
- [ ] **Joint optimization**: Simultaneous optimization of all ADSR stages

## 📝 Citation

If you use this work in your research or projects, please cite:

```bibtex
@software{ali2024perharmonic,
  author = {Ali, Sekander},
  title = {Per-Harmonic ADSR Synthesis: Parametric Audio Modeling Through Optimized Envelope Reconstruction},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/per-harmonic-adsr-synthesis},
  note = {Digital Signal Processing Final Project, Columbia University ELEN E4810}
}
```

**Academic Paper**: See [`paper/DSP_Project_Writeup.pdf`](paper/DSP_Project_Writeup.pdf) for the complete technical report with detailed methodology, experimental validation, and mathematical derivations.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Course**: ELEN E4810 Digital Signal Processing, Columbia University
- **Instructor**: [Your instructor's name if you want to include]
- **References**:
  - Oppenheim & Schafer - *Discrete-Time Signal Processing*
  - Smith - *Spectral Audio Signal Processing*
  - McAulay & Quatieri - Sinusoidal modeling (IEEE 1986)
  - Serra & Smith - Spectral modeling synthesis (CMJ 1990)

## 🔗 Related Work

- [Phase Vocoder](https://en.wikipedia.org/wiki/Phase_vocoder): Alternative pitch-shifting technique (can introduce phase discontinuities)
- [Sinusoidal Modeling](https://ccrma.stanford.edu/~jos/sasp/): General framework for analysis/synthesis
- [WORLD Vocoder](https://github.com/mmorise/World): High-quality speech synthesis
- [DDSP (Differentiable DSP)](https://github.com/magenta/ddsp): Neural network-based audio synthesis

---

**Author**: Sekander Ali (sna2151)
**Institution**: Columbia University
**Course**: Digital Signal Processing (ELEN E4810)
**Project Type**: Final Project
**Year**: 2024

⭐ **Star this repo** if you find it useful for your audio processing work!
