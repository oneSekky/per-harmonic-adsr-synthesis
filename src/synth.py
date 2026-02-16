"""
Sekander Ali - sna2151
ELEN E4810 Digital Signal Processing Final Project
Per-Harmonic ADSR Synthesis Engine
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, ifft, fftfreq
from scipy.interpolate import interp1d
import sounddevice as sd
from scipy.io import wavfile
import json

SAMPLE_RATE = 44100
DURATION = 5.0

def midi_to_freq(midi_note):
    return 440.0 * (2.0 ** ((midi_note - 69) / 12.0))


class SynthEngine:

    def __init__(self, attack_idx=0, decay_idx=0, sustain_idx=0, release_idx=0):
        self.attack_options = [
            np.array([0.5, -0.5]),
            np.array([0.25, 0.25, -0.25, -0.25]),
            np.array([0.33, 0, -0.33]),
            np.array([1, -1]),
            np.array([0.2, 0.2, 0.2, -0.2, -0.2, -0.2]),
            np.array([0.4, 0.1, -0.25, -0.25])
        ]

        self.decay_options = [
            np.array([0.5, 0, -0.5]),
            np.array([0.5, -0.5]),
            np.array([0.25, 0.25, 0, -0.25, -0.25]),
            np.array([1, -1]),
            np.array([0.33, 0.33, 0, 0, -0.33, -0.33]),
            np.array([0.4, 0, 0, -0.3, -0.3])
        ]

        self.sustain_options = [
            np.ones(int(0.1 * SAMPLE_RATE)) / int(0.1 * SAMPLE_RATE),
            np.ones(int(0.05 * SAMPLE_RATE)) / int(0.05 * SAMPLE_RATE),
            None,
            np.ones(int(0.010 * SAMPLE_RATE)) / int(0.010 * SAMPLE_RATE),
            np.ones(int(0.025 * SAMPLE_RATE)) / int(0.025 * SAMPLE_RATE),
            None,
            None 
        ]

        L_100ms = int(0.1 * SAMPLE_RATE)
        hamming_win = np.hamming(L_100ms)
        self.sustain_options[2] = hamming_win / np.sum(hamming_win)

        hann_win = np.hanning(L_100ms)
        self.sustain_options[5] = hann_win / np.sum(hann_win)

        blackman_win = np.blackman(L_100ms)
        self.sustain_options[6] = blackman_win / np.sum(blackman_win)

        self.release_options = [
            np.array([1, -1]),
            np.array([0.5, 0.5, -0.5, -0.5]),
            np.array([0.33, 0.33, 0.33, -0.33, -0.33, -0.33]),
            np.array([0.5, 0, -0.5]),
            None,
            None,
            None 
        ]

        L_10 = 10
        self.release_options[4] = np.concatenate([np.ones(L_10//2), -np.ones(L_10//2)]) / (L_10//2)

        L_20 = 20
        self.release_options[5] = np.concatenate([np.ones(L_20//2), -np.ones(L_20//2)]) / (L_20//2)

        self.release_options[6] = np.array([0.5, 0.25, 0, 0, 0, -0.25, -0.25, -0.25])

        self.attack_indices = None
        self.decay_indices = None
        self.sustain_indices = None
        self.release_indices = None

        self.attack_idx = attack_idx
        self.decay_idx = decay_idx
        self.sustain_idx = sustain_idx
        self.release_idx = release_idx

        self.phi_A = self.attack_options[attack_idx]
        self.phi_D = self.decay_options[decay_idx]
        self.phi_S = self.sustain_options[sustain_idx]
        self.phi_R = self.release_options[release_idx]

        self.f0 = None
        self.K = None
        self.harmonics = None  # Harmonic frequencies 
        self.amplitudes = None  # Relative amplitudes 
        self.phases = None
        self.envelopes_reconstructed = None
        self.transition_points = None  # [(ni1, ni2, ni3) for i in range(K)]
        self.original_duration = 1.0 #length of input sample, 1s is placeholder

        self.optimization_errors = None

    def set_basis_functions(self, attack_idx, decay_idx, sustain_idx, release_idx):
        """Update basis functions"""
        self.attack_idx = attack_idx
        self.decay_idx = decay_idx
        self.sustain_idx = sustain_idx
        self.release_idx = release_idx

        self.phi_A = self.attack_options[attack_idx]
        self.phi_D = self.decay_options[decay_idx]
        self.phi_S = self.sustain_options[sustain_idx]
        self.phi_R = self.release_options[release_idx]

    def get_basis_function_labels(self):
        """Get basis function labels"""
        attack_labels = [
            "ATK0: Sharp 2-sample",
            "ATK1: Smooth 4-sample",
            "ATK2: Medium 3-sample",
            "ATK3: Very sharp instant",
            "ATK4: Very smooth 6-sample",
            "ATK5: Asymmetric fast"
        ]
        decay_labels = [
            "DEC0: Medium 3-sample",
            "DEC1: Sharp 2-sample",
            "DEC2: Smooth 5-sample",
            "DEC3: Very sharp instant",
            "DEC4: Very smooth 6-sample",
            "DEC5: Asymmetric tail"
        ]
        sustain_labels = [
            "SUS0: Uniform 100ms",
            "SUS1: Uniform 50ms",
            "SUS2: Hamming 100ms",
            "SUS3: Uniform 10ms",
            "SUS4: Uniform 25ms",
            "SUS5: Hann 100ms",
            "SUS6: Blackman 100ms"
        ]
        release_labels = [
            "REL0: Sharp 2-sample",
            "REL1: Smooth 4-sample",
            "REL2: Very smooth 6-sample",
            "REL3: Medium 3-sample",
            "REL4: Long 10-sample",
            "REL5: Very long 20-sample",
            "REL6: Multi-scale"
        ]
        return attack_labels, decay_labels, sustain_labels, release_labels

    def optimize_basis_functions(self, envelopes):
        """
        Per-harmonic optimization: find best basis functions for each harmonic independently.
        Uses efficient approach - only reconstructs segments being optimized.

        Args:
            envelopes: Extracted envelopes (K × N) normalized to [0,1]

        Returns:
            attack_indices, decay_indices, sustain_indices, release_indices (K-length arrays),
            per_harmonic_errors (K×4), total_error
        """
        import time
        start_time = time.time()

        print("\n" + "="*80)
        print("OPTIMIZING BASIS FUNCTIONS (Per-Harmonic - MSE Metric)")
        print("="*80)

        K, N = envelopes.shape
        print(f"Optimizing {K} harmonics independently (26 tests per harmonic)...")
        print(f"Estimated time: ~{K*26*0.12:.0f}s\n")

        attack_indices = np.zeros(K, dtype=int)
        decay_indices = np.zeros(K, dtype=int)
        sustain_indices = np.zeros(K, dtype=int)
        release_indices = np.zeros(K, dtype=int)
        per_harmonic_errors = np.zeros((K, 4))

        attack_labels, decay_labels, sustain_labels, release_labels = self.get_basis_function_labels()

        for i in range(K):
            freq_str = f"{self.harmonics[i]:.1f}" if self.harmonics is not None else f"{i+1}"
            print(f"H{i+1}/{K} ({freq_str} Hz): ", end='', flush=True)

            self.set_basis_functions(0, 0, 0, 0)
            _, DEC_init, _, _ = self._extract_adsr_features(envelopes)
            transitions_init = self._detect_transitions(envelopes, DEC_init)
            ni_1_init, ni_2_init, ni_3_init = transitions_init[i]

            clipped_env_i = np.clip(envelopes[i, :], 1e-4, 1.0)
            log_env_i = 20 * np.log10(clipped_env_i)
            log_env_i = np.clip(log_env_i, -80, 0)

            best_attack_error = float('inf')
            best_attack_idx = 0
            ni_1 = ni_1_init

            for attack_idx in range(6):
                phi_A = self.attack_options[attack_idx]
                attack_scale = self._get_reconstruction_scale(phi_A)
                ATK_i = np.convolve(envelopes[i, :], phi_A, mode='same')

                if ni_1 > 0:
                    env_attack = np.cumsum(ATK_i[:ni_1]) * attack_scale
                    error = np.mean((envelopes[i, :ni_1] - env_attack)**2)
                else:
                    error = 0.0

                if error < best_attack_error:
                    best_attack_error = error
                    best_attack_idx = attack_idx

            attack_indices[i] = best_attack_idx
            per_harmonic_errors[i, 0] = best_attack_error

            best_decay_error = float('inf')
            best_decay_idx = 0

            self.set_basis_functions(best_attack_idx, 0, 0, 0)
            _, DEC_temp, _, _ = self._extract_adsr_features(envelopes)
            transitions_temp = self._detect_transitions(envelopes, DEC_temp)
            ni_1, ni_2, ni_3 = transitions_temp[i]

            phi_A_best = self.attack_options[best_attack_idx]
            attack_scale_best = self._get_reconstruction_scale(phi_A_best)
            ATK_i_best = np.convolve(envelopes[i, :], phi_A_best, mode='same')

            for decay_idx in range(6):
                phi_D = self.decay_options[decay_idx]
                decay_scale = self._get_reconstruction_scale(phi_D)
                DEC_i = np.convolve(envelopes[i, :], phi_D, mode='same')

                if ni_2 > ni_1 > 0:
                    env_attack = np.cumsum(ATK_i_best[:ni_1]) * attack_scale_best
                    env_decay = env_attack[-1] + np.cumsum(DEC_i[ni_1:ni_2]) * decay_scale
                    error = np.mean((envelopes[i, ni_1:ni_2] - env_decay)**2)
                else:
                    error = 0.0

                if error < best_decay_error:
                    best_decay_error = error
                    best_decay_idx = decay_idx

            decay_indices[i] = best_decay_idx
            per_harmonic_errors[i, 1] = best_decay_error

            best_sustain_error = float('inf')
            best_sustain_idx = 0

            self.set_basis_functions(best_attack_idx, best_decay_idx, 0, 0)
            _, DEC_temp, _, _ = self._extract_adsr_features(envelopes)
            transitions_temp = self._detect_transitions(envelopes, DEC_temp)
            ni_1, ni_2, ni_3 = transitions_temp[i]

            phi_D_best = self.decay_options[best_decay_idx]
            decay_scale_best = self._get_reconstruction_scale(phi_D_best)
            DEC_i_best = np.convolve(envelopes[i, :], phi_D_best, mode='same')

            if ni_2 > ni_1 > 0:
                env_attack = np.cumsum(ATK_i_best[:ni_1]) * attack_scale_best
                env_decay = env_attack[-1] + np.cumsum(DEC_i_best[ni_1:ni_2]) * decay_scale_best
                decay_end_value = env_decay[-1]
            else:
                decay_end_value = 0.0

            for sustain_idx in range(7):
                phi_S = self.sustain_options[sustain_idx]
                SUS_i = np.convolve(envelopes[i, :], phi_S, mode='same')

                if ni_3 > ni_2:
                    sustain_raw = SUS_i[ni_2:ni_3]
                    if np.max(np.abs(sustain_raw)) > 0:
                        sustain_normalized = sustain_raw / np.max(np.abs(sustain_raw)) * decay_end_value
                        error = np.mean((envelopes[i, ni_2:ni_3] - sustain_normalized)**2)
                    else:
                        error = np.mean((envelopes[i, ni_2:ni_3] - decay_end_value)**2)
                else:
                    error = 0.0

                if error < best_sustain_error:
                    best_sustain_error = error
                    best_sustain_idx = sustain_idx

            sustain_indices[i] = best_sustain_idx
            per_harmonic_errors[i, 2] = best_sustain_error

            best_release_error = float('inf')
            best_release_idx = 0

            self.set_basis_functions(best_attack_idx, best_decay_idx, best_sustain_idx, 0)
            _, DEC_temp, _, _ = self._extract_adsr_features(envelopes)
            transitions_temp = self._detect_transitions(envelopes, DEC_temp)
            ni_1, ni_2, ni_3 = transitions_temp[i]

            phi_S_best = self.sustain_options[best_sustain_idx]
            SUS_i_best = np.convolve(envelopes[i, :], phi_S_best, mode='same')

            if ni_3 > ni_2 > ni_1 > 0:
                env_attack = np.cumsum(ATK_i_best[:ni_1]) * attack_scale_best
                env_decay = env_attack[-1] + np.cumsum(DEC_i_best[ni_1:ni_2]) * decay_scale_best
                decay_end_value = env_decay[-1]
                sustain_raw = SUS_i_best[ni_2:ni_3]
                if np.max(np.abs(sustain_raw)) > 0:
                    sustain_normalized = sustain_raw / np.max(np.abs(sustain_raw)) * decay_end_value
                    release_start_value = sustain_normalized[-1]
                else:
                    release_start_value = decay_end_value
            else:
                release_start_value = 0.0

            for release_idx in range(7):
                phi_R = self.release_options[release_idx]
                REL_i = np.convolve(log_env_i, phi_R, mode='same')

                if ni_3 < N:
                    rel_db_rates = np.clip(REL_i[ni_3:], -10, 10)
                    rel_factors = np.exp(rel_db_rates / 20.0)
                    env_release = release_start_value * np.cumprod(rel_factors)
                    error = np.mean((envelopes[i, ni_3:] - env_release)**2)
                else:
                    error = 0.0

                if error < best_release_error:
                    best_release_error = error
                    best_release_idx = release_idx

            release_indices[i] = best_release_idx
            per_harmonic_errors[i, 3] = best_release_error

            total_harmonic_error = np.mean(per_harmonic_errors[i, :])
            print(f"A{best_attack_idx} D{best_decay_idx} S{best_sustain_idx} R{best_release_idx} (MSE: {total_harmonic_error:.6f})")

        avg_attack_error = np.mean(per_harmonic_errors[:, 0])
        avg_decay_error = np.mean(per_harmonic_errors[:, 1])
        avg_sustain_error = np.mean(per_harmonic_errors[:, 2])
        avg_release_error = np.mean(per_harmonic_errors[:, 3])
        total_error = np.mean(per_harmonic_errors)

        elapsed = time.time() - start_time

        print("\n" + "="*80)
        print(f"OPTIMIZATION COMPLETE ({elapsed:.1f}s)")
        print(f"Per-harmonic basis functions selected (K={K})")
        print(f"Average errors across all harmonics:")
        print(f"  Attack:  {avg_attack_error:.6f}")
        print(f"  Decay:   {avg_decay_error:.6f}")
        print(f"  Sustain: {avg_sustain_error:.6f}")
        print(f"  Release: {avg_release_error:.6f}")
        print(f"  Total (avg MSE): {total_error:.6f}")
        print("="*80)

        print("\nPer-Harmonic Basis Functions:")
        print(f"{'H':>3} {'Freq':>8} {'Amp':>6} {'Phase':>7} {'ATK':>6} {'DEC':>6} {'SUS':>6} {'REL':>6} {'MSE':>10}")
        print("-"*80)

        for i in range(K):
            freq = self.harmonics[i] if self.harmonics is not None else (i+1)
            amp = self.amplitudes[i] if self.amplitudes is not None else 0
            phase = self.phases[i] if self.phases is not None else 0
            phase_deg = np.rad2deg(phase)

            mse = np.mean(per_harmonic_errors[i, :])

            print(f"{i+1:>3} {freq:>8.1f} {amp:>6.3f} {phase_deg:>+7.1f} "
                  f"ATK{attack_indices[i]:>1} DEC{decay_indices[i]:>1} "
                  f"SUS{sustain_indices[i]:>1} REL{release_indices[i]:>1} {mse:>10.6f}")

        print("="*80 + "\n")

        return (attack_indices, decay_indices, sustain_indices, release_indices,
                per_harmonic_errors, total_error)

    def process_sample(self, audio_data, sample_rate):
        """
        Process audio sample through the complete pipeline

        Args:
            audio_data: Input audio
            sample_rate: Sample rate

        Returns:
            f0: Detected fundamental frequency
            K: Number of harmonics
        """
        print("\n" + "="*80)
        print("PROCESSING SAMPLE")
        print("="*80)

        self.attack_indices = None
        self.decay_indices = None
        self.sustain_indices = None
        self.release_indices = None

        attack_labels, decay_labels, sustain_labels, release_labels = self.get_basis_function_labels()
        print(f"Using basis functions:")
        print(f"  Attack:  {attack_labels[self.attack_idx]}")
        print(f"  Decay:   {decay_labels[self.decay_idx]}")
        print(f"  Sustain: {sustain_labels[self.sustain_idx]}")
        print(f"  Release: {release_labels[self.release_idx]}")

        audio_data = self._trim_audio(audio_data, sample_rate)

        self.original_duration = len(audio_data) / sample_rate
        print(f"\nSample duration: {self.original_duration:.3f}s (after trimming)")

        print("\n Harmonic Analysis...")
        self.f0, self.harmonics, self.amplitudes, self.phases = \
            self._analyze_harmonics(audio_data, sample_rate)
        self.K = len(self.harmonics)
        
        print(f"  Fundamental: {self.f0:.2f} Hz")
        print(f"  Harmonics detected: K = {self.K}")
        print(f"  Frequencies: {[f'{f:.1f}' for f in self.harmonics]}")
        print(f"  Amplitudes: {[f'{a:.3f}' for a in self.amplitudes]}")
        print(f"  Phases (rad): {[f'{p:+.3f}' for p in self.phases]}")
        print(f"  Phases (deg): {[f'{np.rad2deg(p):+.1f}' for p in self.phases]}")

        print("\n Filter Bank & Envelope Extraction...")
        envelopes = self._extract_envelopes(audio_data, sample_rate)
        print(f"  Extracted {self.K} envelopes, shape: {envelopes.shape}")

        for i in range(self.K):
            if np.max(envelopes[i, :]) > 0:
                envelopes[i, :] = envelopes[i, :] / np.max(envelopes[i, :])

        (self.attack_indices, self.decay_indices, self.sustain_indices, self.release_indices,
         per_harmonic_errors, total_error) = \
            self.optimize_basis_functions(envelopes)

        self.optimization_errors = {
            'per_harmonic': per_harmonic_errors,  # (K, 4) array
            'attack': np.mean(per_harmonic_errors[:, 0]),
            'decay': np.mean(per_harmonic_errors[:, 1]),
            'sustain': np.mean(per_harmonic_errors[:, 2]),
            'release': np.mean(per_harmonic_errors[:, 3]),
            'total': total_error
        }

        print("\n[Step 3] Extracting ADSR Features...")
        ATK, DEC, SUS, REL = self._extract_adsr_features(envelopes)
        print(f"  ADSR features extracted: ATK, DEC, SUS, REL (each {ATK.shape})")

        print("\n[Step 4] Detecting ADSR Transitions...")
        self.transition_points = self._detect_transitions(envelopes, DEC)
        print(f"  Transition points detected for {len(self.transition_points)} harmonics")

        print("\n[Step 5] Reconstructing Envelopes (Piecewise ADSR)...")
        attack_scale = self._get_reconstruction_scale(self.phi_A)
        decay_scale = self._get_reconstruction_scale(self.phi_D)
        print(f"  Reconstruction scales: Attack={attack_scale:.3f}, Decay={decay_scale:.3f}")
        self.envelopes_reconstructed = self._reconstruct_envelopes(ATK, DEC, SUS, REL, self.transition_points)
        
        if np.any(np.isnan(self.envelopes_reconstructed)) or np.any(np.isinf(self.envelopes_reconstructed)):
            print("  Warning: Found NaN/Inf in reconstructed envelopes, cleaning up...")
            self.envelopes_reconstructed = np.nan_to_num(
                self.envelopes_reconstructed, 
                nan=0.001, 
                posinf=1.0, 
                neginf=0.001
            )
            self.envelopes_reconstructed = np.maximum(self.envelopes_reconstructed, 0.001)
        
        print(f"  Reconstructed envelopes ready: {self.K} harmonics × {self.envelopes_reconstructed.shape[1]} samples")

        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80 + "\n")

        return self.f0, self.K

    def _trim_audio(self, audio, sample_rate):
        """
        Trim audio to remove silence and clicks at beginning and end.
        Uses energy-based onset/offset detection.

        Args:
            audio: Input audio array
            sample_rate: Sample rate in Hz

        Returns:
            Trimmed audio array
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        frame_size = int(0.01 * sample_rate)
        hop_size = frame_size // 2

        energy = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            energy.append(np.sum(frame**2))

        energy = np.array(energy)

        if np.max(energy) > 0:
            energy = energy / np.max(energy)

        onset_threshold = 0.01
        onset_frames = np.where(energy > onset_threshold)[0]

        if len(onset_frames) == 0:
            return audio

        onset_idx = onset_frames[0] * hop_size

        offset_threshold = 0.005
        offset_frames = np.where(energy > offset_threshold)[0]

        if len(offset_frames) > 0:
            offset_idx = min((offset_frames[-1] + 2) * hop_size, len(audio))
        else:
            offset_idx = len(audio)

        padding = int(0.05 * sample_rate)
        onset_idx = max(0, onset_idx - padding)

        trimmed = audio[onset_idx:offset_idx]

        print(f"  Trimmed: {onset_idx/sample_rate:.3f}s to {offset_idx/sample_rate:.3f}s (removed {(len(audio)-len(trimmed))/sample_rate:.3f}s)")

        return trimmed

    def _analyze_harmonics(self, audio, sample_rate):
        """
        Detect fundamental frequency and extract harmonic amplitudes/phases

        Returns:
            f0: Fundamental frequency
            harmonics: Array of harmonic frequencies
            amplitudes: Array of amplitudes relative to fundamental
            phases: Array of phases in radians
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0

        N = len(audio)

        windowed = audio * np.hanning(N)
        fft_mag = fft(windowed)
        freqs = fftfreq(N, 1/sample_rate)

        pos_freqs = freqs[:N//2]
        mag = np.abs(fft_mag[:N//2])

        musical_range = (pos_freqs >= 80) & (pos_freqs <= 1000)
        f0_idx = np.argmax(mag[musical_range])
        f0_idx = np.where(musical_range)[0][f0_idx]
        f0 = pos_freqs[f0_idx]

        fft_phase = fft(audio)

        harmonics = []
        amplitudes = []
        phases_raw = []

        fundamental_amp = mag[f0_idx]
        threshold = 0.005 * fundamental_amp

        nstart = int(0.05 * sample_rate)

        i = 1
        while True:
            target_freq = i * f0

            bin_idx = np.argmin(np.abs(pos_freqs - target_freq))
            harmonic_freq = pos_freqs[bin_idx]
            harmonic_amp = mag[bin_idx]
            
            search_range = 2
            start_bin = max(0, bin_idx - search_range)
            end_bin = min(len(mag), bin_idx + search_range + 1)
            local_peak_idx = start_bin + np.argmax(mag[start_bin:end_bin])
            local_peak_amp = mag[local_peak_idx]
            local_peak_freq = pos_freqs[local_peak_idx]
            
            if local_peak_amp > harmonic_amp * 1.1:  # 10% better
                harmonic_amp = local_peak_amp
                harmonic_freq = local_peak_freq
                bin_idx = local_peak_idx

            freq_error = abs(harmonic_freq - target_freq) / target_freq
            if harmonic_amp < threshold or harmonic_freq >= sample_rate/2 or freq_error > 0.10:
                break

            phase_raw = np.angle(fft_phase[bin_idx])
            phase_advance = 2 * np.pi * harmonic_freq * nstart / sample_rate
            phase_corrected = (phase_raw - phase_advance) % (2 * np.pi)

            harmonics.append(harmonic_freq)
            amplitudes.append(harmonic_amp)
            phases_raw.append(phase_corrected)

            i += 1
            if i > 30:  # breaks after 30 harmonics
                break

        harmonics = np.array(harmonics)
        amplitudes = np.array(amplitudes) / fundamental_amp
        phases = np.array(phases_raw)
        
        if len(phases) > 0:
            phases = phases - phases[0]

        return f0, harmonics, amplitudes, phases

    def _extract_envelopes(self, audio, sample_rate):
        """
        Filter bank + Hilbert transform envelope extraction

        Returns:
            envelopes: K × N matrix of per-harmonic envelopes
        """
        N = len(audio)
        K = self.K
        envelopes = np.zeros((K, N))

        for i in range(K):
            fi = self.harmonics[i]

            delta_f = min(self.f0 / 4, 50.0)

            passband_low = max(fi - delta_f, 1.0)
            passband_high = min(fi + delta_f, sample_rate/2 - 1)
            passband = [passband_low, passband_high]
            
            stopband_low = max(0.1, fi - 2*delta_f)
            stopband_high = min(sample_rate/2 - 0.1, fi + 2*delta_f)
            
            if stopband_low >= passband_low:
                stopband_low = max(0.1, passband_low - delta_f)
            if stopband_high <= passband_high:
                stopband_high = min(sample_rate/2 - 0.1, passband_high + delta_f)
            
            try:
                transition_width = delta_f  # From passband edge to stopband edge
                
                A = 60.0  # Stopband attenuation in dB
                delta_omega = 2 * np.pi * transition_width / sample_rate  # Normalized transition width
                
                if delta_omega > 0:
                    N_taps = int((A - 7.95) / (2.285 * delta_omega)) + 1
                else:
                    N_taps = 201  # Default
                
                if N_taps % 2 == 0:
                    N_taps += 1
                
                N_taps = max(51, min(N_taps, 1001))
                
                try:
                    if A > 50:
                        beta = 0.1102 * (A - 8.7)
                    else:
                        beta = 0.5842 * (A - 21)**0.4 + 0.07886 * (A - 21)
                    
                    b = signal.firwin(
                        numtaps=N_taps,
                        cutoff=passband,
                        pass_zero=False,  # Bandpass (not bandstop)
                        fs=sample_rate,
                        window=('kaiser', beta)  # Kaiser window with calculated beta
                    )
                    a = np.array([1.0])  # FIR filter has no denominator
                    
                except Exception as e1:
                    try:
                        b = signal.firwin(
                            numtaps=N_taps,
                            cutoff=passband,
                            pass_zero=False,
                            fs=sample_rate,
                            window='hamming'
                        )
                        a = np.array([1.0])
                    except Exception as e2:
                        try:
                            nyquist_norm = sample_rate / 2
                            bands = [
                                0.0,
                                max(0.001, (passband_low - transition_width) / nyquist_norm),
                                passband_low / nyquist_norm,
                                passband_high / nyquist_norm,
                                min(0.999, (passband_high + transition_width) / nyquist_norm),
                                1.0
                            ]
                            desired = [0, 1, 0]  # Stop, pass, stop
                            weight = [1, 10, 1]  # Higher weight on passband
                            b = signal.remez(N_taps, bands, desired, weight=weight)
                            a = np.array([1.0])
                        except Exception as e3:
                            N_taps_simple = min(201, N_taps)
                            if N_taps_simple % 2 == 0:
                                N_taps_simple += 1
                            b = signal.firwin(
                                numtaps=N_taps_simple,
                                cutoff=passband,
                                pass_zero=False,
                                fs=sample_rate,
                                window='hamming'
                            )
                            a = np.array([1.0])

                if np.any(np.isnan(b)) or np.any(np.isinf(b)):
                    raise ValueError("FIR filter coefficients contain NaN/Inf")

                filtered = signal.filtfilt(b, a, audio)
                
                if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                    print(f"  Warning: FIR filter produced NaN/Inf for harmonic {i+1} ({fi:.1f} Hz)")
                    b, a = signal.butter(N=4, Wn=passband, btype='bandpass', fs=sample_rate)
                    filtered = signal.filtfilt(b, a, audio)
                    
                    if np.any(np.isnan(filtered)) or np.any(np.isinf(filtered)):
                        print(f"  Warning: Fallback filter also unstable, using envelope fallback for harmonic {i+1}")
                        envelopes[i, :] = np.abs(audio)
                        continue

                analytic = signal.hilbert(filtered)
                envelope = np.abs(analytic)
                
                if np.any(np.isnan(envelope)) or np.any(np.isinf(envelope)):
                    print(f"  Warning: Envelope has NaN/Inf for harmonic {i+1}, using fallback")
                    envelopes[i, :] = np.abs(audio)
                elif np.max(envelope) > 1e6:  # Unreasonably large values
                    print(f"  Warning: Envelope has very large values (max={np.max(envelope):.2e}) for harmonic {i+1}, normalizing")
                    if np.max(envelope) > 0:
                        envelope = envelope / np.max(envelope)
                    envelopes[i, :] = envelope
                else:
                    envelopes[i, :] = envelope

            except Exception as e:
                print(f"  Warning: FIR filter design failed for harmonic {i+1} ({fi:.1f} Hz): {e}")
                try:
                    b, a = signal.butter(N=4, Wn=passband, btype='bandpass', fs=sample_rate)
                    filtered = signal.filtfilt(b, a, audio)
                    if not (np.any(np.isnan(filtered)) or np.any(np.isinf(filtered))):
                        analytic = signal.hilbert(filtered)
                        envelopes[i, :] = np.abs(analytic)
                    else:
                        print(f"    Butterworth fallback also failed, using envelope fallback")
                        envelopes[i, :] = np.abs(audio)
                except Exception as e2:
                    print(f"    All filter methods failed: {e2}, using envelope fallback")
                    envelopes[i, :] = np.abs(audio)

        return envelopes

    def _extract_adsr_features(self, envelopes):
        """
        Convolve envelopes with basis functions to extract ADSR features.
        Uses per-harmonic basis functions if available, otherwise falls back to global.

        Returns:
            ATK, DEC, SUS, REL: Feature matrices (K × N each)
        """
        K, N = envelopes.shape

        ATK = np.zeros((K, N))
        DEC = np.zeros((K, N))
        SUS = np.zeros((K, N))
        REL = np.zeros((K, N))

        clipped_envelopes = np.clip(envelopes, 1e-4, 1.0)
        log_envelopes = 20 * np.log10(clipped_envelopes)
        log_envelopes = np.clip(log_envelopes, -80, 0)

        use_per_harmonic = (self.attack_indices is not None and
                           self.decay_indices is not None and
                           self.sustain_indices is not None and
                           self.release_indices is not None)

        for i in range(K):
            if use_per_harmonic:
                phi_A = self.attack_options[self.attack_indices[i]]
                phi_D = self.decay_options[self.decay_indices[i]]
                phi_S = self.sustain_options[self.sustain_indices[i]]
                phi_R = self.release_options[self.release_indices[i]]
            else:
                phi_A = self.phi_A
                phi_D = self.phi_D
                phi_S = self.phi_S
                phi_R = self.phi_R

            ATK[i, :] = np.convolve(envelopes[i, :], phi_A, mode='same')
            DEC[i, :] = np.convolve(envelopes[i, :], phi_D, mode='same')
            SUS[i, :] = np.convolve(envelopes[i, :], phi_S, mode='same')
            REL[i, :] = np.convolve(log_envelopes[i, :], phi_R, mode='same')

        return ATK, DEC, SUS, REL

    def _detect_transitions(self, envelopes, DEC):
        """
        Detect transition points for each harmonic

        Returns:
            List of (ni1, ni2, ni3) tuples for each harmonic
        """
        K, N = envelopes.shape
        transitions = []

        for i in range(K):
            edge_samples = int(0.01 * SAMPLE_RATE)  # 10ms
            search_end = int(0.9 * N)  # Don't search in last 10%
            search_region = envelopes[i, edge_samples:search_end]
            if len(search_region) > 0:
                ni_1 = edge_samples + np.argmax(search_region)
            else:
                ni_1 = edge_samples  # Fallback

            peak_dec = np.max(np.abs(DEC[i, :]))
            sustain_threshold = 0.01 * peak_dec

            candidates = np.where(np.abs(DEC[i, ni_1:]) < sustain_threshold)[0]
            if len(candidates) > 0:
                ni_2 = ni_1 + candidates[0]
            else:
                ni_2 = min(ni_1 + int(0.5 * SAMPLE_RATE), N - 1)  # Fallback: 0.5s after attack

            min_decay_samples = int(0.05 * SAMPLE_RATE)  # 50ms
            if ni_2 - ni_1 < min_decay_samples:
                ni_2 = min(ni_1 + min_decay_samples, N - 1)

            sustain_region = envelopes[i, ni_2:]

            squared_env = sustain_region**2
            total_remaining_energy = np.sum(squared_env)

            if total_remaining_energy > 1e-10:  # Avoid division by zero
                cumsum_backward = np.cumsum(squared_env[::-1])[::-1]
                energy_fraction = cumsum_backward / total_remaining_energy

                min_sustain_samples = int(0.05 * SAMPLE_RATE)  # 50ms minimum sustain
                candidates = np.where((energy_fraction > 0.95) &
                                     (np.arange(len(energy_fraction)) >= min_sustain_samples))[0]

                if len(candidates) > 0:
                    ni_3 = ni_2 + candidates[0]
                else:
                    ni_3 = int(0.8 * N)
            else:
                ni_3 = int(0.8 * N)

            ni_3 = min(ni_3, int(0.9 * N))

            ni_1 = max(0, min(ni_1, N - 1))
            ni_2 = max(ni_1 + 1, min(ni_2, N - 1))
            ni_3 = max(ni_2 + 1, min(ni_3, N - 1))

            transitions.append((ni_1, ni_2, ni_3))

        return transitions

    def _get_reconstruction_scale(self, basis_func):
        """
        Compute scaling factor for cumsum reconstruction based on basis function.
        For difference-type filters [a, -a], scale = 1/a
        For smoothed differences [a, a, -a, -a], scale ≈ 1/(2a)
        """
        positive_vals = basis_func[basis_func > 0]
        if len(positive_vals) == 0:
            return 1.0  # Fallback

        a = positive_vals[0]
        num_positive = np.sum(basis_func > 0)

        if num_positive > 1:
            return 1.0 / (num_positive * a)
        else:
            return 1.0 / a

    def _reconstruct_envelopes(self, ATK, DEC, SUS, REL, transitions):
        """
        Reconstruct envelopes using per-harmonic basis functions if available.

        Returns:
            Reconstructed envelopes (K × N)
        """
        K, N = ATK.shape
        envelopes_recon = np.zeros((K, N))

        use_per_harmonic = (self.attack_indices is not None and
                           self.decay_indices is not None and
                           self.sustain_indices is not None and
                           self.release_indices is not None)

        for i in range(K):
            if use_per_harmonic:
                phi_A = self.attack_options[self.attack_indices[i]]
                phi_D = self.decay_options[self.decay_indices[i]]
                phi_S = self.sustain_options[self.sustain_indices[i]]
                phi_R = self.release_options[self.release_indices[i]]
            else:
                phi_A = self.phi_A
                phi_D = self.phi_D
                phi_S = self.phi_S
                phi_R = self.phi_R

            L_half = len(phi_S) // 2  # Moving average delay correction

            attack_scale = self._get_reconstruction_scale(phi_A)
            decay_scale = self._get_reconstruction_scale(phi_D)
            ni_1, ni_2, ni_3 = transitions[i]

            env = np.zeros(N)

            if ni_1 > 0:
                env[:ni_1] = attack_scale * np.cumsum(ATK[i, :ni_1])

            if ni_2 > ni_1:
                if ni_1 > 0 and ni_1 <= len(env):
                    decay_start = env[ni_1 - 1]
                else:
                    decay_start = 0
                ni_1_safe = max(0, min(ni_1, N))
                ni_2_safe = max(ni_1_safe, min(ni_2, N))
                if ni_2_safe > ni_1_safe:
                    env[ni_1_safe:ni_2_safe] = decay_start + decay_scale * np.cumsum(DEC[i, ni_1_safe:ni_2_safe])

            if ni_3 > ni_2:
                if ni_2 > 0 and ni_2 <= len(env):
                    decay_end = env[ni_2 - 1]
                else:
                    decay_end = 0

                sus_ref_idx = min(ni_2 + L_half, N - 1)
                sus_ref_val = SUS[i, sus_ref_idx] if sus_ref_idx < N else 0
                delta_i = decay_end - sus_ref_val

                sus_start = min(ni_2 + L_half, N - 1)
                sus_end = min(ni_3 + L_half, N)

                ni_2_safe = max(0, min(ni_2, N))
                ni_3_safe = max(ni_2_safe, min(ni_3, N))

                if sus_end > sus_start and ni_3_safe > ni_2_safe:
                    sus_vals = SUS[i, sus_start:sus_end]
                    write_len = min(len(sus_vals), ni_3_safe - ni_2_safe)
                    if write_len > 0:
                        env[ni_2_safe:ni_2_safe+write_len] = sus_vals[:write_len] + delta_i
                else:
                    if sus_start < N and ni_3_safe > ni_2_safe:
                        env[ni_2_safe:ni_3_safe] = SUS[i, sus_start] + delta_i

            ni_3_safe = max(0, min(ni_3, N))
            if ni_3_safe < N:
                if ni_3_safe > 0 and ni_3_safe <= len(env):
                    release_start = env[ni_3_safe - 1]
                else:
                    release_start = 0
                if ni_3_safe < len(REL[i, :]):
                    rel_db_rates = np.clip(REL[i, ni_3_safe:], -10, 10)  # Limit to reasonable dB rates
                    rel_factors = np.exp(rel_db_rates / 20.0)  # Convert from dB to linear scale
                    if len(rel_factors) > 0:
                        env[ni_3_safe:] = release_start * np.cumprod(rel_factors)

            envelopes_recon[i, :] = env

        envelopes_recon = np.nan_to_num(envelopes_recon, nan=0.0, posinf=1.0, neginf=0.0)

        envelopes_recon = np.maximum(envelopes_recon, 0)
        envelopes_recon = np.minimum(envelopes_recon, 10.0)
        
        for i in range(K):
            if np.max(envelopes_recon[i, :]) > 0:
                envelopes_recon[i, :] = envelopes_recon[i, :] / np.max(envelopes_recon[i, :])

        return envelopes_recon

    def synthesize(self, midi_note, duration=None):
        """
        Synthesize audio at target MIDI note

        Args:
            midi_note: Target MIDI note number
            duration: Duration in seconds 

        Returns:
            Audio waveform
        """
        if self.envelopes_reconstructed is None:
            return None

        F_new = midi_to_freq(midi_note)

        if duration is None:
            duration = min(self.original_duration, 5.0)

        N_samples = int(duration * SAMPLE_RATE)
        t = np.arange(N_samples) / SAMPLE_RATE

        y = np.zeros(N_samples)

        print(f"\nSynthesizing: MIDI {midi_note}, F={F_new:.2f} Hz, Duration={duration:.2f}s")

        print(f"  Harmonics (Amplitude | Phase):")
        harmonics_to_show = min(self.K, 10)
        for i in range(harmonics_to_show):
            phase_deg = np.rad2deg(self.phases[i])
            print(f"    H{i+1}: {self.amplitudes[i]:.3f} | {self.phases[i]:+.3f} rad ({phase_deg:+.1f}°)")
        if self.K > harmonics_to_show:
            print(f"    ... and {self.K - harmonics_to_show} more harmonics")

        envelopes_resampled = self._resample_envelopes(self.envelopes_reconstructed, N_samples)

        for i in range(self.K):
            fi_new = (i + 1) * F_new  # Harmonic frequency at new pitch

            if fi_new >= SAMPLE_RATE / 2:
                print(f"  Skipping harmonic {i+1}: {fi_new:.1f} Hz exceeds Nyquist")
                break

            carrier = np.sin(2 * np.pi * fi_new * t + self.phases[i])

            y += self.amplitudes[i] * envelopes_resampled[i, :] * carrier

        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.max(np.abs(y)) > 0:
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))
            
            target_rms = 0.25
            
            max_scale_by_peak = 0.98 / peak  # Allow up to 0.98 peak
            max_scale_by_rms = target_rms / rms if rms > 0 else 1.0
            
            scale_factor = min(max_scale_by_peak, max_scale_by_rms)
            
            if scale_factor > 1.1:
                y = y * scale_factor
            else:
                y = y / peak * 0.98  # Increased to 0.98 for maximum loudness

        y = np.clip(y, -1.0, 1.0)

        return y

    def synthesize_frequency(self, frequency, duration=None):
        """
        Synthesize audio at a specific frequency (not MIDI note)

        Args:
            frequency: Target frequency in Hz
            duration: Duration in seconds

        Returns:
            Audio waveform
        """
        if self.envelopes_reconstructed is None:
            return None

        if duration is None:
            duration = min(self.original_duration, 5.0)

        N_samples = int(duration * SAMPLE_RATE)
        t = np.arange(N_samples) / SAMPLE_RATE

        y = np.zeros(N_samples)

        print(f"\nSynthesizing: F={frequency:.2f} Hz, Duration={duration:.2f}s")

        envelopes_resampled = self._resample_envelopes(self.envelopes_reconstructed, N_samples)

        for i in range(self.K):
            fi_new = (i + 1) * frequency  # Harmonic frequency at new pitch

            if fi_new >= SAMPLE_RATE / 2:
                print(f"  Skipping harmonic {i+1}: {fi_new:.1f} Hz exceeds Nyquist")
                break

            carrier = np.sin(2 * np.pi * fi_new * t + self.phases[i])

            y += self.amplitudes[i] * envelopes_resampled[i, :] * carrier

        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)

        if np.max(np.abs(y)) > 0:
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y**2))

            target_rms = 0.25
            max_scale_by_peak = 0.98 / peak
            max_scale_by_rms = target_rms / rms if rms > 0 else 1.0

            scale_factor = min(max_scale_by_peak, max_scale_by_rms)

            if scale_factor > 1.1:
                y = y * scale_factor
            else:
                y = y / peak * 0.98

        y = np.clip(y, -1.0, 1.0)

        return y

    def _resample_envelopes(self, envelopes, N_target):
        """
        Resample envelopes to match target duration

        Args:
            envelopes: K × N_original matrix
            N_target: Target number of samples

        Returns:
            Resampled envelopes (K × N_target)
        """
        K, N_original = envelopes.shape
        envelopes_resampled = np.zeros((K, N_target))

        t_old = np.linspace(0, 1, N_original)
        t_new = np.linspace(0, 1, N_target)

        for i in range(K):
            interp_func = interp1d(t_old, envelopes[i, :],
                                   kind='linear',
                                   fill_value='extrapolate')
            envelopes_resampled[i, :] = interp_func(t_new)

        return envelopes_resampled

    def save_parameters(self, filename):
        """Save synthesizer parameters to JSON file"""
        if self.f0 is None:
            return False

        params = {
            'f0': float(self.f0),
            'K': int(self.K),
            'harmonics': self.harmonics.tolist(),
            'amplitudes': self.amplitudes.tolist(),
            'phases': self.phases.tolist(),
            'envelopes': self.envelopes_reconstructed.tolist(),
            'original_duration': float(self.original_duration)
        }

        with open(filename, 'w') as f:
            json.dump(params, f, indent=2)

        print(f"Parameters saved to {filename}")
        return True

    def load_parameters(self, filename):
        """Load synthesizer parameters from JSON file"""
        try:
            with open(filename, 'r') as f:
                params = json.load(f)

            self.f0 = params['f0']
            self.K = params['K']
            self.harmonics = np.array(params['harmonics'])
            self.amplitudes = np.array(params['amplitudes'])
            self.phases = np.array(params['phases'])
            self.envelopes_reconstructed = np.array(params['envelopes'])
            self.original_duration = params['original_duration']

            print(f"Parameters loaded from {filename}")
            return True
        except Exception as e:
            print(f"Error loading parameters: {e}")
            return False

    def is_ready(self):
        """Check if synthesizer is ready to play"""
        return self.envelopes_reconstructed is not None


def record_audio(duration, sample_rate, countdown=1.0):
    """
    Record audio from microphone

    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate
        countdown: Countdown before recording starts

    Returns:
        Recorded audio (mono, float, normalized)
    """
    import time
    print(f"\nGet ready... Recording starts in {countdown:.1f} seconds...")
    time.sleep(countdown)

    print(f"Recording NOW for {duration} seconds...")
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate,
                       channels=1)
    sd.wait()
    print("Recording complete!")

    audio = recording.flatten()
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    return audio


def load_audio_file():
    """Open file dialog to load WAV or MP3 file"""
    from tkinter import Tk, filedialog
    Tk().withdraw()  # Hide tkinter root window
    file_path = filedialog.askopenfilename(
        title="Select Audio File",
        filetypes=[
            ("Audio files", "*.wav *.mp3"),
            ("WAV files", "*.wav"),
            ("MP3 files", "*.mp3"),
            ("All files", "*.*")
        ]
    )
    if file_path:
        try:
            if file_path.lower().endswith('.mp3'):
                try:
                    from pydub import AudioSegment
                    print("Loading MP3 file...")
                    audio_seg = AudioSegment.from_mp3(file_path)

                    samples = np.array(audio_seg.get_array_of_samples())

                    if audio_seg.channels == 2:
                        samples = samples.reshape((-1, 2))
                        samples = np.mean(samples, axis=1)

                    audio_data = samples.astype(float)
                    if np.max(np.abs(audio_data)) > 0:
                        audio_data = audio_data / np.max(np.abs(audio_data))

                    return audio_data, audio_seg.frame_rate

                except ImportError:
                    print("Error: pydub not installed. Install with: pip install pydub")
                    print("You'll also need ffmpeg: https://ffmpeg.org/download.html")
                    return None, None

            else:
                sample_rate, audio = wavfile.read(file_path)

                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)

                audio_data = audio.astype(float)
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

                return audio_data, sample_rate

        except Exception as e:
            print(f"Error loading file: {e}")
            return None, None

    return None, None


if __name__ == "__main__":
    print("Per-Harmonic ADSR Synthesizer")
    print("==============================\n")

    synth = SynthEngine()

    import sys
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Loading: {filepath}")
        sample_rate, audio = wavfile.read(filepath)

        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio = audio.astype(float)
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))

        f0, K = synth.process_sample(audio, sample_rate)

        print("\nSynthesizing at original pitch...")
        audio_synth = synth.synthesize(69)  # A440

        output_file = "synthesized_output.wav"
        wavfile.write(output_file, SAMPLE_RATE,
                     (audio_synth * 32767).astype(np.int16))
        print(f"\nSaved to: {output_file}")
    else:
        print("Usage: python synth.py <audio_file.wav>")
        print("Or import and use with UI")
