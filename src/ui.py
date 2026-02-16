import pygame
import numpy as np
import sys
import os
from scipy.io import wavfile
from synth import SynthEngine, SAMPLE_RATE, DURATION, record_audio, load_audio_file
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for pygame
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import io

KEY_TO_NOTE = {
    pygame.K_a: 60,  # C4
    pygame.K_w: 61,  # C#4
    pygame.K_s: 62,  # D4
    pygame.K_e: 63,  # D#4
    pygame.K_d: 64,  # E4
    pygame.K_f: 65,  # F4
    pygame.K_t: 66,  # F#4
    pygame.K_g: 67,  # G4
    pygame.K_y: 68,  # G#4
    pygame.K_h: 69,  # A4
    pygame.K_u: 70,  # A#4
    pygame.K_j: 71,  # B4
    pygame.K_k: 72,  # C5
    pygame.K_o: 73,  # C#5
    pygame.K_l: 74,  # D5
    pygame.K_p: 75,  # D#5
    pygame.K_SEMICOLON: 76,  # E5
    pygame.K_QUOTE: 77,  # F5
    pygame.K_RIGHTBRACKET: 78,  # F#5
    pygame.K_RETURN: 79,  # G5
    pygame.K_BACKSLASH: 80,  # G#5
}


class Button:
    def __init__(self, x, y, w, h, text, color=(200, 200, 200)):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.color = color
        self.hover = False

    def draw(self, screen, font):
        color = tuple(min(c + 30, 255) for c in self.color) if self.hover else self.color
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)
        text_surf = font.render(self.text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                return True
        return False


class Dropdown:
    """Dropdown menu for selecting options"""
    def __init__(self, x, y, w, h, options, default_idx=0, label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.options = options
        self.selected_idx = default_idx
        self.label = label
        self.is_open = False
        self.hover_idx = -1
        self.option_height = h

    def draw(self, screen, font, small_font):
        if self.label:
            label_surf = small_font.render(self.label, True, (0, 0, 0))
            screen.blit(label_surf, (self.rect.x, self.rect.y - 15))

        color = (220, 220, 220) if self.is_open else (255, 255, 255)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        text = self.options[self.selected_idx]
        text_surf = small_font.render(text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 5, self.rect.centery))
        screen.blit(text_surf, text_rect)

        arrow_x = self.rect.right - 15
        arrow_y = self.rect.centery
        arrow = [(arrow_x - 5, arrow_y - 3), (arrow_x + 5, arrow_y - 3), (arrow_x, arrow_y + 3)]
        pygame.draw.polygon(screen, (0, 0, 0), arrow)

        if self.is_open:
            for i, option in enumerate(self.options):
                option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.option_height,
                                         self.rect.width, self.option_height)

                if i == self.hover_idx:
                    option_color = (200, 200, 255)
                else:
                    option_color = (255, 255, 255)

                pygame.draw.rect(screen, option_color, option_rect)
                pygame.draw.rect(screen, (0, 0, 0), option_rect, 1)

                opt_text = small_font.render(option, True, (0, 0, 0))
                opt_rect = opt_text.get_rect(midleft=(option_rect.x + 5, option_rect.centery))
                screen.blit(opt_text, opt_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            if self.is_open:
                mouse_y = event.pos[1]
                for i in range(len(self.options)):
                    option_y = self.rect.bottom + i * self.option_height
                    if option_y <= mouse_y < option_y + self.option_height:
                        if self.rect.x <= event.pos[0] < self.rect.right:
                            self.hover_idx = i
                            return None

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_open:
                mouse_y = event.pos[1]
                for i in range(len(self.options)):
                    option_rect = pygame.Rect(self.rect.x, self.rect.bottom + i * self.option_height,
                                             self.rect.width, self.option_height)
                    if option_rect.collidepoint(event.pos):
                        old_idx = self.selected_idx
                        self.selected_idx = i
                        self.is_open = False
                        self.hover_idx = -1
                        if old_idx != i:
                            return i  # Return new selection
                        return None
                self.is_open = False
            elif self.rect.collidepoint(event.pos):
                self.is_open = not self.is_open

        return None


class PianoKey:
    def __init__(self, x, y, w, h, midi_note, is_black=False, key_label=""):
        self.rect = pygame.Rect(x, y, w, h)
        self.midi_note = midi_note
        self.is_black = is_black
        self.key_label = key_label
        self.pressed = False

    def draw(self, screen, font):
        if self.is_black:
            color = (50, 50, 50) if not self.pressed else (100, 100, 100)
            text_color = (255, 255, 255)
        else:
            color = (255, 255, 255) if not self.pressed else (200, 200, 255)
            text_color = (0, 0, 0)

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        if self.key_label:
            label_surf = font.render(self.key_label, True, text_color)
            label_rect = label_surf.get_rect(center=(self.rect.centerx, self.rect.bottom - 15))
            screen.blit(label_surf, label_rect)


class FilePathInput:
    """Text input box for entering file paths"""
    def __init__(self, x, y, w, h, placeholder="Enter file path..."):
        self.rect = pygame.Rect(x, y, w, h)
        self.placeholder = placeholder
        self.text = ""
        self.active = False
        self.cursor_visible = True
        self.cursor_timer = 0

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                return True  # Signal that Enter was pressed
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KCONST_CTRL):
                pass
            else:
                if event.unicode and event.unicode.isprintable():
                    self.text += event.unicode

        return False

    def draw(self, screen, font):
        color = (255, 255, 230) if self.active else (255, 255, 255)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (100, 100, 100) if self.active else (150, 150, 150), self.rect, 2)

        display_text = self.text if self.text else self.placeholder
        text_color = (0, 0, 0) if self.text else (150, 150, 150)

        text_surf = font.render(display_text, True, text_color)

        text_rect = text_surf.get_rect(midleft=(self.rect.x + 5, self.rect.centery))

        screen.set_clip(self.rect)
        screen.blit(text_surf, text_rect)
        screen.set_clip(None)

        if self.active:
            self.cursor_timer += 1
            if self.cursor_timer > 30:
                self.cursor_visible = not self.cursor_visible
                self.cursor_timer = 0

            if self.cursor_visible and self.text:
                cursor_x = text_rect.x + text_surf.get_width() + 2
                if cursor_x < self.rect.right - 5:
                    pygame.draw.line(screen, (0, 0, 0),
                                   (cursor_x, self.rect.y + 5),
                                   (cursor_x, self.rect.bottom - 5), 2)

    def get_text(self):
        return self.text

    def clear(self):
        self.text = ""
        self.active = False


class SampleButton:
    """Button for selecting a sample file with play button"""
    def __init__(self, x, y, w, h, filename, filepath):
        self.rect = pygame.Rect(x, y, w, h)
        self.filename = filename
        self.filepath = filepath
        self.hover = False
        self.selected = False

        play_btn_size = h - 4
        self.play_btn_rect = pygame.Rect(x + w - play_btn_size - 2, y + 2, play_btn_size, play_btn_size)
        self.play_hover = False

    def draw(self, screen, font):
        if self.selected:
            color = (150, 200, 255)  # Light blue when selected
        elif self.hover:
            color = (220, 220, 220)  # Light gray when hovering
        else:
            color = (255, 255, 255)  # White normally

        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        display_name = self.filename
        if len(display_name) > 25:
            display_name = display_name[:22] + "..."

        text_surf = font.render(display_name, True, (0, 0, 0))
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 10, self.rect.centery))
        screen.blit(text_surf, text_rect)

        play_color = (100, 200, 100) if self.play_hover else (150, 255, 150)
        pygame.draw.rect(screen, play_color, self.play_btn_rect)
        pygame.draw.rect(screen, (0, 0, 0), self.play_btn_rect, 2)

        center_x = self.play_btn_rect.centerx
        center_y = self.play_btn_rect.centery
        size = self.play_btn_rect.width // 3
        triangle = [
            (center_x - size//2, center_y - size),
            (center_x - size//2, center_y + size),
            (center_x + size, center_y)
        ]
        pygame.draw.polygon(screen, (0, 0, 0), triangle)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
            self.play_hover = self.play_btn_rect.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.play_btn_rect.collidepoint(event.pos):
                return 'play'  # Return 'play' for play button
            elif self.rect.collidepoint(event.pos):
                return 'select'  # Return 'select' for main button
        return None


class TextInput:
    """Simple text input box for frequency entry"""
    def __init__(self, x, y, width, height, max_freq=20000):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = ""
        self.active = False
        self.max_freq = max_freq
        self.font = pygame.font.Font(None, 24)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)

        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                try:
                    freq = float(self.text)
                    if 20 <= freq <= self.max_freq:
                        return freq
                    else:
                        self.text = ""  # Clear if out of range
                except ValueError:
                    self.text = ""  # Clear if invalid
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit() or event.unicode == '.':
                if len(self.text) < 8:  # Limit length
                    self.text += event.unicode

        return None

    def draw(self, screen):
        color = (200, 200, 255) if self.active else (240, 240, 240)
        pygame.draw.rect(screen, color, self.rect)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, 2)

        text_surf = self.font.render(self.text if self.text else "Hz", True, (0, 0, 0) if self.text else (150, 150, 150))
        text_rect = text_surf.get_rect(midleft=(self.rect.x + 5, self.rect.centery))
        screen.blit(text_surf, text_rect)


class HarmonicSynthApp:
    def __init__(self, initial_sample_path=None):
        pygame.init()
        pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2)

        self.WIDTH = 1400  # Increased width for per-harmonic table
        self.HEIGHT = 850  # Reduced height - more compact
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Harmonic Synthesizer - Per-Harmonic ADSR")

        self.PIANO_Y_START = 680  # Piano starts at this Y coordinate

        self.font = pygame.font.Font(None, 20)
        self.title_font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 16)

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.PIANO_BG = (240, 240, 240)  # Light gray background for piano area

        self.synth = SynthEngine()

        self.status_text = "Load or record a sample to begin"

        self.last_midi_note = None

        self.screen_state = 'main'

        self.browse_samples_btn = Button(50, 50, 200, 50, "Browse Samples", (100, 200, 255))
        self.record_btn = Button(270, 50, 150, 50, "Record Sample", (255, 200, 200))


        self.original_audio = None
        self.original_sample_rate = None
        self.last_synthesized = None
        self.graph_surface = None

        self.sample_buttons = []
        self.selected_sample = None
        self.currently_loaded_sample = None  # Track which sample is loaded
        self.scan_sample_folder()

        self.create_piano_keys()

        self.freq_input = TextInput(950, self.PIANO_Y_START + 30, 120, 35, max_freq=20000)

        self.freq_play_btn = Button(1075, self.PIANO_Y_START + 30, 70, 35, "Play")

        self.clock = pygame.time.Clock()
        self.running = True

        if initial_sample_path:
            self.load_sample_from_path(initial_sample_path)

    def scan_sample_folder(self):
        """Scan test_samples folder and create buttons for each audio file"""
        self.sample_buttons = []
        sample_folder = os.path.join(os.path.dirname(__file__), "..", "samples", "test_samples")

        if not os.path.exists(sample_folder):
            print(f"Warning: {sample_folder} folder not found")
            return

        files = []
        for file in os.listdir(sample_folder):
            if file.lower().endswith(('.wav', '.mp3')):
                files.append(file)

        files.sort()

        button_width = 450
        button_height = 30
        start_x = 20
        start_y = 120
        spacing = 3

        for i, filename in enumerate(files):
            y = start_y + i * (button_height + spacing)
            filepath = os.path.join(sample_folder, filename)
            btn = SampleButton(start_x, y, button_width, button_height, filename, filepath)
            self.sample_buttons.append(btn)

        print(f"Found {len(files)} sample files in {sample_folder}/")

    def create_piano_keys(self):
        """Create piano keyboard with 2 octaves - positioned at bottom of window"""
        self.piano_keys = []

        white_key_width = 50
        white_key_height = 120
        black_key_width = 30
        black_key_height = 80

        start_x = 50
        start_y = self.PIANO_Y_START + 20  # 20px padding from piano area start

        white_labels = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ';', "'", 'Enter']
        black_labels = ['W', 'E', 'T', 'Y', 'U', 'O', 'P', '', ']', '\\']

        white_idx = 0
        black_idx = 0

        for octave in range(2):
            base_note = 60 + (octave * 12)  # C4, C5
            octave_x = start_x + (octave * 7 * white_key_width)

            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x, start_y, white_key_width, white_key_height,
                                           base_note, False, label))
            white_idx += 1
            label = black_labels[black_idx] if black_idx < len(black_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + white_key_width - black_key_width//2,
                                           start_y, black_key_width, black_key_height,
                                           base_note + 1, True, label))
            black_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 2, False, label))
            white_idx += 1
            label = black_labels[black_idx] if black_idx < len(black_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 2*white_key_width - black_key_width//2,
                                           start_y, black_key_width, black_key_height,
                                           base_note + 3, True, label))
            black_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 2*white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 4, False, label))
            white_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 3*white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 5, False, label))
            white_idx += 1
            label = black_labels[black_idx] if black_idx < len(black_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 4*white_key_width - black_key_width//2,
                                           start_y, black_key_width, black_key_height,
                                           base_note + 6, True, label))
            black_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 4*white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 7, False, label))
            white_idx += 1
            label = black_labels[black_idx] if black_idx < len(black_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 5*white_key_width - black_key_width//2,
                                           start_y, black_key_width, black_key_height,
                                           base_note + 8, True, label))
            black_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 5*white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 9, False, label))
            white_idx += 1
            label = black_labels[black_idx] if black_idx < len(black_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 6*white_key_width - black_key_width//2,
                                           start_y, black_key_width, black_key_height,
                                           base_note + 10, True, label))
            black_idx += 1
            label = white_labels[white_idx] if white_idx < len(white_labels) else ""
            self.piano_keys.append(PianoKey(octave_x + 6*white_key_width, start_y, white_key_width,
                                           white_key_height, base_note + 11, False, label))
            white_idx += 1

    def load_sample_from_path(self, file_path):
        """Load and process a sample from a file path"""
        if not os.path.exists(file_path):
            self.status_text = f"File not found: {file_path}"
            print(f"ERROR: File not found: {file_path}")
            return False

        try:
            from scipy.io import wavfile

            if file_path.lower().endswith('.mp3'):
                try:
                    from pydub import AudioSegment
                    print(f"Loading MP3 file: {file_path}")
                    audio = AudioSegment.from_mp3(file_path)

                    samples = np.array(audio.get_array_of_samples())

                    if audio.channels == 2:
                        samples = samples.reshape((-1, 2))
                        samples = np.mean(samples, axis=1)

                    if audio.sample_width == 2:  # 16-bit
                        samples = samples.astype(float) / 32768.0
                    elif audio.sample_width == 4:  # 32-bit
                        samples = samples.astype(float) / 2147483648.0
                    else:
                        samples = samples.astype(float) / np.max(np.abs(samples))

                    sample_rate = audio.frame_rate
                    audio_data = samples

                except ImportError:
                    self.status_text = "ERROR: pydub not installed for MP3 support"
                    print("ERROR: pydub not installed. Install with: pip install pydub")
                    return False
            else:
                sample_rate, audio_data = wavfile.read(file_path)
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                audio_data = audio_data.astype(float)
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data))

            print(f"Loaded: {file_path}")
            print(f"Sample rate: {sample_rate} Hz, Length: {len(audio_data)/sample_rate:.2f} seconds")

            self.process_sample(audio_data, sample_rate)
            return True

        except Exception as e:
            self.status_text = f"Error loading file: {str(e)}"
            print(f"ERROR loading file: {e}")
            return False

    def process_sample(self, audio_data, sample_rate):
        """Process the audio sample through the synth engine"""
        try:
            self.original_audio = audio_data.copy()
            self.original_sample_rate = sample_rate

            self.status_text = "Analyzing harmonics..."
            fundamental, K = self.synth.process_sample(audio_data, sample_rate)
            self.status_text = f"Ready! F0={fundamental:.1f}Hz, K={K} harmonics"

            self.graph_surface = self.generate_graphs()
        except Exception as e:
            self.status_text = f"Error processing sample: {str(e)}"
            print(f"Error in process_sample: {e}")
            import traceback
            traceback.print_exc()

    def play_note(self, midi_note):
        """Play a note using the synthesized model"""
        if not self.synth.is_ready():
            self.status_text = "Please load or record a sample first!"
            return

        self.last_midi_note = midi_note

        audio = self.synth.synthesize(midi_note)

        if audio is not None:
            self.last_synthesized = audio.copy()

            self.graph_surface = self.generate_graphs()

            audio_int = (audio * 32767).astype(np.int16)
            stereo_audio = np.column_stack((audio_int, audio_int))
            sound = pygame.sndarray.make_sound(stereo_audio)
            sound.play()

    def play_frequency(self, frequency):
        """Play a specific frequency using the synthesized model"""
        if not self.synth.is_ready():
            self.status_text = "Please load or record a sample first!"
            return

        audio = self.synth.synthesize_frequency(frequency)

        if audio is not None:
            self.last_synthesized = audio.copy()

            self.graph_surface = self.generate_graphs()

            audio_int = (audio * 32767).astype(np.int16)
            stereo_audio = np.column_stack((audio_int, audio_int))
            sound = pygame.sndarray.make_sound(stereo_audio)
            sound.play()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if self.screen_state == 'browser':
                self.handle_browser_events(event)
            else:
                self.handle_main_events(event)

    def generate_graphs(self):
        """Generate time/frequency graphs for original and synthesized audio"""
        if self.original_audio is None:
            return None

        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Audio Visualization: Original vs Synthesized', fontsize=14, fontweight='bold')

            t_orig = np.arange(len(self.original_audio)) / self.original_sample_rate
            axes[0, 0].plot(t_orig, self.original_audio, 'b-', linewidth=0.5)
            axes[0, 0].set_title('Original - Time Domain')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Amplitude')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xlim(0, t_orig[-1])  # Show full duration

            fft_orig = np.fft.rfft(self.original_audio)
            freqs_orig = np.fft.rfftfreq(len(self.original_audio), 1/self.original_sample_rate)
            axes[0, 1].plot(freqs_orig, 20*np.log10(np.abs(fft_orig) + 1e-10), 'b-', linewidth=0.5)
            axes[0, 1].set_title('Original - Frequency Domain')
            axes[0, 1].set_xlabel('Frequency (Hz)')
            axes[0, 1].set_ylabel('Magnitude (dB)')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_xlim(0, 5000)  # Show up to 5kHz

            if self.last_synthesized is not None:
                t_synth = np.arange(len(self.last_synthesized)) / SAMPLE_RATE
                axes[1, 0].plot(t_synth, self.last_synthesized, 'r-', linewidth=0.5)
                axes[1, 0].set_title('Synthesized - Time Domain')
                axes[1, 0].set_xlabel('Time (s)')
                axes[1, 0].set_ylabel('Amplitude')
                axes[1, 0].grid(True, alpha=0.3)
                axes[1, 0].set_xlim(0, t_synth[-1])  # Show full duration

                fft_synth = np.fft.rfft(self.last_synthesized)
                freqs_synth = np.fft.rfftfreq(len(self.last_synthesized), 1/SAMPLE_RATE)
                axes[1, 1].plot(freqs_synth, 20*np.log10(np.abs(fft_synth) + 1e-10), 'r-', linewidth=0.5)
                axes[1, 1].set_title('Synthesized - Frequency Domain')
                axes[1, 1].set_xlabel('Frequency (Hz)')
                axes[1, 1].set_ylabel('Magnitude (dB)')
                axes[1, 1].grid(True, alpha=0.3)
                axes[1, 1].set_xlim(0, 5000)  # Show up to 5kHz
            else:
                axes[1, 0].text(0.5, 0.5, 'Play a note to see\nsynthesized waveform',
                               ha='center', va='center', fontsize=12)
                axes[1, 0].set_title('Synthesized - Time Domain')
                axes[1, 1].text(0.5, 0.5, 'Play a note to see\nsynthesized spectrum',
                               ha='center', va='center', fontsize=12)
                axes[1, 1].set_title('Synthesized - Frequency Domain')

            plt.tight_layout()

            canvas = FigureCanvasAgg(fig)
            canvas.draw()

            buf = canvas.buffer_rgba()
            size = canvas.get_width_height()

            surf = pygame.image.frombuffer(buf, size, "RGBA")
            plt.close(fig)

            return surf

        except Exception as e:
            print(f"Error generating graphs: {e}")
            return None

    def play_original_sample(self, filepath):
        """Play the original WAV file without processing"""
        try:
            sample_rate, audio = wavfile.read(filepath)

            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            audio = audio.astype(float)
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))

            audio_int = (audio * 32767).astype(np.int16)
            stereo_audio = np.column_stack((audio_int, audio_int))

            sound = pygame.sndarray.make_sound(stereo_audio)
            sound.play()

            self.status_text = f"Playing original: {os.path.basename(filepath)}"
        except Exception as e:
            self.status_text = f"Error playing file: {e}"
            print(f"Error playing {filepath}: {e}")

    def handle_browser_events(self, event):
        """Handle events for sample browser screen"""
        freq_result = self.freq_input.handle_event(event)
        if freq_result is not None:
            if self.synth.is_ready():
                self.play_frequency(freq_result)
                self.status_text = f"Playing {freq_result:.1f} Hz"
                self.freq_input.text = ""  # Clear input after playing

        if self.freq_play_btn.handle_event(event):
            if self.freq_input.text:
                try:
                    freq = float(self.freq_input.text)
                    if 20 <= freq <= 20000 and self.synth.is_ready():
                        self.play_frequency(freq)
                        self.status_text = f"Playing {freq:.1f} Hz"
                        self.freq_input.text = ""  # Clear after playing
                except ValueError:
                    pass

        if self.freq_input.active and event.type in (pygame.KEYDOWN, pygame.KEYUP):
            return  # Freq input gets all keyboard events

        for btn in self.sample_buttons:
            result = btn.handle_event(event)
            if result == 'play':
                self.play_original_sample(btn.filepath)
                return
            elif result == 'select':
                self.currently_loaded_sample = btn
                for b in self.sample_buttons:
                    b.selected = False
                btn.selected = True
                self.selected_sample = btn
                print(f"Loading: {btn.filename}")
                self.load_sample_from_path(btn.filepath)
                return

        if event.type == pygame.MOUSEBUTTONDOWN:
            for key in sorted(self.piano_keys, key=lambda k: not k.is_black):
                if key.rect.collidepoint(event.pos):
                    key.pressed = True
                    self.play_note(key.midi_note)
                    break

        if event.type == pygame.MOUSEBUTTONUP:
            for key in self.piano_keys:
                key.pressed = False

        if event.type == pygame.KEYDOWN:
            if event.key in KEY_TO_NOTE:
                midi_note = KEY_TO_NOTE[event.key]
                for key in self.piano_keys:
                    if key.midi_note == midi_note:
                        key.pressed = True
                        self.play_note(midi_note)
                        break
            elif event.key == pygame.K_ESCAPE:
                self.screen_state = 'main'

        if event.type == pygame.KEYUP:
            if event.key in KEY_TO_NOTE:
                midi_note = KEY_TO_NOTE[event.key]
                for key in self.piano_keys:
                    if key.midi_note == midi_note:
                        key.pressed = False
                        break

    def handle_main_events(self, event):
        """Handle events for main screen"""
        freq_result = self.freq_input.handle_event(event)
        if freq_result is not None:
            if self.synth.is_ready():
                self.play_frequency(freq_result)
                self.status_text = f"Playing {freq_result:.1f} Hz"
                self.freq_input.text = ""  # Clear input after playing

        if self.freq_play_btn.handle_event(event):
            if self.freq_input.text:
                try:
                    freq = float(self.freq_input.text)
                    if 20 <= freq <= 20000 and self.synth.is_ready():
                        self.play_frequency(freq)
                        self.status_text = f"Playing {freq:.1f} Hz"
                        self.freq_input.text = ""  # Clear after playing
                except ValueError:
                    pass

        if self.freq_input.active and event.type in (pygame.KEYDOWN, pygame.KEYUP):
            return  # Freq input gets all keyboard events

        if self.browse_samples_btn.handle_event(event):
            self.screen_state = 'browser'
            return

        if self.record_btn.handle_event(event):
            self.status_text = "Recording in 1 second..."
            pygame.display.flip()
            audio_data = record_audio(DURATION, SAMPLE_RATE, countdown=1.0)
            self.process_sample(audio_data, SAMPLE_RATE)

        if event.type == pygame.MOUSEBUTTONDOWN:
            for key in sorted(self.piano_keys, key=lambda k: not k.is_black):
                if key.rect.collidepoint(event.pos):
                    key.pressed = True
                    self.play_note(key.midi_note)
                    break

        if event.type == pygame.MOUSEBUTTONUP:
            for key in self.piano_keys:
                key.pressed = False

        if event.type == pygame.KEYDOWN:
            if event.key in KEY_TO_NOTE:
                midi_note = KEY_TO_NOTE[event.key]
                for key in self.piano_keys:
                    if key.midi_note == midi_note:
                        key.pressed = True
                        self.play_note(midi_note)
                        break

        if event.type == pygame.KEYUP:
            if event.key in KEY_TO_NOTE:
                midi_note = KEY_TO_NOTE[event.key]
                for key in self.piano_keys:
                    if key.midi_note == midi_note:
                        key.pressed = False
                        break

        self.record_btn.handle_event(event)
        self.browse_samples_btn.handle_event(event)
        self.freq_play_btn.handle_event(event)

    def draw(self):
        self.screen.fill(self.WHITE)

        if self.screen_state == 'browser':
            self.draw_browser()
        else:
            self.draw_main()

        self.draw_piano()

        pygame.display.flip()

    def draw_piano(self):
        """Draw piano keyboard area - always visible at bottom of window"""
        piano_area = pygame.Rect(0, self.PIANO_Y_START, self.WIDTH, self.HEIGHT - self.PIANO_Y_START)
        pygame.draw.rect(self.screen, self.PIANO_BG, piano_area)

        pygame.draw.line(self.screen, self.BLACK, (0, self.PIANO_Y_START), (self.WIDTH, self.PIANO_Y_START), 3)

        for key in self.piano_keys:
            if not key.is_black:
                key.draw(self.screen, self.font)
        for key in self.piano_keys:
            if key.is_black:
                key.draw(self.screen, self.font)

        freq_label = self.font.render("Direct Frequency:", True, self.BLACK)
        self.screen.blit(freq_label, (950, self.PIANO_Y_START + 5))
        self.freq_input.draw(self.screen)
        self.freq_play_btn.draw(self.screen, self.font)

    def draw_browser(self):
        """Draw sample browser screen (content area only - piano is separate)"""
        title = self.title_font.render("Sample Browser - Click to Load & Play", True, self.BLACK)
        self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 10))

        instructions = self.font.render("Click a sample to load it, then use the piano keyboard below. Press ESC to go back.", True, self.BLACK)
        self.screen.blit(instructions, (20, 50))

        samples_label = self.font.render("Available Samples:", True, self.BLACK)
        self.screen.blit(samples_label, (20, 85))

        for btn in self.sample_buttons:
            if btn.rect.bottom < self.PIANO_Y_START - 10:
                btn.draw(self.screen, self.font)

        if self.currently_loaded_sample:
            loaded_text = f"Currently Loaded: {self.currently_loaded_sample.filename}"
            loaded_surf = self.font.render(loaded_text, True, (0, 150, 0))
            self.screen.blit(loaded_surf, (500, 85))

        status = self.font.render(self.status_text, True, self.BLACK)
        self.screen.blit(status, (500, 110))

        if self.graph_surface is not None:
            graph_start_y = 140
            graph_end_y = self.PIANO_Y_START - 20  # End before piano with gap
            graph_height = graph_end_y - graph_start_y
            graph_width = self.WIDTH - 520  # Leave room for sample list/dropdowns on left
            scaled_graph = pygame.transform.smoothscale(self.graph_surface, (graph_width, graph_height))
            self.screen.blit(scaled_graph, (500, graph_start_y))

    def draw_main(self):
        """Draw main screen"""
        title = self.title_font.render("Harmonic Synthesizer", True, self.BLACK)
        self.screen.blit(title, (self.WIDTH//2 - title.get_width()//2, 10))

        self.browse_samples_btn.draw(self.screen, self.font)
        self.record_btn.draw(self.screen, self.font)

        status = self.font.render(self.status_text, True, self.BLACK)
        self.screen.blit(status, (50, 130))

        if self.graph_surface is not None:
            graph_start_y = 160
            graph_end_y = self.PIANO_Y_START - 20  # End before piano with gap
            graph_height = graph_end_y - graph_start_y
            graph_width = self.WIDTH - 520  # Leave room for dropdowns on left
            scaled_graph = pygame.transform.smoothscale(self.graph_surface, (graph_width, graph_height))
            self.screen.blit(scaled_graph, (480, graph_start_y))
        else:
            instructions = [
                "Welcome to Harmonic Synthesizer!",
                "",
                "Click 'Browse Samples' to:",
                "  - View all available test samples",
                "  - Click any sample to load it instantly",
                "  - Green play button (>) plays the original audio",
                "  - Play synthesized notes with the on-screen piano keyboard",
                "",
                "Or click 'Record Sample' to record your own audio",
                "",
                "Graphs will appear automatically when you load a sample"
            ]
            y_start = 220
            for i, line in enumerate(instructions):
                text = self.font.render(line, True, self.BLACK)
                self.screen.blit(text, (50, y_start + i * 25))

    def run(self):
        while self.running:
            self.handle_events()
            self.draw()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()
