import sys
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel,
    QFileDialog, QComboBox, QHBoxLayout, QLineEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AudioProcessor(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Procesador de Audio")
        self.setGeometry(100, 100, 1000, 600)

        self.audio = None
        self.sr = None
        self.filtered_audio = None

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Botón Cargar archivo
        self.load_btn = QPushButton("Cargar archivo")
        self.load_btn.clicked.connect(self.load_audio)
        layout.addWidget(self.load_btn)

        # Visualización
        self.plot_original = FigureCanvas(Figure(figsize=(5, 2.5)))
        self.plot_filtered = FigureCanvas(Figure(figsize=(5.5, 3)))
        layout.addWidget(QLabel("Señal Original"))
        layout.addWidget(self.plot_original)
        layout.addWidget(QLabel("Señal Filtrada"))
        layout.addWidget(self.plot_filtered)

        # Controles de Filtro
        filter_layout = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["Pasa-bajas", "Pasa-altas", "Pasa-banda"])
        self.cutoff_input = QLineEdit("1000")  # Frecuencia de corte
        self.order_input = QLineEdit("5")      # Orden del filtro
        self.apply_filter_btn = QPushButton("Aplicar Filtro")
        self.apply_filter_btn.clicked.connect(self.apply_filter)

        filter_layout.addWidget(QLabel("Filtro:"))
        filter_layout.addWidget(self.filter_combo)
        filter_layout.addWidget(QLabel("Frecuencia de corte:"))
        filter_layout.addWidget(self.cutoff_input)
        filter_layout.addWidget(QLabel("Orden:"))
        filter_layout.addWidget(self.order_input)
        filter_layout.addWidget(self.apply_filter_btn)
        layout.addLayout(filter_layout)

        # Transformada
        self.fft_btn = QPushButton("Aplicar Transformada de Fourier")
        self.fft_btn.clicked.connect(self.plot_fft)
        layout.addWidget(self.fft_btn)

        self.fft_canvas = FigureCanvas(Figure(figsize=(5, 2.5)))
        layout.addWidget(self.fft_canvas)

        # Guardar resultado
        self.save_btn = QPushButton("Guardar Resultado")
        self.save_btn.clicked.connect(self.save_audio)
        layout.addWidget(self.save_btn)

        self.setLayout(layout)

    def load_audio(self):
        filename, _ = QFileDialog.getOpenFileName(
            self, "Cargar archivo", "", "Archivos de audio (*.wav *.mp3 *.aac)"
        )
        if filename:
            self.audio, self.sr = librosa.load(filename, sr=None)
            self.plot_signal(self.plot_original.figure, self.audio, self.sr, color='hotpink')  # Rosa
            self.filtered_audio = None
            self.plot_signal(self.plot_filtered.figure, [], self.sr, color='purple')  # Morado

    def butter_filter(self, data, cutoff, fs, order=5, ftype='low'):
        nyq = 0.5 * fs
        if ftype == 'band':
            low, high = cutoff
            low /= nyq
            high /= nyq
            b, a = butter(order, [low, high], btype='band')
        else:
            cutoff /= nyq
            b, a = butter(order, cutoff, btype=ftype)
        return lfilter(b, a, data)

    def apply_filter(self):
        if self.audio is None:
            return
        try:
            ftype = self.filter_combo.currentText()
            cutoff = float(self.cutoff_input.text())
            order = int(self.order_input.text())

            if ftype == "Pasa-bajas":
                self.filtered_audio = self.butter_filter(self.audio, cutoff, self.sr, order, ftype='low')
            elif ftype == "Pasa-altas":
                self.filtered_audio = self.butter_filter(self.audio, cutoff, self.sr, order, ftype='high')
            elif ftype == "Pasa-banda":
                self.filtered_audio = self.butter_filter(self.audio, [cutoff, cutoff * 1.5], self.sr, order, ftype='band')

            self.plot_signal(self.plot_filtered.figure, self.filtered_audio, self.sr, color='purple')  # Morado
        except Exception as e:
            print(f"Error aplicando filtro: {e}")

    def plot_fft(self):
        if self.audio is None:
            return

        ax = self.fft_canvas.figure.subplots()
        ax.clear()

        # FFT de la señal original (rojo)
        N_orig = len(self.audio)
        yf_orig = np.abs(fft(self.audio))[:N_orig // 2]
        xf_orig = fftfreq(N_orig, 1 / self.sr)[:N_orig // 2]
        ax.plot(xf_orig, yf_orig, color='red', label='Original')

        # FFT de la señal filtrada (azul)
        if self.filtered_audio is not None:
            N_filt = len(self.filtered_audio)
            yf_filt = np.abs(fft(self.filtered_audio))[:N_filt // 2]
            xf_filt = fftfreq(N_filt, 1 / self.sr)[:N_filt // 2]
            ax.plot(xf_filt, yf_filt, color='blue', label='Filtrada')

        ax.set_title("Transformada de Fourier")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.legend()
        self.fft_canvas.draw()

    def plot_signal(self, fig, data, sr, color='blue'):
        ax = fig.subplots()
        ax.clear()
        if len(data) > 0:
            t = np.linspace(0, len(data) / sr, len(data))
            ax.plot(t, data, color=color)
            ax.set_xlabel("Tiempo (s)")
            ax.set_ylabel("Amplitud")
        fig.canvas.draw()

    def save_audio(self):
        if self.filtered_audio is None:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Guardar archivo", "", "Archivo WAV (*.wav)"
        )
        if filename:
            sf.write(filename, self.filtered_audio, self.sr)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioProcessor()
    window.show()
    sys.exit(app.exec_())
