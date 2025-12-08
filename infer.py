import os
import torch
import pystoi
import nisqa
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gtcrn import GTCRN
import subprocess

def compute_spectrogram(channel, n_fft, hop_length):
    n_frames = 1 + (len(channel) - n_fft) // hop_length
    spectrogram = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)

    for i in range(n_frames):
        start = i * hop_length
        end = start + n_fft

        if end > len(channel):
            frame = np.zeros(n_fft)
            frame[:len(channel)-start] = channel[start:]
        else:
            frame = channel[start:end]

        window = np.hanning(n_fft)
        windowed_frame = frame * window
        spectrogram[:, i] = np.fft.rfft(windowed_frame, n=n_fft)
    return spectrogram


def spec_to_db(spectrogram, is_db_=False):
    """Преобразуем комплексную спектрограмму в децибелы"""
    amplitude = np.abs(spectrogram)
    amplitude = np.maximum(amplitude, 1e-10)
    if is_db_:
        amplitude = 20 * np.log10(amplitude)
    return amplitude

def hz_to_mel(freq):
    """Преобразование частоты в Герцах в MEL-шкалу"""
    return 2595 * np.log10(1 + freq / 700)

def mel_to_hz(mel):
    """Обратное преобразование"""
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filters(sr, n_fft, n_mels=100, fmin=0, fmax=None):
    if fmax is None:
        fmax = sr / 2

    fmin_mel = hz_to_mel(fmin)
    fmax_mel = hz_to_mel(fmax)

    mel_points = np.linspace(fmin_mel, fmax_mel, n_mels + 2)

    hz_points = mel_to_hz(mel_points)

    fft_bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    # Создаем матрицу фильтров
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    
    for m in range(n_mels):
        left = fft_bins[m]
        center = fft_bins[m + 1]
        right = fft_bins[m + 2]

        # Ensure indices are within bounds
        left = max(left, 0)
        center = max(center, 0)
        right = min(right, n_fft // 2 + 1)

        if center == left:
            center = left + 1
        if right == center:
            right = center + 1

        if center - left > 0 and center <= filters.shape[1]:
            length = center - left
            filters[m, left:center] = np.linspace(0, 1, length)
        if right - center > 0 and right <= filters.shape[1]:
            length = right - center
            filters[m, center:right] = np.linspace(1, 0, length)

    return filters

def apply_mel_filters(spectrogram, mel_filters):
    # На вход принимает спеткрограмму с амплитудами комплексных чисел
    mel_spectrum = np.dot(mel_filters, spectrogram)
    return mel_spectrum

def plot_mel_spectrograms(mel_left_db, mel_right_db, sr, hop_length, fmin, fmax, size=20):
    """Визуализация MEL-спектрограмм с частотной шкалой"""
    time_left = np.arange(mel_left_db.shape[1]) * hop_length / sr
    time_right = np.arange(mel_right_db.shape[1]) * hop_length / sr

    n_mels = mel_left_db.shape[0]
    fmin_mel = hz_to_mel(fmin)
    fmax_mel = hz_to_mel(fmax)

    mel_points = np.linspace(fmin_mel, fmax_mel, n_mels + 1)
    freq_boundaries = mel_to_hz(mel_points)

    fig, axes = plt.subplots(2, 1, figsize=(size, size))

    im1 = axes[0].imshow(mel_left_db,
                        aspect='auto',
                        origin='lower',
                        cmap='magma',
                        extent=[time_left[0], time_left[-1],
                                freq_boundaries[0], freq_boundaries[-1]])
    axes[0].set_title('MEL-спектрограмма до обработки')
    axes[0].set_xlabel('Время (сек)')
    axes[0].set_ylabel('Частота (Гц)')
    plt.colorbar(im1, ax=axes[0], label='дБ')

    im2 = axes[1].imshow(mel_right_db,
                        aspect='auto',
                        origin='lower',
                        cmap='magma',
                        extent=[time_right[0], time_right[-1],
                                freq_boundaries[0], freq_boundaries[-1]])
    axes[1].set_title('MEL-спектрограмма после обработки')
    axes[1].set_xlabel('Время (сек)')
    axes[1].set_ylabel('Частота (Гц)')
    plt.colorbar(im2, ax=axes[1], label='дБ')
    plt.tight_layout()

    plt.savefig('test_wavs/mel_spec_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

def compute_SNR_uplift(original, enhanced):
    # Метрика SNR

    if not isinstance(original, torch.Tensor):
        original = torch.from_numpy(original)
    if not isinstance(enhanced, torch.Tensor):
        enhanced = torch.from_numpy(enhanced)
    
    # Выравниваем длины (берём минимальную)
    min_len = min(len(original), len(enhanced))
    original = original[:min_len]
    enhanced = enhanced[:min_len]
    
    noise = original - enhanced
    snr = 10 * torch.log10(torch.mean(original ** 2) / torch.mean(noise ** 2))
    return snr.item() 

def evaluate_nisqa(audio_path):
    output_dir = 'nisqa_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Команды из README
    cmd = [
        'python', 'run_predict.py',
        '--mode', 'predict_file',
        '--pretrained_model', 'weights/nisqa.tar',
        '--deg', audio_path,
        '--output_dir', output_dir,
        '--ms_channel', '1'
    ]
    try:
        result = subprocess.run(cmd, 
                               capture_output=True, 
                               text=True, 
                               cwd='NISQA')
        
        if result.returncode != 0:
            print(f"Ошибка NISQA:\n{result.stderr}")
            return None
        
        import glob
        result_files = glob.glob(os.path.join(output_dir, '*.csv'))
        
        if result_files:
            # Берем последний созданный файл
            latest_file = max(result_files, key=os.path.getctime)
            df = pd.read_csv(latest_file)
            
            if not df.empty:
                return df.iloc.to_dict()
        
        return None
        
    except Exception as e:
        return None

# Задаем параметры для построения Mel-спектрограммы
frame_size = 2_048
hop_length = 512
n_fft = 2_048
n_mels = 100
fmin = 0
fmax = 20_000

device = torch.device("cpu")
model = GTCRN().eval()
ckpt = torch.load(os.path.join('checkpoints', 'model_trained_on_dns3.tar'), 
                               map_location=device)
model.load_state_dict(ckpt['model'])

mix, fs = sf.read(os.path.join('test_wavs', 'mix.wav'), dtype='float32')
assert fs == 16000

input = torch.stft(torch.from_numpy(mix), 
                   n_fft=512, 
                   hop_length=256, 
                   win_length=512, 
                   window=torch.hann_window(512).pow(0.5), 
                   return_complex=True)

input = torch.view_as_real(input)

with torch.no_grad():
    output = model(input[None])[0]

# Преобразуем вещественный формат в комплексный для istft
output_complex = torch.complex(output[..., 0], output[..., 1])

enh = torch.istft(output_complex, 
                  n_fft=512, 
                  hop_length=256, 
                  win_length=512, 
                  window=torch.hann_window(512).pow(0.5))


sf.write(os.path.join('test_wavs', 'enh.wav'), 
         enh.detach().cpu().numpy(), fs)

audio_path = os.path.abspath('test_wavs/enh.wav')
nisqa_scores = evaluate_nisqa(audio_path)

snr_before = compute_SNR_uplift(mix, mix)
snr_after = compute_SNR_uplift(mix, enh)

metrics = pd.DataFrame({
    'SNR Before': [snr_before],
    'SNR After': [snr_after]
})

metrics.to_csv('test_wavs/metrics.csv', index=False)

# Построим MEL-спектрограммы до и после улучшения
spec_before = compute_spectrogram(mix, n_fft, hop_length)
spec_after = compute_spectrogram(enh.cpu().numpy(), n_fft, hop_length)

# Спектрограммы с амплитудами комплексных чисел
spec_complex_before = spec_to_db(spec_before, False)
spec_complex_after = spec_to_db(spec_after, False)

# Создаем MEL-фильтры
mel_filters = create_mel_filters(fs, n_fft, n_mels, fmin, fmax)
print(mel_filters.shape)

mel_applied_before = apply_mel_filters(spec_complex_before, mel_filters)
mel_db_before = spec_to_db(mel_applied_before, True)
print(mel_db_before.shape)

mel_applied_after = apply_mel_filters(spec_complex_after, mel_filters)
mel_db_after = spec_to_db(mel_applied_after, True)

# Построение спектрограмм и сохранение графиков
plot_mel_spectrograms(mel_db_before, mel_db_after, fs, hop_length, fmin, fmax)
