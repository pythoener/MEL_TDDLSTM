import os
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import librosa
import _lzma
import soundfile as sf
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.signal
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.io.wavfile import write
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from scipy.io import wavfile
from tensorflow import keras
from sklearn.model_selection import train_test_split
from scipy.signal import butter, filtfilt
from pesq import pesq
from pystoi import stoi
from tensorflow.keras.layers import Input, Conv1D, Conv1DTranspose, BatchNormalization, MaxPooling1D, LSTM, Dropout, UpSampling1D, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv1D, Conv1DTranspose, LSTM, Dropout, BatchNormalization, MaxPooling1D, UpSampling1D, Flatten, Reshape



binsperoctave=48
min_window= 512
minf=62
maxf=7900
n_fft=1024
hop_length=128


def plot_spectrogram(signal, sample_rate, title='Spectrogram'):
    D = librosa.amplitude_to_db(np.abs(librosa.stft(signal, n_fft=1024, hop_length=128)), ref=np.max)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(D, sr=sample_rate, hop_length=128, x_axis='time', y_axis='hz')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel(0-16000)
    plt.ylabel('Hz')
    plt.show()

def apply_fft(signal, fs, a):
    fft_signal = np.fft.fft(signal)
    n = len(signal)
    freq = np.fft.fftfreq(n, 1/fs)
    plt.figure(figsize=(10, 3))
    plt.plot(freq, np.abs(fft_signal))  # Plot both frequencies
    plt.title(f'FFT Magnitude of {a}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def lowpass_filter(data, cutoff, fs=16000, order=50):
    nyquist = fs / 2
    normalized_cutoff = cutoff / nyquist
    sos = scipy.signal.butter(order, normalized_cutoff, btype='low', output='sos')
    signal_prefiltered = scipy.signal.sosfilt(sos, data)
    return signal_prefiltered

def extract_stft(frame, sample_rate=16000,
                 n_fft=1024, hop_length=128, win_length=None):
    """
    stft_mag  → 128-bin log-mel magnitude  (shape: 128 × T, *dB*)
    stft_phase → original complex-STFT phase (shape: 513 × T)
    """
    if win_length is None:
        win_length = n_fft

    # complex STFT
    stft_complex = librosa.stft(frame,
                                n_fft=n_fft,
                                hop_length=hop_length,
                                win_length=win_length,
                                window="hamming")
    stft_phase = np.angle(stft_complex)

    # amplitude → mel
    mag = np.abs(stft_complex)                       # amplitude (not power)
    mel_filter = librosa.filters.mel(sr=sample_rate,
                                     n_fft=n_fft,
                                     n_mels=128)
    mel_mag = mel_filter @ mag                       # (128, T)

    # log scale
    stft_mag = 20 * np.log10(np.maximum(mel_mag, 1e-10))

    return stft_mag, stft_phase

def extract_stft_(frame, sample_rate, n_fft=1024,hop_length=128, win_length=None):
    if win_length is None:
        win_length = n_fft
    stft_complex = librosa.stft(frame, n_fft=n_fft, hop_length= hop_length, win_length=win_length, window= "hamming")

    # Compute magnitude and convert to decibels (dB). The dB conversion helps in dynamic range compression.
    #stft_mag = librosa.amplitude_to_db(np.abs(stft_complex), ref=np.max) #not sure bout this one
    stft_mag = 20 * np.log10(np.abs(stft_complex))

    stft_phase = np.angle(stft_complex)

    ################no additional bincopy preperation here############################

    return stft_mag, stft_phase

def load_audio(file_path):
    try:
        # Attempt to load with soundfile
        audio_signal, fs = sf.read(file_path)
    except Exception as e:
        print(f"Soundfile failed: {e}")
        try:
            # Fallback to pydub
            audio = AudioSegment.from_file(file_path)
            audio_signal = np.array(audio.get_array_of_samples())
            fs = audio.frame_rate
        except Exception as e:
            print(f"Pydub also failed: {e}")
            return None, None
    return audio_signal, fs

# PARAMETERS
minW = 512
min_freq = 62
max_freq = 7900
Binsperoctave = 48
fs = 16000
num_bins_additional = 0
binsrequired = 0
Total_bins = 0
num_bins_low = 0

Low_STFT_FEATURES = []
High_STFT_FEATURES = []
Low_STFT_FEATURES_PHASE = []
High_STFT_FEATURES_PHASE = []

def save_audio(final_aud, source_file_path, base_results_dir, fs):
    relative_path = os.path.relpath(source_file_path, start=test_folder)
    save_dir = os.path.join(base_results_dir, os.path.dirname(relative_path))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.basename(source_file_path)
    filepath = os.path.join(save_dir, filename)

    sf.write(filepath, final_aud, fs)
    print(f"Audio saved to {filepath}")

def create_dataset_stft(file_path):

   audio_signal, fs = load_audio(file_path)
   if audio_signal is None or fs is None:
        return
   frame_length = int(0.032 * fs)

   #padding the signal to make it equal to minimum window length
   pad_width = (frame_length - len(audio_signal) % frame_length) % frame_length
   audio_signal = np.pad(audio_signal, (0, pad_width), mode='constant')

   # Highband STFT: full frequency range (up to Nyquist)
   high_stft_mag, high_stft_phase = extract_stft(audio_signal.astype(np.float32), fs, n_fft=n_fft,
                                                 hop_length=hop_length)
   High_STFT_FEATURES.extend(high_stft_mag.T)  # Transposing to match [frames, bins] if needed.
   High_STFT_FEATURES_PHASE.extend(high_stft_phase.T)

   # Lowband STFT: Apply a lowpass filter with a cutoff (e.g., 4000 Hz), then extract STFT features.
   low_audio = lowpass_filter(audio_signal, cutoff=4000, fs=fs)
   low_stft_mag, low_stft_phase = extract_stft(low_audio.astype(np.float32), fs, n_fft=n_fft, hop_length=hop_length)

   #################################################################################
   #bin copy logic completely deleted maybe consider to include this afterwards
   #################################################################################

   Low_STFT_FEATURES.extend(low_stft_mag.T)  # Adjust the shape accordingly.
   Low_STFT_FEATURES_PHASE.extend(low_stft_phase.T)

# Process the TIMIT dataset
target_folder = r'DSET\train\clean'

drfolders = os.listdir(target_folder)

file_limit = 1  # Set the limit to 1 file for testing
processed_files = 0
Timit = 0
for root, dirs, files in os.walk(target_folder):
    for file in files:
        if file.endswith('.wav'):
            file_pathlow = os.path.join(root, file)
            Timit += 1
            processed_files += 1
            create_dataset_stft(file_pathlow)
            if processed_files >= file_limit:
                break
    if processed_files >= file_limit:
        break


def scale_additional_low_bins(low_cqt, original_cqt, total_bins, min_bin):
    low_cqt = np.array(low_cqt)
    original_cqt = np.array(original_cqt)

    original_bins_power = calculate_power(original_cqt[:, min_bin:total_bins])
    original_bins_power_mean = np.mean(original_bins_power)
    print("original_bins_power_mean:", original_bins_power_mean)

    low_cqt_bins_power = calculate_power(low_cqt[:, min_bin:total_bins])
    low_cqt_bins_power_mean = np.mean(low_cqt_bins_power)
    print("low_cqt_bins_power_mean:", low_cqt_bins_power_mean)

    scaling_factor = np.sqrt(original_bins_power_mean / low_cqt_bins_power_mean)
    print("scaling_factor:", scaling_factor)

    scaled_low_cqt = low_cqt.copy()
    scaled_low_cqt[:, min_bin:total_bins] *= scaling_factor
    return scaled_low_cqt

Low_STFT_FEATURES = np.array(Low_STFT_FEATURES)
High_STFT_FEATURES = np.array(High_STFT_FEATURES)
Low_STFT_FEATURES_PHASE = np.array(Low_STFT_FEATURES_PHASE)
High_STFT_FEATURES_PHASE = np.array(High_STFT_FEATURES_PHASE)

from sklearn.preprocessing import StandardScaler
feature_scaler = StandardScaler()
feature_scaler.fit(Low_STFT_FEATURES)
X_train_normalized = feature_scaler.transform(Low_STFT_FEATURES)

label_scaler = StandardScaler()
label_scaler.fit(High_STFT_FEATURES)
#y_train_normalized = label_scaler.transform(High_CQT_FEATURES)

#Xq_train, Xq_temp, y_train, y_temp = train_test_split(X_train_normalized, y_train_normalized, test_size=0.2, random_state=42)
#Xq_val, Xq_test, y_val, y_test = train_test_split(Xq_temp, y_temp, test_size=0.2, random_state=42)

#Xq_train = np.array(Xq_train)
#Xq_val = np.array(Xq_val)
#Xq_test = np.array(Xq_test)
#y_train = np.array(y_train)
#y_val = np.array(y_val)
#y_test = np.array(y_test)

#print(f"Xq_train shape: {Xq_train.shape}, y_train shape: {y_train.shape}")
#print(f"Xq_val shape: {Xq_val.shape}, y_val shape: {y_val.shape}")

model = load_model('Model_VCTK.keras')
model.summary()



def create_sliding_windows_features(X, window_size):
    """
    Converts a 2D feature matrix X of shape (n_frames, 336)
    into overlapping windows of shape (n_windows, window_size, 336),
    where n_windows = n_frames - window_size + 1.
    """
    windows = []
    n_frames = X.shape[0]
    for i in range(n_frames - window_size + 1):
        windows.append(X[i : i + window_size])
    return np.array(windows)

def combine_window_predictions(pred_windows, window_size):
    """
    Combines overlapping window predictions (of shape (n_windows, window_size, 336))
    into a single feature matrix of shape (n_frames, 336) by averaging overlapping windows.
    """
    n_windows = pred_windows.shape[0]
    n_frames = n_windows + window_size - 1
    combined = np.zeros((n_frames, pred_windows.shape[2]))
    counts = np.zeros((n_frames, 1))
    for i in range(n_windows):
        combined[i : i + window_size] += pred_windows[i]
        counts[i : i + window_size] += 1
    return combined / counts


def reconstruct_audio(lowaudio, sample_rate, highAudio, mirrored_low_audio):
    # Extract STFT features (log-magnitude and phase) from the low-band input.
    # low_stft_db: shape (n_bins, n_frames) and low_phase: same shape.
    low_stft_db, low_phase = extract_stft(lowaudio.astype(np.float32), sample_rate,
                                          n_fft=1024, hop_length=128)

    low_stft_db_, low_phase = extract_stft_(lowaudio.astype(np.float32), sample_rate,
                                          n_fft=1024, hop_length=128)

    # Extract the phase from the mirrored low audio (to be used in inversion).
    _, mirrored_phase = extract_stft_(mirrored_low_audio.astype(np.float32), sample_rate,
                                     n_fft=1024, hop_length=128)

    window_size = 5
    n_feats = low_stft_db.shape[0]  # =128
    X_windows = create_sliding_windows_features(low_stft_db.T, window_size)
    n_windows = X_windows.shape[0]

    X_windows_flat = X_windows.reshape(-1, n_feats)
    X_windows_norm = feature_scaler.transform(X_windows_flat).reshape(
        n_windows, window_size, n_feats)

    # ── 3.  model prediction ───────────────────────────────────────────────────
    pred_windows = model.predict(X_windows_norm, verbose=0)
    pred_combined = combine_window_predictions(pred_windows, window_size)

    # inverse-scale
    y_pred_log_mag = label_scaler.inverse_transform(pred_combined)

    # ── 4.  log-mel ➜ linear STFT magnitude (513 bins) ─────────────────────────
    mel_linear = 10 ** (y_pred_log_mag.T / 20)  # (128, T) amplitude
    y_pred_linear = librosa.feature.inverse.mel_to_stft(
        mel_linear, sr=sample_rate,
        n_fft=1024, power=1.0)  # (513, T)

    # low-band magnitude: same mel→STFT conversion for the input
    low_Mag_linear = librosa.feature.inverse.mel_to_stft(
        10 ** (low_stft_db / 20), sr=sample_rate,
        n_fft=1024, power=1.0)  # (513, T)

    # ── 5.  4 kHz cut-off stitching (unchanged) ───────────────────────────────
    cutoff_bin = int(4000 / (sample_rate / 1024))  # ≈ 256
    y_pred_linear_T = y_pred_linear  # already (513, T)

    combined_magnitude = np.zeros_like(low_Mag_linear)
    combined_magnitude[:cutoff_bin + 1, :] = low_Mag_linear[:cutoff_bin + 1, :]
    combined_magnitude[cutoff_bin + 1:, :] = y_pred_linear_T[cutoff_bin + 1:, :]

    # ── 6.  ISTFT with mirrored phase ──────────────────────────────────────────
    stft_complex_combined = combined_magnitude * np.exp(
        1j * mirrored_phase)
    reconstructed_signal = librosa.istft(stft_complex_combined,
                                         hop_length=128,
                                         win_length=1024,
                                         window='hamming')
    return reconstructed_signal


def processfile(file_path):
    global tot, stoiavg, Pesavg, best_pesq_score, worst_pesq_score, best_file_path, worst_file_path
    print(f"Processing file: {file_path}")
    try:
        fs = 16000
        originalAudio, fs = librosa.load(file_path, sr=fs)

        frame_length = int(0.032 * fs)

        pad_width = (min_window - len(originalAudio) % min_window) % min_window
        originalAudio = np.pad(originalAudio, (0, pad_width), mode='constant')

        low_audio = lowpass_filter(originalAudio, cutoff=4000, fs=fs)
        mirrored_file_path = file_path.replace(test_folder, test_folder_low)
        mirrored_low_audio, _ = librosa.load(mirrored_file_path, sr=fs)
        pad_width = (min_window - len(mirrored_low_audio) % min_window) % min_window
        mirrored_low_audio = np.pad(mirrored_low_audio, (0, pad_width), mode='constant')

        #audio_upsampled = UpsampledAudio(low_audio, fs)

        reconstructed_audio = reconstruct_audio(low_audio.astype(np.float32), fs, originalAudio, mirrored_low_audio)

        minlen = min(len(originalAudio), len(reconstructed_audio))
        originalAudio = originalAudio[:minlen]
        reconstructed_audio = reconstructed_audio[:minlen]

        stoi_score = stoi(originalAudio, reconstructed_audio, fs, extended=False)
        pesq_score = pesq(fs, originalAudio, reconstructed_audio, 'wb')
        print(f'STOI: {stoi_score}, PESQ: {pesq_score}')

        save_audio(reconstructed_audio, file_path, r'RSLTS\predicted', fs)

        if pesq_score > best_pesq_score:
            best_pesq_score = pesq_score
            best_file_path = file_path
        if pesq_score < worst_pesq_score:
            worst_pesq_score = pesq_score
            worst_file_path = file_path

        tot += 1
        stoiavg += stoi_score
        Pesavg += pesq_score

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

test_folder = r'DSET\test\clean'
test_folder_low = r'DSET\test\upsampled'

results = []

tot = 0
stoiavg = 0.0
Pesavg = 0.0

best_pesq_score = -np.inf
worst_pesq_score = np.inf
best_file_path = ""
worst_file_path = ""

for root, dirs, files in os.walk(test_folder):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            processfile(file_path)

if tot > 0:
    print('Averages')
    print(f'Average STOI: {stoiavg/tot}, Average PESQ: {Pesavg/tot}')
else:
    print('No valid .wav files found in the directory.')

print(f'Highest PESQ score: {best_pesq_score} from file: {best_file_path}')
print(f'Lowest PESQ score: {worst_pesq_score} from file: {worst_file_path}')

def save_audio(final_aud, source_file_path, base_results_dir, fs):
    relative_path = os.path.relpath(source_file_path, start=test_folder)
    save_dir = os.path.join(base_results_dir, os.path.dirname(relative_path))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    filename = os.path.basename(source_file_path)
    filepath = os.path.join(save_dir, filename)

    sf.write(filepath, final_aud, fs)
    print(f"Audio saved to {filepath}")

