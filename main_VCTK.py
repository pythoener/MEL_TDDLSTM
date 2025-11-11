############################ vctk_large_train.py ############################
# Complete, fully‑commented training script for MEL‑TDDLSTM on VCTK. Both
# small (list) and large (mem‑mapped) modes now execute end‑to‑end, including
# evaluation, plotting and model/scaler persistence.
##############################################################################

import os, warnings, numpy as np, soundfile as sf, librosa, scipy.signal
import matplotlib.pyplot as plt, tensorflow as tf
from pydub import AudioSegment
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Dense, LSTM
import joblib     # for saving/loading StandardScaler objects

##############################################################################
# --------------------------- 1. Helper functions --------------------------- #
##############################################################################

def lowpass_filter(sig, cutoff, fs=16_000, order=50):
    """Return low‑passed version of *sig* with a Butterworth filter."""
    sos = scipy.signal.butter(order, cutoff/(fs/2), 'low', output='sos')
    return scipy.signal.sosfilt(sos, sig)

def extract_stft(frame, sr, n_fft=1024, hop=128, n_mels=128):
    """Compute log‑mel magnitude + phase from a waveform frame."""
    S = np.abs(librosa.stft(frame, n_fft=n_fft, hop_length=hop, window='hamming'))**2
    mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_S = mel @ S + 1e-8  # avoid log(0)
    mag = 10 * np.log10(mel_S)
    phase = np.angle(librosa.stft(frame, n_fft=n_fft, hop_length=hop, window='hamming'))
    return mag, phase

def load_audio(path):
    """Robust audio loader (soundfile → pydub fallback)."""
    try:
        return sf.read(path)
    except Exception as e1:
        try:
            au = AudioSegment.from_file(path)
            return np.array(au.get_array_of_samples()), au.frame_rate
        except Exception as e2:
            warnings.warn(f"Failed loading {path}: {e1}/{e2}"); return None, None

##############################################################################
# -------------------------- 2. Global parameters --------------------------- #
##############################################################################

USE_LARGE_DATASET = True  # True: mem‑map full VCTK, False: small list mode
VCTK_ROOT = r'E:\Raw Data\VCTK_16kHz\train\clean'

# signal & model hyper‑parameters
FS = 16_000; N_FFT = 1024; HOP = 128; N_MELS = 128
WINDOW_SIZE = 5; EPOCHS = 50; BATCH_SIZE = 64; LR = 1e-3
MODEL_OUT = 'Model_VCTK.keras'

##############################################################################
# ------------------------ 3. List‑based small pipeline --------------------- #
##############################################################################
if not USE_LARGE_DATASET:
    Low, High = [], []  # containers for frame‑wise features

    def collect(wav):
        """Extract low/high mel frames from one file and append to lists."""
        sig, sr = load_audio(wav);
        if sig is None: return
        frame = int(0.032*sr); pad=(frame-len(sig)%frame)%frame; sig=np.pad(sig,(0,pad))
        hi,_ = extract_stft(sig.astype(np.float32),sr,N_FFT,HOP,N_MELS)
        lo,_ = extract_stft(lowpass_filter(sig,4000,sr).astype(np.float32), sr,N_FFT,HOP,N_MELS)
        High.extend(hi.T); Low.extend(lo.T)

    for root,_,files in os.walk(VCTK_ROOT):
        for f in files:
            if f.lower().endswith('.wav'): collect(os.path.join(root,f))

    X = np.asarray(Low,'float32'); Y = np.asarray(High,'float32')

    feat_scaler = StandardScaler().fit(X);  X = feat_scaler.transform(X)
    labl_scaler = StandardScaler().fit(Y);  Y = labl_scaler.transform(Y)
    joblib.dump(feat_scaler,'feat_scaler.pkl'); joblib.dump(labl_scaler,'label_scaler.pkl')

    Xtr,Xtmp,ytr,ytmp = tf.keras.utils.train_test_split(X,Y,test_size=0.2,random_state=42)
    Xva,Xte,yva,yte   = tf.keras.utils.train_test_split(Xtmp,ytmp,test_size=0.2,random_state=42)

    def make_windows(A,B,w):
        xs,ys=[],[]
        for i in range(len(A)-w+1): xs.append(A[i:i+w]); ys.append(B[i:i+w])
        return np.asarray(xs,'float32'), np.asarray(ys,'float32')

    Xtr,ytr = make_windows(Xtr,ytr,WINDOW_SIZE)
    Xva,yva = make_windows(Xva,yva,WINDOW_SIZE)
    Xte,yte = make_windows(Xte,yte,WINDOW_SIZE)

##############################################################################
# --------------------- 4. Memory‑mapped large pipeline --------------------- #
##############################################################################
if USE_LARGE_DATASET:

    def count_frames(wav):
        sig,sr=load_audio(wav)
        if sig is None: return 0
        frame=int(0.032*sr); pad=(frame-len(sig)%frame)%frame; sig=np.pad(sig,(0,pad))
        return extract_stft(sig.astype(np.float32),sr,N_FFT,HOP,N_MELS)[0].shape[1]

    wavs=[os.path.join(r,f) for r,_,fs in os.walk(VCTK_ROOT) for f in fs if f.lower().endswith('.wav')]
    total_frames=sum(map(count_frames,wavs)); N_BINS=N_MELS

    lo_mm=np.memmap('low_mel.dat','float32','w+',shape=(total_frames,N_BINS))
    hi_mm=np.memmap('high_mel.dat','float32','w+',shape=(total_frames,N_BINS))

    idx, counts=0,[]
    for p in wavs:
        sig,sr=load_audio(p); frame=int(0.032*sr); pad=(frame-len(sig)%frame)%frame; sig=np.pad(sig,(0,pad))
        hi,_=extract_stft(sig.astype(np.float32),sr,N_FFT,HOP,N_MELS)
        lo,_=extract_stft(lowpass_filter(sig,4000,sr).astype(np.float32),sr,N_FFT,HOP,N_MELS)
        n=hi.shape[1]
        hi_mm[idx:idx+n]=hi.T; lo_mm[idx:idx+n]=lo.T; idx+=n; counts.append(n)
    lo_mm.flush(); hi_mm.flush()

    feat_scaler=StandardScaler().fit(lo_mm); labl_scaler=StandardScaler().fit(hi_mm)
    joblib.dump(feat_scaler,'feat_scaler.pkl'); joblib.dump(labl_scaler,'label_scaler.pkl')
    mu_lo,sig_lo=feat_scaler.mean_,feat_scaler.scale_
    mu_hi,sig_hi=labl_scaler.mean_,labl_scaler.scale_

    def gen():
        s=0
        for n in counts:
            for off in range(n-WINDOW_SIZE+1):
                st=s+off
                yield ((lo_mm[st:st+WINDOW_SIZE]-mu_lo)/sig_lo).astype('float32'), ((hi_mm[st:st+WINDOW_SIZE]-mu_hi)/sig_hi).astype('float32')
            s+=n

    tot=sum(n-WINDOW_SIZE+1 for n in counts); tr=int(0.8*tot); va=int(0.1*tot)
    spec=(tf.TensorSpec((WINDOW_SIZE,N_BINS),tf.float32), tf.TensorSpec((WINDOW_SIZE,N_BINS),tf.float32))
    ds=tf.data.Dataset.from_generator(gen,output_signature=spec)
    train=ds.take(tr).shuffle(10_000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val  =ds.skip(tr).take(va).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test =ds.skip(tr+va).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

##############################################################################
# ----------------------------- 5. Model ------------------------------------ #
##############################################################################
model=Sequential([
    TimeDistributed(Dense(64,activation='relu'),input_shape=(WINDOW_SIZE,N_MELS)),
    TimeDistributed(Dense(32,activation='relu')),
    LSTM(32,return_sequences=True),
    TimeDistributed(Dense(32,activation='relu')),
    TimeDistributed(Dense(64,activation='relu')),
    TimeDistributed(Dense(N_MELS,activation='linear')),
])
model.compile(optimizer=Adam(LR),loss='mse',metrics=['mae'])
model.summary()

##############################################################################
# --------------------------- 6. Training loop ------------------------------ #
##############################################################################
if USE_LARGE_DATASET:
    hist=model.fit(train,epochs=EPOCHS,validation_data=val)
    test_loss,test_mae=model.evaluate(test)
    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")
else:
    hist=model.fit(Xtr,ytr,epochs=EPOCHS,batch_size=BATCH_SIZE,validation_data=(Xva,yva))
    test_loss,test_mae=model
    model.save(MODEL_OUT)
    print(f"Model saved to {MODEL_OUT}")