# import io
# import numpy as np
# import pandas as pd
# import librosa

# def load_and_preprocess_bytes(audio_bytes):
#     import soundfile as sf
#     y, sr = sf.read(io.BytesIO(audio_bytes))
#     if y.ndim > 1:
#         y = np.mean(y, axis=1)
#     y = librosa.resample(y, orig_sr=sr, target_sr=16000)
#     return y, 16000, len(y) / 16000

# def extract_features(y, sr):
#     features = {}
#     # Core acoustic features (like your reference image)
#     features["MDVP:Fo(Hz)"] = np.mean(librosa.yin(y, fmin=50, fmax=500))
#     features["MDVP:Fhi(Hz)"] = np.max(librosa.yin(y, fmin=50, fmax=500))
#     features["MDVP:Flo(Hz)"] = np.min(librosa.yin(y, fmin=50, fmax=500))
#     features["MDVP:jitter(%)"] = np.std(librosa.feature.zero_crossing_rate(y))
#     features["MDVP:jitter(Abs)"] = np.mean(np.abs(np.diff(y)))
#     features["MDVP:RAP"] = np.var(y)
#     features["MDVP:PPQ"] = np.mean(librosa.feature.rms(y=y))
#     features["Jitter:DDP"] = np.std(librosa.feature.spectral_flatness(y=y))
#     features["MDVP:Shimmer"] = np.mean(librosa.feature.spectral_bandwidth(y=y))
#     features["MDVP:Shimmer(dB)"] = np.std(librosa.feature.spectral_bandwidth(y=y))
#     features["Shimmer:APQ3"] = np.mean(librosa.feature.spectral_contrast(y=y)[0])
#     features["Shimmer:APQ5"] = np.mean(librosa.feature.spectral_contrast(y=y)[1])
#     features["MDVP:APQ"] = np.mean(librosa.feature.spectral_centroid(y=y))
#     features["Shimmer:DDA"] = np.var(librosa.feature.spectral_centroid(y=y))
#     features["NHR"] = np.mean(librosa.feature.rms(y=y))
#     features["HNR"] = np.mean(librosa.feature.spectral_rolloff(y=y))
#     features["RPDE"] = np.mean(librosa.feature.spectral_flatness(y=y))
#     features["DFA"] = np.mean(librosa.feature.zero_crossing_rate(y))
#     features["spread1"] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[0])
#     features["spread2"] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[1])
#     features["D2"] = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)[2])
#     features["PPE"] = np.std(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)[1])

#     df = pd.DataFrame(list(features.items()), columns=["Feature Name", "Value"])
#     X = np.array(list(features.values())).reshape(1, -1)
#     return X, df


# feature.py — Parkinson’s voice feature extractor (with scaling)
# Author: Dathu

import io
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import StandardScaler
import soundfile as sf

def load_and_preprocess_bytes(audio_bytes):
    """Load, resample, and normalize audio"""
    y, sr = sf.read(io.BytesIO(audio_bytes))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y / np.max(np.abs(y)) if np.max(np.abs(y)) > 0 else y
    return y.astype(np.float32), 16000, len(y) / 16000


def extract_features(y, sr):
    """Extract 22 Parkinson’s acoustic features + scale them"""
    features = {}

    # Core acoustic features
    f0 = librosa.yin(y, fmin=50, fmax=500)
    f0 = f0[np.isfinite(f0)]
    features["MDVP:Fo(Hz)"] = np.mean(f0) if f0.size else 0.0
    features["MDVP:Fhi(Hz)"] = np.max(f0) if f0.size else 0.0
    features["MDVP:Flo(Hz)"] = np.min(f0) if f0.size else 0.0

    features["MDVP:jitter(%)"] = np.std(librosa.feature.zero_crossing_rate(y))
    features["MDVP:jitter(Abs)"] = np.mean(np.abs(np.diff(y)))
    features["MDVP:RAP"] = np.var(y)
    features["MDVP:PPQ"] = np.mean(librosa.feature.rms(y=y))
    features["Jitter:DDP"] = np.std(librosa.feature.spectral_flatness(y=y))
    features["MDVP:Shimmer"] = np.mean(librosa.feature.spectral_bandwidth(y=y))
    features["MDVP:Shimmer(dB)"] = np.std(librosa.feature.spectral_bandwidth(y=y))
    features["Shimmer:APQ3"] = np.mean(librosa.feature.spectral_contrast(y=y)[0])
    features["Shimmer:APQ5"] = np.mean(librosa.feature.spectral_contrast(y=y)[1])
    features["MDVP:APQ"] = np.mean(librosa.feature.spectral_centroid(y=y))
    features["Shimmer:DDA"] = np.var(librosa.feature.spectral_centroid(y=y))
    features["NHR"] = np.mean(librosa.feature.rms(y=y))
    features["HNR"] = np.mean(librosa.feature.spectral_rolloff(y=y))
    features["RPDE"] = np.mean(librosa.feature.spectral_flatness(y=y))
    features["DFA"] = np.mean(librosa.feature.zero_crossing_rate(y))
    features["spread1"] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[0])
    features["spread2"] = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=2)[1])
    features["D2"] = np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)[2])
    features["PPE"] = np.std(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)[1])

    # Convert to array + dataframe
    feature_names = list(features.keys())
    X = np.array(list(features.values())).reshape(1, -1)
    df = pd.DataFrame(list(features.items()), columns=["Feature Name", "Value"])

    # --- ⚙️ Apply StandardScaler dynamically ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, df
