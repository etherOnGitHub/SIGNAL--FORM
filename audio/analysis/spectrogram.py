import numpy as np
import librosa

# Compute the mel spectrogram of an audio signal
def compute_mel_spectrogram(
        audio,
        sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        fmin=22,
        fmax=None,
):
    # Compute the mel spectrogram using librosa 
    # return in dB scale
    # mel_DB shape: (n_mels, t)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )

    # Convert to log scale (dB)
    mel_dB = librosa.power_to_db(mel, ref=np.max)

    return mel_dB.astype(np.float32)