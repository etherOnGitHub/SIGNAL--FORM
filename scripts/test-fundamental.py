import sys
from pathlib import Path

# Establish project root and adjust sys.path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

from audio.io.load import load_wav
from audio.analysis.pitch import estimate_f0
from audio.analysis.spectrogram import compute_mel_spectrogram
from audio.analysis.loudness import estimate_loudness_db
from context.context import AudioContext

# Load an example audio file
audio, sr = load_wav("data/dataset-v1/inputs/test.wav")

# Create an AudioContext and analyze the audio
ctx = AudioContext(sample_rate=sr)
# Estimate fundamental frequency (f0)
ctx.f0 = estimate_f0(audio, sr)
# Compute mel spectrogram
ctx.spectrogram = compute_mel_spectrogram(audio, sr)
# Estimate loudness in dB
ctx.loudness = estimate_loudness_db(audio)


# Print summary of the context
print(ctx.summary())

# Print shape of the spectrogram
print(f"Spectrogram shape: {ctx.spectrogram.shape}")