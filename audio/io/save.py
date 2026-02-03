import soundfile as sf
from pathlib import Path

def save_wav(path, audio, sr):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)