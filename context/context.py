
class AudioContext:

    def __init__(self):
        self.f0 = None
        self.sample_rate = None
        self.loudness = None
        self.spectrogram = None

    def summary(self):
        return {
            "f0": self.f0,
            "sample_rate": self.sample_rate,
            "loudness": self.loudness,
            "spectrogram": self.spectrogram is not None,
        }