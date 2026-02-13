import numpy as np
import torch

from modules.base.module import BaseModule
from audio.analysis.fft.stft import STFTConfig, stft_ri, istft_from_ri
from evaluation.brain.harmonicus import Harmonicus
from audio.analysis.fft.add_harmonics import generate_harmonics

class TorchBoundaryModule(BaseModule):
    def __init__(
            self,
            brain: Harmonicus,
            stft_cfg: STFTConfig,
            device: str | None = None,
        ):

        # Initialize base module
        super().__init__()
        self.brain = brain
        self.stft_cfg = stft_cfg
        self.device = torch.device(device) if device else torch.device("cpu")

    # Process method to handle numpy input and output
    def process(self, audio: np.ndarray, context) -> np.ndarray:

        if not isinstance(audio, np.ndarray):
            raise TypeError("Input audio must be a numpy array")
        
        if audio.ndim != 1:
            raise ValueError("Input audio numpy array must be 1D")

        if audio.dtype != np.float32:
            audio = audio.astype(np.float32, copy=False)

        # Store original length for ISTFT
        length = audio.shape[0]

        # Convert numpy array to torch tensor
        x = torch.from_numpy(audio).to(self.device)

        X_ri = stft_ri(x, self.stft_cfg)
        X_ri = X_ri.unsqueeze(0)  # Add batch dimension

        print(
            "[ML BOUNDARY] → entering Harmonicus | "
            f"X_ri shape={X_ri.shape} dtype={X_ri.dtype} device={X_ri.device}"
        )

        # Pass to Harmonicus the Wise for descision making
        Y_ri = self.brain.forward(X_ri, context)

        # sprinkle some glitter
        f0_data = context.f0_track
        if f0_data is not None:
            print("[ML BOUNDARY] Adding harmonics based on F0 track")
            Y_ri = generate_harmonics(
                Y_ri, 
                f0_track=torch.from_numpy(f0_data["f0"]).to(self.device),
                voiced=torch.from_numpy(f0_data["voiced_flag"]).to(self.device),
                confidence=torch.from_numpy(f0_data["voiced_probs"]).to(self.device),
                sr=context.sample_rate,
                n_fft=self.stft_cfg.n_fft,
            )
        

        print(
            "[ML BOUNDARY] ← exiting Harmonicus | "
            f"Y_ri shape={Y_ri.shape}"
        )
        delta = (Y_ri - X_ri).abs().max().item()
        print(f"[ML BOUNDARY] max |Δ| after Harmonicus = {delta:.6e}")

        Y_ri = Y_ri.squeeze(0)  # Remove batch dimension

        # Convert back to time domain and to numpy array
        # ensure it's float32 and has the same length as the input
        y = istft_from_ri(Y_ri, self.stft_cfg, length=length)

        return y.detach().cpu().numpy().astype(np.float32, copy=False)