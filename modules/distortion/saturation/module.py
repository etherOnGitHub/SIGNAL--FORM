from modules.base.module import BaseModule
from audio.dsp.nonlinear import saturation as sat
from audio.dsp.mix.blend import linear_blend
from .enum import SaturationType

# Mapping of SaturationType to corresponding functions
_SATURATION_FUNCTIONS = {
    SaturationType.TANH: sat.tanh_shaper,
    SaturationType.ATAN: sat.atan_shaper,
    SaturationType.HARD_CLIP: sat.hard_clip_shaper,
    SaturationType.POLY: sat.poly_shaper,
    SaturationType.ASYMMETRIC: sat.asymmetric_shaper,
}

# Saturation Module Class
class SaturationModule(BaseModule):
    def __init__(
            self,
            mode: SaturationType = SaturationType.TANH,
            drive: float = 1.0,
            mix: float = 1.0,
        ):
        self.mode = mode
        self.drive = drive
        # Clamp mix between 0.0 and 1.0
        self.mix = float(max(0.0, min(1.0, mix))) 
    
    # Implement the process method, applying the selected saturation
    def process(self, audio, context):
        try:
            fn = _SATURATION_FUNCTIONS[self.mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported saturation mode: {self.mode}") from exc
        
        wet = fn(audio, self.drive)
        return linear_blend(audio, wet, self.mix)