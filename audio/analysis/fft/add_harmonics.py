import torch
import torch.nn.functional as F


def generate_harmonics(
        X_ri: torch.Tensor,
        f0_track: torch.Tensor, 
        voiced: torch.Tensor,
        confidence: torch.Tensor,
        sr: int,
        n_fft: int,
        max_harmonics: int=10000,
        strength: float=10,
        sigma_bins=1.5,
    ):
    B, _, F, T = X_ri.shape
    device = X_ri.device
    freqs = torch.linspace(0, sr / 2, F, device=device)
    G = torch.zeros((B, F, T), device=device)

    for t in range(T):
        if not voiced[t]:
            continue

        f0 = f0_track[t]
        conf = confidence[t]

        if not torch.isfinite(f0) or conf < 0.3:
            continue
        
        for k in range(2, max_harmonics + 1):
            fk = k * f0 * (1.0 + 0.9 * k)
            if fk >= sr / 2:
                break

            dist = freqs - fk
            bump = torch.exp(-(dist ** 2) / (2 * (sigma_bins * sr / n_fft) ** 2))

            amp = conf / k

            G[:, :, t] += amp * bump

    G = torch.sign(G) * (torch.abs(G) ** 1.5)

    real = X_ri[:, 0]
    imag = X_ri[:, 1]
    phase = torch.rand_like(G) * 2 * torch.pi

    G_real = G * torch.cos(phase)
    G_imag = G * torch.sin(phase)

    G_ri = torch.stack((G_real, G_imag), dim=1)
    return X_ri + G_ri