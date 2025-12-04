import torch


def batched_vad(model, x, sampling_rate: int = 16000, threshold: float = 0.1):
    """Return True if mean VAD probability over tensor exceeds threshold.

    Accepts either a 1D tensor of raw samples (length N), which will be split
    into 512-sample frames, or a 2D tensor with shape [batch_size, 512].
    If N is not divisible by 512, the remainder is padded and evaluated
    in a separate model call. Returns a boolean based on mean probability.
    """

    if not torch.is_tensor(x):
        x = torch.tensor(x)

    if x.dim() == 1:
        total_samples = x.shape[0]
        full_frames = total_samples // 512
        probs_parts = []

        if full_frames:
            batch = x[: full_frames * 512].reshape(full_frames, 512)
            out = model(batch, sampling_rate)
            out_tensor = out if isinstance(out, torch.Tensor) else torch.tensor(out)
            probs_parts.append(out_tensor.view(-1))

        remainder = total_samples - (full_frames * 512)
        if remainder:
            rem = x[-remainder:]
            pad = 512 - remainder
            rem_padded = torch.nn.functional.pad(rem, (0, pad))
            out_rem = model(rem_padded.unsqueeze(0), sampling_rate)
            out_rem_tensor = (
                out_rem if isinstance(out_rem, torch.Tensor) else torch.tensor(out_rem)
            )
            probs_parts.append(out_rem_tensor.view(-1))

        if not probs_parts:
            return False
        probs = torch.cat(probs_parts).float()
        mean_prob = probs.mean().item()
        return bool(mean_prob > threshold)

    if x.dim() == 2:
        if x.shape[1] != 512:
            raise ValueError("Expected 2D input with shape [batch_size, 512]")
        out = model(x, sampling_rate)
        out_tensor = out if isinstance(out, torch.Tensor) else torch.tensor(out)
        mean_prob = out_tensor.view(-1).float().mean().item()
        return bool(mean_prob > threshold)

    raise ValueError("Input must be 1D [N] or 2D [batch_size, 512]")
