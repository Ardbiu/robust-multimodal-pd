import torch


def get_torch_device(prefer_mps: bool = True) -> torch.device:
    """
    Select a torch device. Prefers CUDA, then MPS (Apple Silicon), then CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
