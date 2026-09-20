import torch

def get_workload(batch_size=64, num_features=1024, num_batches=100, device="cpu"):
    """Deterministic batch generator ensuring vulnerable and fixed see identical inputs."""
    torch.manual_seed(42)  # Pinned seed for reproducible traces
    for _ in range(num_batches):
        yield torch.randn(batch_size, num_features, device=device, requires_grad=True)