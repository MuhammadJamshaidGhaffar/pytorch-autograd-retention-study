import torch


def get_workload(num_batches=80, device="cuda", seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(num_batches):
        yield torch.randn(32, 128, generator=generator, device=device)
