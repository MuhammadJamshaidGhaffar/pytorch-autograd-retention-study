# Pilot environment

- **generated_utc**: 2026-10-12T16:12:32.860546+00:00
- **python**: 3.12.3
- **os**: Linux-6.14.0-37-generic-x86_64-with-glibc2.39
- **pytorch**: 2.14.0+cu130
- **cuda_runtime**: 13.0
- **cuda_available**: True
- **gpu_driver_memory**: NVIDIA GeForce RTX 5090, 610.43.02, 32607 MiB
- **git**: git version 2.43.0
- **runtime_interpreter**: <usr>/bin/python3

Runtime uses system Python with CUDA PyTorch. Static baselines use the adjacent `.venv` and CodeQL CLI. The authoritative baseline versions/configurations are in [tools_manifest.csv](../results/tools_manifest.csv); full Python baseline dependencies are in [requirements.lock](../baselines/requirements.lock). Runtime package inventory: [runtime_requirements.txt](../results/runtime_requirements.txt).

Each runtime trace also has a JSON sidecar recording its seed, warm-up, iteration count, environment, and cleanup measurement. The historical September report is archived; it is not the environment for these October traces.
