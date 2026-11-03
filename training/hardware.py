"""Conservative hardware discovery and named training profiles."""

from __future__ import annotations

import os
import platform
import subprocess
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class HardwareInfo:
    os: str
    cpu: str
    logical_processors: int
    ram_gb: float | None
    cuda_available: bool
    gpu: str | None
    gpu_memory_gb: float | None
    compute_capability: str | None
    torch_version: str
    cuda_version: str | None

    def as_dict(self) -> dict:
        return asdict(self)


def detect_hardware() -> HardwareInfo:
    ram_gb = None
    try:
        import psutil  # optional; never required to run training

        ram_gb = round(psutil.virtual_memory().total / (1024**3), 2)
    except Exception:
        pass
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"], capture_output=True, text=True,
                timeout=2, check=False,
            )
            lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
            if len(lines) > 1:
                cpu = lines[1]
        except Exception:
            pass
    available = bool(torch.cuda.is_available())
    gpu = memory = capability = None
    if available:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = props.name
        memory = round(props.total_memory / (1024**3), 2)
        capability = f"{props.major}.{props.minor}"
    return HardwareInfo(
        os=f"{platform.system()} {platform.release()}", cpu=cpu,
        logical_processors=os.cpu_count() or 1, ram_gb=ram_gb,
        cuda_available=available, gpu=gpu, gpu_memory_gb=memory,
        compute_capability=capability, torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
    )


def resolve_profile(profile: str, requested_device: str | None = None) -> tuple[str, HardwareInfo]:
    """Return concrete profile and safe device without changing dependencies."""
    if profile not in {"personal", "organization", "auto"}:
        raise ValueError(f"Unknown hardware profile: {profile}")
    info = detect_hardware()
    concrete = "personal" if profile == "auto" else profile
    if requested_device:
        if requested_device.startswith("cuda") and not info.cuda_available:
            raise ValueError("CUDA was requested but this PyTorch installation has no available CUDA device.")
        device = requested_device
    else:
        # Personal preserves the existing agent default. Organization and auto
        # use CUDA when present and otherwise retain a valid CPU fallback.
        device = "cuda" if info.cuda_available else "cpu"
    return f"{concrete}:{device}", info
