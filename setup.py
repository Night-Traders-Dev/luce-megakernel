import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def _normalize_arch(value: str) -> str:
    cleaned = value.strip().lower().replace("sm_", "").replace("compute_", "")
    cleaned = cleaned.replace("+ptx", "").replace(".", "")
    return cleaned


def _cuda_arch_flags() -> list[str]:
    override = _normalize_arch(os.getenv("ELFWEAVE_MEGAKERNEL_CUDA_ARCH", ""))
    if not override:
        arch_list = os.getenv("TORCH_CUDA_ARCH_LIST", "").strip()
        if arch_list:
            override = _normalize_arch(arch_list.split(";")[0].split()[0])
    if not override and torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        override = f"{major}{minor}"
    if not override:
        override = "86"

    compute = f"compute_{override}"
    sm = f"sm_{override}"
    return [
        "-gencode",
        f"arch={compute},code={sm}",
        "-gencode",
        f"arch={compute},code={compute}",
    ]

setup(
    name="qwen35_megakernel_bf16",
    ext_modules=[
        CUDAExtension(
            name="qwen35_megakernel_bf16_C",
            sources=[
                "torch_bindings.cpp",
                "kernel.cu",
                "prefill.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-std=c++17",
                    "-DNUM_BLOCKS=82",
                    "-DBLOCK_SIZE=512",
                    "-DLM_NUM_BLOCKS=512",
                    "-DLM_BLOCK_SIZE=256",
                ] + _cuda_arch_flags(),
            },
            libraries=["cublas"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
