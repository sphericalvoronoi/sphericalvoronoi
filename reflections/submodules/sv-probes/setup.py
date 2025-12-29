from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="sv_probes",
    ext_modules=[
        CUDAExtension(
            name="sv_probes._C",
            sources=[
                "binding.cpp",
                "sv_probes.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
