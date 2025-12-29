from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="spherical_voronoi",
    version="0.0.1",
    packages=["spherical_voronoi"],
    ext_modules=[
        CUDAExtension(
            name="spherical_voronoi._C",
            sources=[
                "bindings.cpp",
                "spherical_voronoi_cuda.cu",
            ],
            extra_compile_args={
                "cxx": [],
                "nvcc": [],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
