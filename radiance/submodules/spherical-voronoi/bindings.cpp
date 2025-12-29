#include <torch/extension.h>

torch::Tensor spherical_voronoi_forward(
    torch::Tensor sites,      // [K,8,3]
    torch::Tensor directions, // [K,3]
    torch::Tensor tau,        // [K,8]
    torch::Tensor color       // [K,8,3]
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def(
        "forward",
        &spherical_voronoi_forward,
        "Spherical Voronoi forward (CUDA, exact L2 + stable softmax)");
}
