#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_specular_near_forward_m8(
    torch::Tensor sites,
    torch::Tensor colors,
    torch::Tensor alpha,
    torch::Tensor idx_topk,
    torch::Tensor iidx,
    torch::Tensor weights,
    torch::Tensor wo,
    torch::Tensor k_level,
    torch::Tensor cell_id
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "forward",
        &fused_specular_near_forward_m8,
        "Voronoi specular fused forward (near, M=8)"
    );
}
