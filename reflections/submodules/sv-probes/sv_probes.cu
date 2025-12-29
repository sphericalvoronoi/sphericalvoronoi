#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_FLOAT_OR_HALF(x) TORCH_CHECK(                  \
    x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf, \
    #x " must be float32 or float16")

static inline __device__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

template <typename idx_t>
static inline __device__ int64_t load_idx(const idx_t* p) {
    return (int64_t)(*p);
}

template <typename scalar_t, typename idx_t>
__global__ void fused_specular_near_m8_kernel(
    const scalar_t* __restrict__ sites,     // [V,S,3] flattened
    const scalar_t* __restrict__ colors,    // [V,S,3] flattened
    const scalar_t* __restrict__ alpha,     // [V] flattened (pre-sigmoid)
    const idx_t*   __restrict__ idx_topk,   // [V,C,8] flattened (M=8)
    const idx_t*   __restrict__ iidx,       // [B,nn_k] flattened
    const scalar_t* __restrict__ weights,   // [B,nn_k] flattened
    const scalar_t* __restrict__ wo,        // [B,3] flattened (assumed normalized)
    const float*   __restrict__ k_level,    // [B] float32
    const idx_t*   __restrict__ cell_id,    // [B]
    scalar_t* __restrict__ out_probes,      // [B,3]
    scalar_t* __restrict__ out_alpha,       // [B]
    int64_t B,
    int64_t S,
    int64_t C,
    int64_t nn_k)
{
    int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    // load wo[b]
    const int64_t wo_base = b * 3;
    float w0 = (float)wo[wo_base + 0];
    float w1 = (float)wo[wo_base + 1];
    float w2 = (float)wo[wo_base + 2];

    float kappa = k_level[b];
    int64_t cell = load_idx(cell_id + b);

    float out0 = 0.0f, out1 = 0.0f, out2 = 0.0f;
    float outa = 0.0f;

    for (int64_t j = 0; j < nn_k; ++j) {
        int64_t v  = load_idx(iidx + b * nn_k + j);
        float   wj = (float)weights[b * nn_k + j];

        // alpha blend (sigmoid like your torch)
        float a_sig = sigmoidf((float)alpha[v]);
        outa += wj * a_sig;

        float logits[8];

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            int64_t topk_off = (v * C + cell) * 8 + m;
            int64_t sidx = load_idx(idx_topk + topk_off);

            int64_t sbase = (v * S + sidx) * 3;
            float s0 = (float)sites[sbase + 0];
            float s1 = (float)sites[sbase + 1];
            float s2 = (float)sites[sbase + 2];

            float dot = s0 * w0 + s1 * w1 + s2 * w2;   
            logits[m] = kappa * dot;
        }

        float mmax = logits[0];
        #pragma unroll
        for (int m = 1; m < 8; ++m) mmax = fmaxf(mmax, logits[m]);

        float exps[8];
        float denom = 0.0f;
        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            float e = expf(logits[m] - mmax);
            exps[m] = e;
            denom += e;
        }
        denom = fmaxf(denom, 1e-20f);
        float inv_denom = 1.0f / denom;

        float c0 = 0.0f, c1 = 0.0f, c2 = 0.0f;

        #pragma unroll
        for (int m = 0; m < 8; ++m) {
            float wm = exps[m] * inv_denom;

            int64_t topk_off = (v * C + cell) * 8 + m;
            int64_t sidx = load_idx(idx_topk + topk_off);

            int64_t cbase = (v * S + sidx) * 3;
            float cc0 = (float)colors[cbase + 0];
            float cc1 = (float)colors[cbase + 1];
            float cc2 = (float)colors[cbase + 2];

            c0 += wm * cc0;
            c1 += wm * cc1;
            c2 += wm * cc2;
        }

        // blend across probes
        out0 += wj * c0;
        out1 += wj * c1;
        out2 += wj * c2;
    }

    // write outputs
    int64_t obase = b * 3;
    out_probes[obase + 0] = (scalar_t)out0;
    out_probes[obase + 1] = (scalar_t)out1;
    out_probes[obase + 2] = (scalar_t)out2;
    out_alpha[b] = (scalar_t)outa;
}

std::vector<torch::Tensor> fused_specular_near_forward_m8(
    torch::Tensor sites,      // [V,S,3] float/half (normalized)
    torch::Tensor colors,     // [V,S,3] float/half
    torch::Tensor alpha,      // [V]     float/half (pre-sigmoid)
    torch::Tensor idx_topk,   // [V,C,8] int32/int64
    torch::Tensor iidx,       // [B,nn_k] int32/int64
    torch::Tensor weights,    // [B,nn_k] float/half
    torch::Tensor wo,         // [B,3] float/half (normalized)
    torch::Tensor k_level,    // [B] float32
    torch::Tensor cell_id     // [B] int32/int64
)
{
    CHECK_CUDA(sites); CHECK_CUDA(colors); CHECK_CUDA(alpha);
    CHECK_CUDA(idx_topk); CHECK_CUDA(iidx); CHECK_CUDA(weights);
    CHECK_CUDA(wo); CHECK_CUDA(k_level); CHECK_CUDA(cell_id);

    CHECK_CONTIGUOUS(sites); CHECK_CONTIGUOUS(colors); CHECK_CONTIGUOUS(alpha);
    CHECK_CONTIGUOUS(idx_topk); CHECK_CONTIGUOUS(iidx); CHECK_CONTIGUOUS(weights);
    CHECK_CONTIGUOUS(wo); CHECK_CONTIGUOUS(k_level); CHECK_CONTIGUOUS(cell_id);

    CHECK_DTYPE_FLOAT_OR_HALF(sites);
    TORCH_CHECK(colors.scalar_type() == sites.scalar_type(), "colors dtype must match sites");
    TORCH_CHECK(alpha.scalar_type() == sites.scalar_type(), "alpha dtype must match sites");
    TORCH_CHECK(weights.scalar_type() == sites.scalar_type(), "weights dtype must match sites");
    TORCH_CHECK(wo.scalar_type() == sites.scalar_type(), "wo dtype must match sites");

    TORCH_CHECK(k_level.scalar_type() == at::kFloat, "k_level must be float32");
    TORCH_CHECK(idx_topk.scalar_type() == at::kInt || idx_topk.scalar_type() == at::kLong, "idx_topk must be int32 or int64");
    TORCH_CHECK(iidx.scalar_type() == idx_topk.scalar_type(), "iidx dtype must match idx_topk");
    TORCH_CHECK(cell_id.scalar_type() == idx_topk.scalar_type(), "cell_id dtype must match idx_topk");

    TORCH_CHECK(sites.dim() == 3 && sites.size(2) == 3, "sites must be [V,S,3]");
    TORCH_CHECK(colors.sizes() == sites.sizes(), "colors must match sites [V,S,3]");
    TORCH_CHECK(alpha.dim() == 1 && alpha.size(0) == sites.size(0), "alpha must be [V]");
    TORCH_CHECK(idx_topk.dim() == 3 && idx_topk.size(0) == sites.size(0) && idx_topk.size(2) == 8, "idx_topk must be [V,C,8]");
    TORCH_CHECK(wo.dim() == 2 && wo.size(1) == 3, "wo must be [B,3]");
    TORCH_CHECK(k_level.dim() == 1 && k_level.size(0) == wo.size(0), "k_level must be [B]");
    TORCH_CHECK(cell_id.dim() == 1 && cell_id.size(0) == wo.size(0), "cell_id must be [B]");
    TORCH_CHECK(iidx.dim() == 2 && iidx.size(0) == wo.size(0), "iidx must be [B,nn_k]");
    TORCH_CHECK(weights.sizes() == iidx.sizes(), "weights must match iidx [B,nn_k]");

    const auto V = sites.size(0);
    const auto S = sites.size(1);
    const auto B = wo.size(0);
    const auto C = idx_topk.size(1);
    const auto nn_k = iidx.size(1);

    auto out_probes = torch::empty({B, 3}, sites.options());
    auto out_alpha  = torch::empty({B}, sites.options());

    const int threads = 256;
    const int blocks  = (int)((B + threads - 1) / threads);
    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    if (idx_topk.scalar_type() == at::kInt) {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(sites.scalar_type(), "fused_specular_near_m8_int32", [&] {
            fused_specular_near_m8_kernel<scalar_t, int32_t><<<blocks, threads, 0, stream>>>(
                (const scalar_t*)sites.data_ptr<scalar_t>(),
                (const scalar_t*)colors.data_ptr<scalar_t>(),
                (const scalar_t*)alpha.data_ptr<scalar_t>(),
                (const int32_t*)idx_topk.data_ptr<int32_t>(),
                (const int32_t*)iidx.data_ptr<int32_t>(),
                (const scalar_t*)weights.data_ptr<scalar_t>(),
                (const scalar_t*)wo.data_ptr<scalar_t>(),
                (const float*)k_level.data_ptr<float>(),
                (const int32_t*)cell_id.data_ptr<int32_t>(),
                (scalar_t*)out_probes.data_ptr<scalar_t>(),
                (scalar_t*)out_alpha.data_ptr<scalar_t>(),
                B, S, C, nn_k
            );
        });
    } else {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(sites.scalar_type(), "fused_specular_near_m8_int64", [&] {
            fused_specular_near_m8_kernel<scalar_t, int64_t><<<blocks, threads, 0, stream>>>(
                (const scalar_t*)sites.data_ptr<scalar_t>(),
                (const scalar_t*)colors.data_ptr<scalar_t>(),
                (const scalar_t*)alpha.data_ptr<scalar_t>(),
                (const int64_t*)idx_topk.data_ptr<int64_t>(),
                (const int64_t*)iidx.data_ptr<int64_t>(),
                (const scalar_t*)weights.data_ptr<scalar_t>(),
                (const scalar_t*)wo.data_ptr<scalar_t>(),
                (const float*)k_level.data_ptr<float>(),
                (const int64_t*)cell_id.data_ptr<int64_t>(),
                (scalar_t*)out_probes.data_ptr<scalar_t>(),
                (scalar_t*)out_alpha.data_ptr<scalar_t>(),
                B, S, C, nn_k
            );
        });
    }

    return {out_probes, out_alpha};
}
