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

static inline __device__ float clamp_eps(float x, float eps)
{
    return x < eps ? eps : x;
}

template <typename scalar_t>
__global__ void spherical_voronoi_fwd_kernel(
    const scalar_t *__restrict__ sites, // [K,8,3] flatten
    const scalar_t *__restrict__ dirs,  // [K,3]   flatten
    const scalar_t *__restrict__ tau,   // [K,8]   flatten
    const scalar_t *__restrict__ color, // [K,8,3] flatten
    scalar_t *__restrict__ out,         // [K,3]   flatten
    int64_t K,
    float eps)
{
    int64_t k = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K)
        return;

    const int64_t d_base = k * 3;
    float d0 = (float)dirs[d_base + 0];
    float d1 = (float)dirs[d_base + 1];
    float d2 = (float)dirs[d_base + 2];

    float logits[8];
    
    #pragma unroll
    for (int s = 0; s < 8; ++s)
    {
        // sites[k,s,:] offset: k*(8*3) + s*3 + xyz
        const int64_t s_base = k * (8 * 3) + s * 3;
        float s0 = (float)sites[s_base + 0];
        float s1 = (float)sites[s_base + 1];
        float s2 = (float)sites[s_base + 2];

        float dx = s0 - d0;
        float dy = s1 - d1;
        float dz = s2 - d2;
        float dist2 = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(clamp_eps(dist2, eps));

        float t = (float)tau[k * 8 + s];
        logits[s] = -t * dist;
    }

    float m = logits[0];
    #pragma unroll
    for (int s = 1; s < 8; ++s) m = fmaxf(m, logits[s]);

    float exps[8];
    float denom = 0.0f;
    
    #pragma unroll
    for (int s = 0; s < 8; ++s)
    {
        float e = expf(logits[s] - m); 
        exps[s] = e;
        denom += e;
    }
    float inv_denom = 1.0f / denom;

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f;

    #pragma unroll
    for (int s = 0; s < 8; ++s)
    {
        float w = exps[s] * inv_denom;

        // color[k,s,:] offset: k*(8*3) + s*3 + rgb
        const int64_t c_base = k * (8 * 3) + s * 3;
        float c0 = (float)color[c_base + 0];
        float c1 = (float)color[c_base + 1];
        float c2 = (float)color[c_base + 2];

        o0 += w * c0;
        o1 += w * c1;
        o2 += w * c2;
    }

    const int64_t o_base = k * 3;
    out[o_base + 0] = (scalar_t)o0;
    out[o_base + 1] = (scalar_t)o1;
    out[o_base + 2] = (scalar_t)o2;
}


torch::Tensor spherical_voronoi_forward(
    torch::Tensor sites,      // [K,8,3]
    torch::Tensor directions, // [K,3]
    torch::Tensor tau,        // [K,8]
    torch::Tensor color       // [K,8,3]
)
{
    CHECK_CUDA(sites);
    CHECK_CUDA(directions);
    CHECK_CUDA(tau);
    CHECK_CUDA(color);

    CHECK_CONTIGUOUS(sites);
    CHECK_CONTIGUOUS(directions);
    CHECK_CONTIGUOUS(tau);
    CHECK_CONTIGUOUS(color);

    CHECK_DTYPE_FLOAT_OR_HALF(sites);
    TORCH_CHECK(directions.scalar_type() == sites.scalar_type(), "directions dtype must match sites dtype");
    TORCH_CHECK(tau.scalar_type() == sites.scalar_type(), "tau dtype must match sites dtype");
    TORCH_CHECK(color.scalar_type() == sites.scalar_type(), "color dtype must match sites dtype");

    TORCH_CHECK(sites.dim() == 3 && sites.size(1) == 8 && sites.size(2) == 3, "sites must be [K,8,3]");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3, "directions must be [K,3]");
    TORCH_CHECK(tau.dim() == 2 && tau.size(1) == 8, "tau must be [K,8]");
    TORCH_CHECK(color.dim() == 3 && color.size(1) == 8 && color.size(2) == 3, "color must be [K,8,3]");

    const auto K = sites.size(0);
    TORCH_CHECK(directions.size(0) == K, "K mismatch: directions");
    TORCH_CHECK(tau.size(0) == K, "K mismatch: tau");
    TORCH_CHECK(color.size(0) == K, "K mismatch: color");

    auto out = torch::empty({K, 3}, torch::TensorOptions().device(sites.device()).dtype(sites.dtype()));


    const int threads = 256;
    const int blocks = (int)((K + threads - 1) / threads);

    const float eps = 1e-12f;

    cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sites.scalar_type(), "spherical_voronoi_fwd", [&]
                                        { spherical_voronoi_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                                              (const scalar_t *)sites.data_ptr<scalar_t>(),
                                              (const scalar_t *)directions.data_ptr<scalar_t>(),
                                              (const scalar_t *)tau.data_ptr<scalar_t>(),
                                              (const scalar_t *)color.data_ptr<scalar_t>(),
                                              (scalar_t *)out.data_ptr<scalar_t>(),
                                              K,
                                              eps); });

    return out;
}