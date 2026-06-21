/*
 * MatMul CUDA kernel — edit this file to make it faster.
 * C = A @ B   (A: [m,k], B: [k,n], C: [m,n])
 */

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

__global__ void matmul_kernel(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m, int k, int n)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int mn = m * n;
    if (idx >= mn) return;

    const int row = idx / n;
    const int col = idx % n;

    float acc = 0.f;
    for (int t = 0; t < k; t++)
        acc += __bfloat162float(a[row * k + t]) * __bfloat162float(b[t * n + col]);

    c[row * n + col] = __float2bfloat16(acc);
}

void matmul_forward(torch::Tensor a, torch::Tensor b, torch::Tensor c)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && c.is_cuda(), "tensors must be on CUDA");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && c.is_contiguous(), "tensors must be contiguous");
    TORCH_CHECK(a.scalar_type() == torch::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(b.scalar_type() == torch::kBFloat16, "b must be bfloat16");
    TORCH_CHECK(c.scalar_type() == torch::kBFloat16, "c must be bfloat16");

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));

    const int threads = 256;
    const int blocks = (m * n + threads - 1) / threads;

    matmul_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(a.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(b.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(c.data_ptr()),
        m, k, n);

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "matmul_kernel launch failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matmul_forward", &matmul_forward);
}
