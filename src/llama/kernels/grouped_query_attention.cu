#include <llama/grouped_query_attention.hpp>
#include <tensor/dtype.hpp>
#include <tensor/device_type.hpp>
#include <cuda.h>
#include <mma.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <type_traits>

namespace llama {

using namespace nvcuda;
namespace wmma = nvcuda::wmma;
using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;
namespace ptx = cuda::ptx;

__device__ __forceinline__ float safe_exp_delta(float old_m, float new_m) {
    if (!isfinite(old_m) && !isfinite(new_m)) return 0.0f;
    if (!isfinite(old_m)) return 0.0f;
    if (!isfinite(new_m)) return 0.0f; // no valid new max
    return expf(old_m - new_m);
}

template<typename T, int M, int N, int K, int NUM_THREADS, int A_STRIDE=K, int B_STRIDE=K, int C_STRIDE=N>
__forceinline__ __device__ void compute_wmma_trans(
    T (&a_smem)[M][A_STRIDE],
    T (&b_smem)[N][B_STRIDE],
    float (&c_smem)[M][C_STRIDE],
    float scale = 1.0
) {
    static_assert(M % 16 == 0);
    static_assert(N % 16 == 0);
    static_assert(K % 16 == 0);

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    int warp_id = threadIdx.x / 32;
    int num_warps = NUM_THREADS / 32;

    constexpr int NUM_M_TILES = M / WMMA_M;
    constexpr int NUM_N_TILES = N / WMMA_N;
    constexpr int NUM_OUT_TILES = NUM_M_TILES * NUM_N_TILES;

    for (int tile_id = warp_id; tile_id < NUM_OUT_TILES; tile_id += num_warps) {
        int a_tile_id = tile_id / NUM_N_TILES;
        int b_tile_id = tile_id % NUM_N_TILES;

        int a_off = a_tile_id * WMMA_M;
        int b_off = b_tile_id * WMMA_N;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        #pragma unroll
        for (int d0 = 0; d0 < K; d0 += WMMA_K) {
            wmma::load_matrix_sync(
                a_frag,
                &a_smem[a_off][d0],
                A_STRIDE
            );

            wmma::load_matrix_sync(
                b_frag,
                &b_smem[b_off][d0],
                B_STRIDE
            );

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        if (scale != 1.0) {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= scale;
            }
        }

        wmma::store_matrix_sync(
            &c_smem[a_off][b_off],
            acc_frag,
            C_STRIDE,
            wmma::mem_row_major
        );
    }

    __syncthreads();
}

template<typename T, int M, int N, int K, int NUM_THREADS, int A_STRIDE=K, int B_STRIDE=N, int C_STRIDE=N>
__forceinline__ __device__ void compute_wmma(
    T (&a_smem)[M][A_STRIDE],
    T (&b_smem)[K][B_STRIDE],
    float (&c_smem)[M][C_STRIDE],
    float scale = 1.0
) {
    static_assert(M % 16 == 0);
    static_assert(N % 16 == 0);
    static_assert(K % 8 == 0);

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    int warp_id = threadIdx.x / 32;
    int num_warps = NUM_THREADS / 32;

    constexpr int NUM_M_TILES = M / WMMA_M;
    constexpr int NUM_N_TILES = N / WMMA_N;
    constexpr int NUM_OUT_TILES = NUM_M_TILES * NUM_N_TILES;

    for (int tile_id = warp_id; tile_id < NUM_OUT_TILES; tile_id += num_warps) {
        int a_tile_id = tile_id / NUM_N_TILES;
        int b_tile_id = tile_id % NUM_N_TILES;

        int a_off = a_tile_id * WMMA_M;
        int b_off = b_tile_id * WMMA_N;

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, T, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;

        wmma::fill_fragment(acc_frag, 0.0f);

        #pragma unroll
        for (int d0 = 0; d0 < K; d0 += WMMA_K) {
            wmma::load_matrix_sync(
                a_frag,
                &a_smem[a_off][d0],
                A_STRIDE
            );

            wmma::load_matrix_sync(
                b_frag,
                &b_smem[d0][b_off],
                B_STRIDE
            );

            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        if (scale != 1.0) {
            #pragma unroll
            for (int i = 0; i < acc_frag.num_elements; ++i) {
                acc_frag.x[i] *= scale;
            }
        }

        wmma::store_matrix_sync(
            &c_smem[a_off][b_off],
            acc_frag,
            C_STRIDE,
            wmma::mem_row_major
        );
    }

    __syncthreads();
}

template<typename T, int TILE_SIZE, int HEAD_DIM, int NUM_THREADS>
struct Tile {
    T (*smem)[HEAD_DIM];
    const CUtensorMap *tmap; // for full-tile TMA loads
    const T* __restrict__ gmem_ptr;
    static constexpr int TILE_BYTES = TILE_SIZE * HEAD_DIM * sizeof(T);
    size_t b_stride;
    size_t h_stride;
    size_t s_stride;
    int seq_len;
    size_t d_stride;

    barrier* bar;
    barrier::arrival_token token;
    bool loading = false;

    void __device__ clear() {
        for (int i = threadIdx.x; i < TILE_SIZE * HEAD_DIM; i += NUM_THREADS) {
            int row = i / HEAD_DIM;
            int d   = i % HEAD_DIM;
            smem[row][d] = T(0);
        }
        __syncthreads();
    }

    void __device__ initialize() {
        assert(d_stride == 1);
        assert(s_stride == HEAD_DIM);

        if (threadIdx.x == 0) {
            init(bar, blockDim.x);
        }
    }

    void __device__ load_async(int b, int h, int s_start) {
        loading = true;
        bool full_tile = s_start + TILE_SIZE <= seq_len;
        if (!full_tile) {
            clear();
        }

        if (threadIdx.x > 0) {
            token = bar->arrive();
            return;
        }

        if (full_tile) {
            cde::cp_async_bulk_tensor_4d_global_to_shared(
                smem,
                tmap,
                0, s_start, h, b,
                *bar
            );
            token = cuda::device::barrier_arrive_tx(*bar, 1, TILE_BYTES);
        } else { // seq len is too small, memcpy_async fallback
            int valid_rows = max(0, min(TILE_SIZE, seq_len - s_start));
            size_t bytes_to_load = valid_rows * HEAD_DIM * sizeof(T);
            cuda::memcpy_async(
                smem,
                gmem_ptr + (b * b_stride) + (h * h_stride) + (s_start * s_stride),
                cuda::aligned_size_t<16>(bytes_to_load),
                *bar
            );
            token = bar->arrive();
        }
    }

    void __device__ wait() {
        if (loading)  {
            bar->wait(std::move(token));
            __syncthreads();
        }
        loading = false;
    }
};

template<typename T, int BLOCK_Q, int BLOCK_K, int HEAD_DIM, int NUM_THREADS>
__global__ void gqa_fused(
    T* __restrict__ out, size_t o_b_stride, size_t o_h_stride, size_t o_s_stride, size_t o_d_stride,
    const T* __restrict__ qs, size_t q_b_stride, size_t q_h_stride, size_t q_s_stride, size_t q_d_stride, int seq_len, int num_q_heads,
    const T* __restrict__ ks, size_t k_b_stride, size_t k_h_stride, size_t k_s_stride, size_t k_d_stride,
    const T* __restrict__ vs, size_t v_b_stride, size_t v_h_stride, size_t v_s_stride, size_t v_d_stride, int kv_seq_len, int num_kv_heads,
    // tensor maps
    const __grid_constant__ CUtensorMap q_tmap,
    const __grid_constant__ CUtensorMap k_tmap,
    const __grid_constant__ CUtensorMap v_tmap,
    const __grid_constant__ CUtensorMap o_tmap,
    float scale_factor, int group_size, int d_out, int q_pos_offset, bool is_causal) {

    int q_block = blockIdx.x;
    int bh = blockIdx.y;

    int b  = bh / num_q_heads;
    int hq = bh % num_q_heads;

    int hk = hq / group_size;

    int q_start = q_block * BLOCK_Q; // which q token should i start on?

    T* o_base = out + b * o_b_stride + hq * o_h_stride;

    assert(q_s_stride == HEAD_DIM && "q s stride is not head dim");
    assert(k_s_stride == HEAD_DIM && "k s stride is not head dim");
    assert(v_s_stride == HEAD_DIM && "v s stride is not head dim");

    int tid = threadIdx.x;

    __shared__ alignas(128) T qs_tile[BLOCK_Q][HEAD_DIM];
    __shared__ alignas(128) T ks_tile[BLOCK_K][HEAD_DIM];
    __shared__ alignas(128) T vs_tile[BLOCK_K][HEAD_DIM];

    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier q_bar;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier k_bar;
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier v_bar;

    Tile<T, BLOCK_Q, HEAD_DIM, NUM_THREADS> qtile{
        &qs_tile[0],
        &q_tmap,
        qs,
        q_b_stride, q_h_stride, q_s_stride, seq_len, q_d_stride,
        &q_bar
    };

    Tile<T, BLOCK_K, HEAD_DIM, NUM_THREADS> ktile{
        &ks_tile[0],
        &k_tmap,
        ks,
        k_b_stride, k_h_stride, k_s_stride, kv_seq_len, k_d_stride,
        &k_bar
    };

    Tile<T, BLOCK_K, HEAD_DIM, NUM_THREADS> vtile{
        &vs_tile[0],
        &v_tmap,
        vs,
        v_b_stride, v_h_stride, v_s_stride, kv_seq_len, v_d_stride,
        &v_bar
    };

    qtile.initialize();
    ktile.initialize();
    vtile.initialize();
    __syncthreads();

    //constexpr int BLOCK_K_PAD = BLOCK_K + 1;
    constexpr int BLOCK_K_PAD = BLOCK_K + 4;
    constexpr int HEAD_DIM_PAD = HEAD_DIM + 4;

    __shared__ float scores_fp32[BLOCK_Q][BLOCK_K_PAD]; // [q, k]
    __shared__ T scores_bf16[BLOCK_Q][BLOCK_K_PAD]; // [q, k]

    // online softmax stuff
    __shared__ float m[BLOCK_Q]; // running max
    __shared__ float l[BLOCK_Q]; // running denominator / sum exp
    __shared__ float row_max[BLOCK_Q];
    __shared__ float row_sum[BLOCK_Q];
    __shared__ float o_accum[BLOCK_Q][HEAD_DIM_PAD]; // running output accumulator

    qtile.load_async(b, hq, q_start);

    // initialize online softmax counters
    for (int i = tid; i < BLOCK_Q; i += NUM_THREADS) {
        m[i] = -INFINITY;
        l[i] = 0.0f;
    }

    for (int i = tid; i < BLOCK_Q * HEAD_DIM; i += NUM_THREADS) {
        int q = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        o_accum[q][d] = 0.0f;
    }
    __syncthreads();

    for(int k_idx = 0; k_idx < kv_seq_len; k_idx += BLOCK_K) {
        ktile.load_async(b, hk, k_idx);
        vtile.load_async(b, hk, k_idx);

        qtile.wait();
        ktile.wait();

        compute_wmma_trans<T, BLOCK_Q, BLOCK_K, HEAD_DIM, NUM_THREADS,
            HEAD_DIM, // a stride
            HEAD_DIM, // b stride
            BLOCK_K_PAD // c stride
            >(
            qs_tile,
            ks_tile,
            scores_fp32,
            scale_factor
        );

        // apply attn mask
        #pragma unroll
        for (int i = tid; i < BLOCK_Q * BLOCK_K; i += NUM_THREADS) {
            int q = i / BLOCK_K;
            int k = i % BLOCK_K;

            int q_local = q_start + q;
            int q_abs = q_pos_offset + q_local;
            int k_abs = k_idx + k;

            if (q_local >= seq_len || k_abs >= kv_seq_len) {
                scores_fp32[q][k] = -INFINITY;
            } else if (is_causal && k_abs > q_abs) {
                scores_fp32[q][k] = -INFINITY;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int q = tid; q < BLOCK_Q; q += NUM_THREADS) {
            float rmax = -INFINITY;

            #pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                rmax = fmaxf(rmax, scores_fp32[q][k]);
            }
            row_max[q] = fmaxf(m[q], rmax);
        }
        __syncthreads();


        // scores = exp(scores - row_max), row_sum
        #pragma unroll
        for (int q = tid; q < BLOCK_Q; q += NUM_THREADS) {
            float sum = 0.0f;
            float m_new = row_max[q];

            #pragma unroll
            for (int k = 0; k < BLOCK_K; ++k) {
                float p = 0.0f;

                if (isfinite(m_new)) {
                    p = expf(scores_fp32[q][k] - m_new);
                }
                scores_fp32[q][k] = p;
                sum += p;
            }
            row_sum[q] = sum;
        }
        __syncthreads();

        // Rescale old O accumulator by alpha.
        #pragma unroll
        for (int i = tid; i < BLOCK_Q * HEAD_DIM; i += NUM_THREADS) {
            int q = i / HEAD_DIM;
            int d = i % HEAD_DIM;
            float alpha = safe_exp_delta(m[q], row_max[q]);
            o_accum[q][d] *= alpha;
        }
        __syncthreads();

        #pragma unroll
        for (int i = tid; i < BLOCK_Q * BLOCK_K; i += NUM_THREADS) {
            int q = i / BLOCK_K;
            int k = i % BLOCK_K;
            scores_bf16[q][k] = T(scores_fp32[q][k]);
        }
        __syncthreads();

        vtile.wait();

        // accumulate O
        // o_accum += p @ V
        // scores is BLOCK_Q x BLOCK_K
        // vtile is BLOCK_K x HEAD_DIM
        compute_wmma<T, BLOCK_Q, HEAD_DIM, BLOCK_K, NUM_THREADS,
            BLOCK_K_PAD, // a stride
            HEAD_DIM, // b stride
            HEAD_DIM_PAD // c stride
            >(
            scores_bf16,
            vs_tile,
            o_accum
        );

        // Update l and m.
        for (int q = tid; q < BLOCK_Q; q += NUM_THREADS) {
            float alpha = safe_exp_delta(m[q], row_max[q]);
            l[q] = l[q] * alpha + row_sum[q];
            m[q] = row_max[q];
        }
        __syncthreads();
    }

    // Final store.
    for (int i = tid; i < BLOCK_Q * HEAD_DIM; i += NUM_THREADS) {
        int q = i / HEAD_DIM;
        int d = i % HEAD_DIM;
        int q_local = q_start + q;

        if (q_local < seq_len && d < d_out) {
            float denom = l[q];
            float val = denom > 0.0f && isfinite(denom) ? o_accum[q][d] / denom : 0.0f;
            o_base[q_local * o_s_stride + d * o_d_stride] = static_cast<T>(val);
        }
    }

    if (tid == 0) {
        (&q_bar)->~barrier();
        (&k_bar)->~barrier();
        (&v_bar)->~barrier();
    }
}

// Map CUtensorMapDataType from C++ type
template<typename T>
constexpr CUtensorMapDataType tma_dtype() {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    } else if constexpr (std::is_same_v<T, __half>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    } else if constexpr (std::is_same_v<T, float>) {
        return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    }
}

template<typename T, int TILE_SEQ>
CUtensorMap create_tensor_map(
    const T* gmem_ptr, // pointer to tensor[batch, head, seq, dim];
    unsigned int b, unsigned int h, unsigned int s, unsigned int d,
    size_t b_stride, size_t h_stride, size_t s_stride
    // q_d,
    // batch_size, num_q_heads, seq_len, q_head_dim,
    // q_b_stride, q_h_stride, q_s_stride
) {
    CUtensorMap tensor_map;

    // global dimensions (in elements, not bytes)
    // innermost to outermost
    uint64_t global_dims[4] = {
        static_cast<uint64_t>(d),
        static_cast<uint64_t>(s),
        static_cast<uint64_t>(h),
        static_cast<uint64_t>(b)
    };

    // global strides (in bytes, only for dims > 0)
    uint64_t global_strides[3] = {
        static_cast<uint64_t>(s_stride * sizeof(T)),
        static_cast<uint64_t>(h_stride * sizeof(T)),
        static_cast<uint64_t>(b_stride * sizeof(T))
    };

    // box dimensions: the tile size to load
    uint32_t box_dims[4] = {
        static_cast<uint32_t>(d),
        static_cast<uint32_t>(TILE_SEQ),
        1, 1
    };

    uint32_t elem_strides[4] = {1,1,1,1};

    CUresult result = cuTensorMapEncodeTiled(
        &tensor_map,
        tma_dtype<T>(),
        4, // tensor rank
        const_cast<void*>(static_cast<const void*>(gmem_ptr)),
        global_dims,
        global_strides,
        box_dims,
        elem_strides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, // no swizzle (simpler for mma/WMMA)
        CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
    );

    assert(result == CUDA_SUCCESS);
    return tensor_map;
}

template <typename T, typename D, int HEAD_DIM>
Tensor<std::remove_const_t<T>, D> gqa_forward_fused(const TensorView<T, D> &qs,
                                                    const TensorView<T, D> &ks,
                                                    const TensorView<T, D> &vs,
                                                    float scale_factor,
                                                    size_t group_size,
                                                    size_t d_out,
                                                    int position_offset,
                                                    bool is_causal) {
  Shape shape = qs.shape;
  unsigned int batch_size = shape[0];

  unsigned int num_q_heads = shape[1];
  unsigned int seq_len = shape[2];

  unsigned int num_kv_heads = ks.shape[1];
  unsigned int kv_seq_len = ks.shape[2];

  unsigned int q_head_dim = shape[3];
  unsigned int kv_head_dim = ks.shape[3];

  assert(q_head_dim == HEAD_DIM && "head dim is hardcoded");

  size_t q_b_stride = qs.stride[0]; size_t q_h_stride = qs.stride[1];
  size_t q_s_stride = qs.stride[2]; size_t q_d_stride = qs.stride[3];

  size_t k_b_stride = ks.stride[0]; size_t k_h_stride = ks.stride[1];
  size_t k_s_stride = ks.stride[2]; size_t k_d_stride = ks.stride[3];

  size_t v_b_stride = vs.stride[0]; size_t v_h_stride = vs.stride[1];
  size_t v_s_stride = vs.stride[2]; size_t v_d_stride = vs.stride[3];

  size_t o_b_stride = seq_len * num_q_heads * q_head_dim;
  size_t o_h_stride = q_head_dim;
  size_t o_s_stride = num_q_heads * q_head_dim;
  size_t o_d_stride = 1;

  auto n_elements = static_cast<size_t>(batch_size) * seq_len * num_q_heads * q_head_dim;

  Shape out_shape{batch_size, seq_len, q_head_dim * num_q_heads};

  TensorStorage<T, CUDA> storage(n_elements);
  Tensor<T, CUDA> out{out_shape, std::move(storage)};

  auto* out_d = reinterpret_cast<Cuda<T>*>(out.data()); // NOLINT
  const auto* q_d = reinterpret_cast<Cuda<T>*>(qs.data); // NOLINT
  const auto* k_d = reinterpret_cast<Cuda<T>*>(ks.data); // NOLINT
  const auto* v_d = reinterpret_cast<Cuda<T>*>(vs.data); // NOLINT

  constexpr int BLOCK_Q = 16; // queries per CTA
  constexpr int BLOCK_K = 16; // keys streamed per iteration
  constexpr int THREADS = 128; // 4 warps

  auto q_tmap = create_tensor_map<Cuda<T>, BLOCK_Q>(
      q_d,
      batch_size, num_q_heads, seq_len, q_head_dim,
      q_b_stride, q_h_stride, q_s_stride
  );
  auto k_tmap = create_tensor_map<Cuda<T>, BLOCK_K>(
      k_d,
      batch_size, num_kv_heads, kv_seq_len, kv_head_dim,
      k_b_stride, k_h_stride, k_s_stride
  );
  auto v_tmap = create_tensor_map<Cuda<T>, BLOCK_K>(
      v_d,
      batch_size, num_kv_heads, kv_seq_len, kv_head_dim,
      v_b_stride, v_h_stride, v_s_stride
  );
  auto o_tmap = create_tensor_map<Cuda<T>, BLOCK_Q>(
      out_d,
      batch_size, num_q_heads, seq_len, q_head_dim,
      o_b_stride, o_h_stride, o_s_stride
  );

  dim3 grid_size{
      static_cast<unsigned int>((seq_len + BLOCK_Q-1) / BLOCK_Q),
      batch_size*num_q_heads
  };

  gqa_fused<Cuda<T>, BLOCK_Q, BLOCK_K, HEAD_DIM, THREADS><<<grid_size, THREADS>>>(
      out_d, o_b_stride, o_h_stride, o_s_stride, o_d_stride,
      q_d, q_b_stride, q_h_stride, q_s_stride, q_d_stride, seq_len, num_q_heads,
      k_d, k_b_stride, k_h_stride, k_s_stride, k_d_stride,
      v_d, v_b_stride, v_h_stride, v_s_stride, v_d_stride, kv_seq_len, num_kv_heads,
      q_tmap, k_tmap, v_tmap, o_tmap,
      scale_factor, group_size, d_out, position_offset, is_causal
      );

  // cudaError_t err = cudaGetLastError();
  // if (err != cudaSuccess) {
  //   fmt::print(stderr, "GQA launch error: {}\n", cudaGetErrorString(err));
  // }

  // err = cudaDeviceSynchronize();
  // if (err != cudaSuccess) {
  //     fmt::print(stderr, "GQA kernel error: {}\n", cudaGetErrorString(err));
  // }

  return out;
}

template
Tensor<bfloat16, CUDA>
gqa_forward_fused<bfloat16, CUDA, 64>(const TensorView<bfloat16, CUDA>& qs,
                  const TensorView<bfloat16, CUDA>& ks,
                  const TensorView<bfloat16, CUDA>& vs, float scale_factor,
                  size_t group_size, size_t d_out, int position_offset, bool is_causal);

}
