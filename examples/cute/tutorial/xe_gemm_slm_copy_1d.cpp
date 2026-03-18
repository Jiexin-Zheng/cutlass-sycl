/***************************************************************************************************
* Copyright (C) 2025 Intel Corporation, All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* 3. Neither the name of the copyright holder nor the names of its
* contributors may be used to endorse or promote products derived from
* this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
**************************************************************************************************/

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "../../common/sycl_cute_common.hpp"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

template <class ATensor, class BTensor, class CTensor,
          class TiledMMA>
void
gemm_device(ATensor   const& A,         // (M,K)
            BTensor   const& B,         // (N,K)
            CTensor        & C,         // (M,N)
            TiledMMA const & mma,
            CTensor        & Aux)      // (M,N)
{
  // -----
  // Setup
  // -----

  /* Get workgroup and local IDs */
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto wg_n = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  /* Create proxy coordinate tensors for each global tensor */
  Tensor cA = make_identity_tensor(A.shape());   // (M,K)
  Tensor cB = make_identity_tensor(B.shape());   // (N,K)
  Tensor cC = make_identity_tensor(C.shape());   // (M,N)
  Tensor cAux = make_identity_tensor(Aux.shape()); // (M,N)

  /* Split GEMM into workgroup tiles, and identify our workgroup's tile (wg_coord) */
  auto wg_tile = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  /*allocate shared local memory*/
  using SlmT = typename ATensor::element_type;
  auto smem = compat::local_mem<SlmT[size(select<0,1>(wg_tile))]>();
  auto BLK_M = size<0>(wg_tile);
  auto BLK_N = size<1>(wg_tile);
  Tensor STensor = make_tensor(make_smem_ptr(smem), make_layout(make_shape(BLK_M, BLK_N), make_stride(Int<BLK_N>{}, _1{})));  // row major tensor
  Tensor SInTensor = make_tensor(make_smem_ptr(smem), make_layout(make_shape(BLK_N, BLK_M), make_stride(_1{}, Int<BLK_N>{})));  // column major tensor

  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m,_));  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n,_));  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(cC, wg_tile, wg_coord, Step<_1,_1, X>{});       // (BLK_M,BLK_N)
  Tensor gAux = local_tile(cAux, wg_tile, wg_coord, Step<_1,_1, X>{}); // (BLK_M,BLK_N)
  /* Create block 2D TiledCopies */
  auto copy_a = make_block_2d_copy_A(mma, A);
  auto copy_Y = make_block_2d_copy_A(mma, make_tensor(A.data(), make_layout(shape(A))));
  auto copy_b = make_block_2d_copy_B(mma, B);
  auto copy_c = make_block_2d_copy_D(mma, C);
  auto copy_aux = make_block_2d_copy_D(mma, Aux);

  auto copy_X = make_block_2d_copy_D(mma, make_tensor(make_gmem_ptr(static_cast<SlmT*>(nullptr)), C.layout()));
  // SLM store 
  using StoreAtom = Copy_Atom<XE_1D_STSM<SlmT>, SlmT>;
  using Tiler_MN = typename decltype(copy_X)::Tiler_MN;
  using TVLayout = typename decltype(copy_X)::TiledLayout_TV;
  auto slm_store = TiledCopy<StoreAtom, TVLayout, Tiler_MN>{};
  auto thr_slm_store = slm_store.get_slice(local_id);

  using LoadAtom = Copy_Atom<XE_1D_LDSM<SlmT>, SlmT>;
  using TVLayoutLoad = typename decltype(copy_Y)::TiledLayout_TV;
  using Tiler_MN_load = typename decltype(copy_Y)::Tiler_MN;
  auto slm_load = TiledCopy<LoadAtom, TVLayoutLoad, Tiler_MN_load>{};
  auto thr_slm_load = slm_load.get_slice(local_id);

  /* Slice TiledCopy/TiledMMA operations to thread (work-item) level */
  auto thr_mma    =    mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_X = copy_X.get_slice(local_id);
  auto thr_copy_Y = copy_Y.get_slice(local_id);
  /* Register fragments for MMA */
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
  auto tCrB = thr_mma.partition_sg_fragment_B(gB(_,_,0));

  /* Register fragments for copies */
  auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
  auto tArAX = thr_copy_Y.partition_sg_fragment_D(gA(_,_,0));
  auto tBrB = thr_copy_b.partition_sg_fragment_D(gB(_,_,0));

  /* Partition global tensor (proxies) for copies */
  Tensor tAgA = thr_copy_a.partition_S(gA);
  Tensor tBgB = thr_copy_b.partition_S(gB);

  /* Partition C */
  auto tCrC = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile)));
  auto r16 = thr_copy_X.partition_sg_fragment_S(gC);
  auto tCrAux = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile)));
  Tensor tCgC = thr_mma.partition_C(gC);    /* also matches copy_c's source layout */
  Tensor tAgAux = thr_mma.partition_C(gAux);

  Tensor tOrO = thr_slm_store.retile_S(r16);
  Tensor tOsO = thr_slm_store.partition_D(STensor);
  Tensor tIrI = thr_slm_load.retile_D(tArAX);
  Tensor tIsI = thr_slm_load.partition_S(SInTensor);

  /* Create prefetch TiledCopy instances */
  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);

  auto thr_prefetch_A = prefetch_a.get_slice(local_id);
  auto thr_prefetch_B = prefetch_b.get_slice(local_id);

  /* Partition global tensor (proxies) for prefetch */
  auto pAgA = thr_prefetch_A.partition_S(gA);
  auto pBgB = thr_prefetch_B.partition_S(gB);

  /* Prefetch distance, in units of k tiles */
  const int prefetch_dist = 3;

  // ------
  // Kernel
  // ------

  constexpr int barrier_scope = 2;

  int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  /* Clear the accumulators */
  clear(tCrC);

  /* Warm up loops with prefetch to L1 */
  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
  }

  /* Main loop */
  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    /* Split barrier keeping threads loosely together */
    barrier_arrive(barrier_scope);

    /* Copy A/B from global memory (ideally L1 cache) to registers */
    copy(copy_a, tAgA(_,_,_,k_tile), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);

    /* Prefetch A/B tiles to L1 */
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));

    /* Shuffle data from copy fragments to MMA fragments */
    reorder(tArA, tCrA);
    reorder(tBrB, tCrB);

    /* Accumulate C += A * B */
    gemm(mma, tCrA, tCrB, tCrC);

    /* Other half of split barrier */
    barrier_wait(barrier_scope);
  }

  copy(copy_aux, tCrC, tAgAux);
  reorder(tCrC, r16);
  copy(slm_store, tOrO, tOsO);
  barrier_arrive(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
  barrier_wait(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

  clear(tCrC);
  // Second GEMM: K dimension is BLK_M (columns of transposed SLM tile), not original K
  int k_tile_count_slm = ceil_div(int(BLK_M), int(get<2>(wg_tile)));
  for(int k_tile = 0; k_tile < k_tile_count_slm; k_tile++) {
    copy(slm_load, tIsI(_,_,k_tile), tIrI(_,_,0));
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);
    barrier_arrive(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);
    reorder(tArAX, tCrA);
    reorder(tBrB, tCrB);
    gemm(mma, tCrA, tCrB, tCrC);
  }

  /* Write C to global memory */
  copy(copy_c, tCrC, tCgC);
}

template <typename TA, typename TB, typename TC>
auto
choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else  /* Use f16 by default as upconversion sequences are typically faster */
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto
choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&)
{
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA,TB,TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k = a_t                                  // Use one DPAS in k dimension for A^T case
                                  || (byte && b_n);                       //  pending compiler improvements (also int8 B^N).
  constexpr bool use_4x8_sg = ((sizeof_bits_v<TB> < sizeof_bits_v<TA>)    // Use smaller B loads for expensive reorders.
                                  && !(is_same_v<TB, cute::float_e5m2_t>))
                           || (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k,
                           C<op.K>, C<op.K*2>>;

  using WGTile = Shape<_256, _256, _K>;                               // 256x256 WG tile size
  using SGLayout8x4 = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;  // 8x4 SG tiling, n-major
  using SGLayout4x8 = Layout<Shape<_4, _8, _1>, Stride<_8, _1, _0>>;  // 4x8 SG tiling, n-major
  using SGLayout = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;

  return MMA{};
}

template <class, class, char, char> class GemmCuteName;
template <class ATensor, class BTensor, class CTensor, typename TA, typename TB, char layoutA, char layoutB>
void
gemm_cute(sycl::queue &Q,
          ATensor   const& A,         // (M,K)
          BTensor   const& B,         // (N,K)
          CTensor        & C,
          CTensor        & Aux)      // (M,N)
{
  auto mma = choose_tiled_mma(A, B, C);

  sycl::range<2> local = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  auto event = Q.parallel_for<GemmCuteName<TA, TB, layoutA, layoutB>>(sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      gemm_device(A, B, C, mma, Aux);
    }
  );

  EventManager::getInstance().addEvent(event);
}

template <class...> class GemmVerifyKernelName;
template <class ATensor, class BTensor, class CTensor>
bool
gemm_verify(sycl::queue &Q,
            ATensor const& A,         // (M,K)
            BTensor const& B,         // (N,K)
            CTensor const& C)         // (M,N)
{
  int m = size<0>(A);
  int n = size<0>(B);
  int k = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  Q.parallel_for<GemmVerifyKernelName<ATensor, BTensor, CTensor>>(sycl::range<2>(m, n), [=](sycl::item<2> id) {
    int i = id[0], j = id[1];

    using AccType = typename CTensor::element_type;
    using SignedAccType = ensure_signed_t<AccType>;

    auto c = AccType(0);
    for (int h = 0; h < k; h++)
      c += AccType(A(h,i)) * AccType(B(j,h));

    auto tol = AccType(2e-2f);
    if (std::abs(SignedAccType(c - AccType(C(i,j)))) > tol) {
      #define SHOW_DIFF
#ifdef SHOW_DIFF
      printf("Error at (%d,%d): got %f, expected %f\n", i, j, double(C(i,j)), double(c));
#endif
      *ok = false;
    }
  }).wait();

  bool read_ok = *ok;

  sycl::free(ok, Q);

  return read_ok;
}

template <typename TA, typename TB, typename TC,
          char layoutA = 'R', char layoutB = 'R'>
void
test_case(sycl::queue &Q, int m, int n, int k)
{
  std::cout << type_str<TA>() << " (" << layoutA << ") x "
            << type_str<TB>() << " (" << layoutB << ") -> "
            << type_str<TC>() << ": \t";

  // Transpose B to match CuTe conventions
  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');

  // Prepare data:
  auto A = make_shared_usm_tensor<TA,  layoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto C = make_shared_usm_tensor<TC,      'R'>(Q, m, n);
  auto Aux = make_shared_usm_tensor<TC,      'R'>(Q, m, n);

  random_fill(A);
  random_fill(B);
  zero_fill(C);
  zero_fill(Aux);

#ifndef SKIP_VERIFY
  auto A_ref = make_shared_usm_tensor<float,  layoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);

  copy(A, A_ref);
  copy(B, B_ref);
#endif

  subbyte_pack(A);
  subbyte_pack(B);

  // Test accuracy:
  gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, layoutA, layoutB>(Q, A, B, C, Aux);
  Q.wait_and_throw();

#ifdef SKIP_VERIFY
  const bool ok = true;
  std::cout << "verification skipped";
#else
  auto Aux_ref = make_shared_usm_tensor<float, 'R'>(Q, m, n);
  
  // The SLM path truncates accumulators (TC) to TA before the second GEMM,
  // so the reference must apply the same truncation for correct comparison.
  for (int i = 0; i < size(Aux); i++)
    Aux_ref(i) = float(TA(Aux(i)));

  // Per-tile verification: each workgroup independently computes
  //   C_tile = transpose(SlmT(Acc_tile)) @ B[:, :BLK]^T
  // where Acc_tile is the first GEMM result stored in Aux.
  // The transpose through SLM means local_r (= I % BLK) becomes the
  // column index into the Aux tile, and h2 sweeps over its rows.
  constexpr int BLK = 256;  // must match WGTile M/N dimension
  int inner_k = std::min(BLK, k);

  auto ok_ptr = sycl::malloc_shared<bool>(1, Q);
  *ok_ptr = true;

  Q.parallel_for(sycl::range<2>(m, n), [=](sycl::item<2> id) {
    int I = id[0], J = id[1];
    int wg_m = I / BLK;
    int local_r = I % BLK;
    int wg_n = J / BLK;

    using AccType = TC;
    using SignedAccType = ensure_signed_t<AccType>;

    auto c = AccType(0);
    for (int h2 = 0; h2 < inner_k; h2++) {
      c += AccType(Aux_ref(wg_m * BLK + h2, wg_n * BLK + local_r)) * AccType(B_ref(J, h2));
    }

    auto tol = AccType(2e-2f);
    if (std::abs(SignedAccType(c - AccType(C(I, J)))) > tol) {
#ifdef SHOW_DIFF
      printf("Error at (%d,%d): got %f, expected %f\n", I, J, double(C(I,J)), double(c));
#endif
      *ok_ptr = false;
    }
  }).wait();

  bool ok = *ok_ptr;
  sycl::free(ok_ptr, Q);
  std::cout << (ok ? "passed" : "failed");
#endif

  if (ok) {
    // Test performance:
    const int timing_iterations = 0;
    GPU_Clock timer;

    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      gemm_cute<decltype(A), decltype(B), decltype(C), TA, TB, layoutA, layoutB>(Q, A, B, C, Aux);
    Q.wait_and_throw();

    double avg = timer.seconds() / timing_iterations;
    double tops = (2.0*m*n*k) * 1e-12;

    printf(", %4.3f TF/s", tops / avg, avg*1000);
  }

  free_usm_tensor(A, Q);
  free_usm_tensor(B, Q);
  free_usm_tensor(C, Q);
  free_usm_tensor(Aux, Q);

#ifndef SKIP_VERIFY
  free_usm_tensor(A_ref, Q);
  free_usm_tensor(B_ref, Q);
  free_usm_tensor(Aux_ref, Q);
#endif

  std::cout << '\n';

  // Pause for a short period of time to allow the GPU to cool.
  static bool first = true;
  if (first)
    first = false;
  else
    sleep(1);
}


int main(int argc, char** argv)
{
  auto shift = [&] {
    return (argc-- > 0) ? *argv++ : nullptr;
  };

  auto parse_size = [&] {
    static constexpr int default_size = 4096;
    if (auto e = shift())
      return atoi(e);
    else
      return default_size;
  };

  (void) shift();

  auto m = parse_size();
  auto n = parse_size();
  auto k = parse_size();

  sycl::queue Q = compat::get_default_queue();

  test_case<half_t, half_t, float, 'R', 'R'>(Q, m, n, k);
  test_case<int8_t, int8_t, int32_t, 'R', 'R'>(Q, m, n, k);
}
