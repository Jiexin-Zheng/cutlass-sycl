/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/command_line.h"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "../../common/sycl_cute_common.hpp"

// This is a simple tutorial showing several ways to partition a tensor into tiles then
// perform efficient, coalesced copies. This example also shows how to vectorize accesses
// which may be a useful optimization or required for certain workloads.
//
// The result are a part of compatible tensors with dimensions ((M, N), m', n'), where
// (M, N) denotes a statically sized tile, and m' and n' denote the number of such tiles
// within the tensor.
//
// Each statically sized tile is mapped to a CUDA threadblock which performs efficient
// loads and stores to Global Memory.
//
// `copy_kernel_vectorized()` uses `cute::make_tiled_copy()` to perform a similar
// partitioning using `cute::Copy_Atom` to perform vectorization. The actual vector
// size is defined by `ThreadShape`.
//
// This example assumes the overall tensor shape is divisible by the tile size and
// does not perform predication.


/// Vectorized copy kernel.
///
/// Uses `make_tiled_copy()` to perform a copy using vector instructions. This operation
/// has the precondition that pointers are aligned to the vector size.
///
template <class...> class CopyKernelVectorizedName;

template <class TensorS, class TensorD, class MMA>
void copy_kernel_vectorized(TensorS S, TensorD D, MMA)
{
  using namespace cute;
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto local_id = int(item.get_local_id(0));
  using Element = typename TensorS::value_type;

  // global memory load which layout is the same with MMA result
  auto smem = compat::local_mem<Element[size(S)]>();
  Tensor STensor = make_tensor(make_smem_ptr(smem), S.layout());

  auto store = make_block_2d_copy_D(MMA{}, S);
  using Tiler_MN = typename decltype(store)::Tiler_MN;
  using TVLayout = typename decltype(store)::TiledLayout_TV;

  using Op = XE_LOAD_2D<sizeof_bits_v<Element>, 8, 16>;
  using SGCopy = decltype(make_block_2d_copy<Element>(Op{}, S.stride(), find_x_mode(S.stride()), find_y_mode(S.stride())));
  using LoadAtom = typename SGCopy::Atom;
  auto load = TiledCopy<LoadAtom,TVLayout,Tiler_MN>{}.with(S);

  auto thr_load = load.get_slice(local_id);
  Tensor cS = make_identity_tensor(S.shape());   // (M,N)
  Tensor tIgI = thr_load.partition_S(cS);
  Tensor tIrI = thr_load.partition_sg_fragment_D(cS);

  // SLM store 
  using Atom = Copy_Atom<UniversalCopy<Element>, Element>;
  using SGCopySLM = decltype(make_tiled_copy(Atom{}, Layout<Shape<_1, _16>>{}, Layout<Shape<_8, _1>>{}));
  
  auto slm_store = TiledCopy<Atom, TVLayout, Tiler_MN>{};
  auto thr_slm_store = slm_store.get_slice(local_id);
  auto tOrO = thr_slm_store.retile_S(tIrI);
  auto tOsO = thr_slm_store.partition_D(STensor);


  // Copy from GMEM to RMEM and from RMEM to GMEM
  clear(tIrI);
  copy(load, tIgI, tIrI);
//   copy(slm_store, tOrO, tOsO);
  copy(store, tIrI, tIgI);
  // Barrier, with memory fence
//   barrier_arrive(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
//   barrier_wait(SPIRVScope::ScopeWorkgroup, SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);
  #define PRINT(x) print(#x ": "); print(x); print("\n");
  if(cute::thread0()) {
    PRINT(load);
    PRINT(slm_store);
    PRINT(tIgI);
    PRINT(tIrI);
    PRINT(tOrO);
    PRINT(tOsO);
    PRINT(tIrI[0])
    PRINT(tIrI[1])
    PRINT(tIrI[2])
    PRINT(tIrI[3])
    PRINT(tIrI[4])
    PRINT(tIrI[5])
    PRINT(tIrI[6])
    PRINT(tIrI[7])
    // for(int i = 0; i < 32; i++) {
    //     for(int j = 0; j< 32;j++) {
    //         print(STensor(i, j)); print(", ");
    //     }
    //     print("\n");
    // }
  }

}

/// Main function
int main(int argc, char** argv)
{
  //
  // Given a 2D shape, perform an efficient copy
  //

  using namespace cute;
  using Element = float;

  // Define a tensor shape with dynamic extents (m, n)
  auto tensor_shape = make_shape(_32{}, _32{});

  //
  // Allocate and initialize
  //
  std::vector<Element> h_S(size(tensor_shape));
  std::vector<Element> h_D(size(tensor_shape));

  auto d_S = compat::malloc<Element>(size(tensor_shape));
  auto d_D = compat::malloc<Element>(size(tensor_shape));

  for (size_t i = 0; i < h_S.size(); ++i) {
    h_S[i] = static_cast<Element>(i);
  }

  compat::memcpy<Element>(d_S, h_S.data(), size(tensor_shape));
  compat::memcpy<Element>(d_D, h_D.data(), size(tensor_shape));

  //
  // Make tensors
  //

  Tensor tensor_S = make_tensor(d_S, make_layout(tensor_shape, make_stride(_32{}, _1{})));
  Tensor tensor_D = make_tensor(d_D, make_layout(tensor_shape, make_stride(_32{}, _1{})));

  //
  // Tile tensors
  //

  // Thread arrangement
  using WGTile = Shape<_32, _32, _32>;
  using SGLayout8x4 = Layout<Shape<_4,_2,_1>, Stride<_2,_1,_0>>;
  using MMA = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, Element, half_t, half_t>>, Layout<WGTile>, SGLayout8x4>::TiledMMA;

  //
  // Determine grid and block dimensions
  //

  auto gridDim  = compat::dim3(1, 1);  // Grid shape corresponds to modes m' and n'
  auto blockDim = compat::dim3(size(MMA{}), 1);

  //
  // Launch the kernel
  //
  compat::launch<copy_kernel_vectorized<decltype(tensor_S), decltype(tensor_D), MMA>, 
                                       CopyKernelVectorizedName<decltype(tensor_S), decltype(tensor_D), MMA>>(
      gridDim, blockDim, tensor_S, tensor_D, MMA{});
  compat::wait_and_throw();

  //
  // Verify
  //

  compat::memcpy<Element>(h_D.data(), d_D, size(tensor_shape));

  int32_t errors = 0;
  int32_t const kErrorLimit = 10;

  for (size_t i = 0; i < h_D.size(); ++i) {
    if (h_S[i] != h_D[i]) {
      std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

      if (++errors >= kErrorLimit) {
        std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
        return -1;
      }
    }
  }

  std::cout << "Success." << std::endl;

  return 0;
}
