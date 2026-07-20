// Correctness-first independent AscendC prototype. Stage 0 validates this ABI
// and mathematical boundary statically; a CANN compile and every NPU claim are
// forbidden until a later exact-SHA reservation is granted.
#include "kernel_operator.h"

using namespace AscendC;

struct DenseBf16SwiGluTiling {
  uint32_t m;
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t block_m;
  uint32_t block_n;
  uint32_t block_k;
};

extern "C" __global__ __aicore__ void dense_bf16_swiglu_proto(
    GM_ADDR x, GM_ADDR gate_weight, GM_ADDR up_weight, GM_ADDR down_weight,
    GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  const auto* spec = reinterpret_cast<__gm__ DenseBf16SwiGluTiling*>(tiling);
  GlobalTensor<bfloat16_t> input;
  GlobalTensor<bfloat16_t> gate;
  GlobalTensor<bfloat16_t> up;
  GlobalTensor<bfloat16_t> down;
  GlobalTensor<bfloat16_t> output;
  input.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), spec->m * spec->hidden);
  gate.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gate_weight), spec->intermediate * spec->hidden);
  up.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(up_weight), spec->intermediate * spec->hidden);
  down.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(down_weight), spec->hidden * spec->intermediate);
  output.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y), spec->m * spec->hidden);

  if (spec->block_m == 0 || spec->block_n == 0 || spec->block_k == 0) return;
  const uint32_t row_tile_count = (spec->m + spec->block_m - 1) / spec->block_m;
  const uint32_t block_index = GetBlockIdx();
  const uint32_t column = block_index / row_tile_count;
  const uint32_t row_begin = (block_index % row_tile_count) * spec->block_m;
  if (column >= spec->hidden || row_begin >= spec->m) return;
  const uint32_t row_end = row_begin + spec->block_m < spec->m
      ? row_begin + spec->block_m : spec->m;
  for (uint32_t row = row_begin; row < row_end; ++row) {
    float result = 0.0F;
    for (uint32_t inner_base = 0; inner_base < spec->intermediate;
         inner_base += spec->block_n) {
      const uint32_t inner_end = inner_base + spec->block_n < spec->intermediate
          ? inner_base + spec->block_n : spec->intermediate;
      for (uint32_t inner = inner_base; inner < inner_end; ++inner) {
        float gate_sum = 0.0F;
        float up_sum = 0.0F;
        for (uint32_t k_base = 0; k_base < spec->hidden; k_base += spec->block_k) {
          const uint32_t k_end = k_base + spec->block_k < spec->hidden
              ? k_base + spec->block_k : spec->hidden;
          for (uint32_t k = k_base; k < k_end; ++k) {
            const float value = static_cast<float>(input.GetValue(row * spec->hidden + k));
            gate_sum += value * static_cast<float>(gate.GetValue(inner * spec->hidden + k));
            up_sum += value * static_cast<float>(up.GetValue(inner * spec->hidden + k));
          }
        }
        const float silu = gate_sum / (1.0F + expf(-gate_sum));
        result += silu * up_sum *
            static_cast<float>(down.GetValue(column * spec->intermediate + inner));
      }
    }
    output.SetValue(row * spec->hidden + column, static_cast<bfloat16_t>(result));
  }
}
