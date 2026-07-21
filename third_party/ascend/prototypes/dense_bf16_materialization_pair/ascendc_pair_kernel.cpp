// Independent tiled AscendC pair for the packed gate/up materialization boundary.
// This source is never compiled or launched before an exact central grant.
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

struct DenseBf16PairTiling {
  uint32_t m;
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t block_m;
  uint32_t block_h;
  uint32_t block_i;
  uint32_t row_tiles;
  uint32_t intermediate_tiles;
  uint32_t tail_m;
  uint32_t tail_h;
  uint32_t tail_i;
  uint64_t schedule_identity;
};

using InputType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
using WeightType = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t, true>;
using AccType = MatmulType<TPosition::VECCALC, CubeFormat::ND, float>;
using BiasType = MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <bool MaterializePacked>
class DenseBf16GateUpPair {
 public:
  __aicore__ inline DenseBf16GateUpPair(
      GM_ADDR x, GM_ADDR gate_weight, GM_ADDR up_weight, GM_ADDR output,
      const DenseBf16PairTiling* tiling)
      : spec_(tiling) {
    input_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x), spec_->m * spec_->hidden);
    gate_weight_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(gate_weight), spec_->intermediate * spec_->hidden);
    up_weight_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(up_weight), spec_->intermediate * spec_->hidden);
    output_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output),
                            spec_->m * (MaterializePacked ? 2 * spec_->intermediate : spec_->intermediate));
  }

  __aicore__ inline void Run() {
    if (!ValidTiling()) return;
    const uint32_t tile = GetBlockIdx();
    if (tile >= spec_->row_tiles * spec_->intermediate_tiles) return;
    const uint32_t tile_m = tile / spec_->intermediate_tiles;
    const uint32_t tile_i = tile % spec_->intermediate_tiles;
    const uint32_t m_begin = tile_m * spec_->block_m;
    const uint32_t i_begin = tile_i * spec_->block_i;
    const uint32_t valid_m = Min(spec_->block_m, spec_->m - m_begin);
    const uint32_t valid_i = Min(spec_->block_i, spec_->intermediate - i_begin);

    // Both entry points execute these exact two Tmat-selected producer matmuls.
    gate_mm_.SetSingleShape(valid_m, valid_i, spec_->hidden);
    gate_mm_.SetTensorA(input_[m_begin * spec_->hidden]);
    gate_mm_.SetTensorB(gate_weight_[i_begin * spec_->hidden], true);
    up_mm_.SetSingleShape(valid_m, valid_i, spec_->hidden);
    up_mm_.SetTensorA(input_[m_begin * spec_->hidden]);
    up_mm_.SetTensorB(up_weight_[i_begin * spec_->hidden], true);
    gate_mm_.Iterate();
    up_mm_.Iterate();

    LocalTensor<float> gate = gate_mm_.GetTensorC<false>();
    LocalTensor<float> up = up_mm_.GetTensorC<false>();
    if constexpr (MaterializePacked) {
      CopyPackedTile(gate, up, m_begin, i_begin, valid_m, valid_i);
    } else {
      LocalTensor<float> sigmoid = scratch_.Get<float>();
      LocalTensor<float> activation = activation_.Get<float>();
      Sigmoid(sigmoid, gate, valid_m * valid_i);
      Mul(activation, gate, sigmoid, valid_m * valid_i);
      Mul(activation, activation, up, valid_m * valid_i);
      CopyActivationTile(activation, m_begin, i_begin, valid_m, valid_i);
    }
    gate_mm_.End();
    up_mm_.End();
  }

 private:
  __aicore__ inline bool ValidTiling() const {
    return spec_->m > 0 && spec_->hidden > 0 && spec_->intermediate > 0 &&
           (spec_->block_m == 8 || spec_->block_m == 16 || spec_->block_m == 32) &&
           (spec_->block_h == 64 || spec_->block_h == 128 || spec_->block_h == 256) &&
           (spec_->block_i == 64 || spec_->block_i == 128 || spec_->block_i == 256) &&
           spec_->row_tiles == (spec_->m + spec_->block_m - 1) / spec_->block_m &&
           spec_->intermediate_tiles ==
               (spec_->intermediate + spec_->block_i - 1) / spec_->block_i;
  }

  __aicore__ inline uint32_t Min(uint32_t left, uint32_t right) const {
    return left < right ? left : right;
  }

  __aicore__ inline void CopyPackedTile(LocalTensor<float> gate, LocalTensor<float> up,
                                        uint32_t m_begin, uint32_t i_begin,
                                        uint32_t valid_m, uint32_t valid_i) {
    DataCopyExtParams copy{static_cast<uint16_t>(valid_m),
                           static_cast<uint32_t>(valid_i * sizeof(bfloat16_t)),
                           0, static_cast<uint32_t>((spec_->intermediate - valid_i) * sizeof(bfloat16_t)), 0};
    DataCopyPad(output_[m_begin * 2 * spec_->intermediate + i_begin], gate, copy);
    DataCopyPad(output_[m_begin * 2 * spec_->intermediate + spec_->intermediate + i_begin], up, copy);
  }

  __aicore__ inline void CopyActivationTile(LocalTensor<float> activation,
                                            uint32_t m_begin, uint32_t i_begin,
                                            uint32_t valid_m, uint32_t valid_i) {
    DataCopyExtParams copy{static_cast<uint16_t>(valid_m),
                           static_cast<uint32_t>(valid_i * sizeof(bfloat16_t)),
                           0, static_cast<uint32_t>((spec_->intermediate - valid_i) * sizeof(bfloat16_t)), 0};
    DataCopyPad(output_[m_begin * spec_->intermediate + i_begin], activation, copy);
  }

  const DenseBf16PairTiling* spec_;
  GlobalTensor<bfloat16_t> input_;
  GlobalTensor<bfloat16_t> gate_weight_;
  GlobalTensor<bfloat16_t> up_weight_;
  GlobalTensor<bfloat16_t> output_;
  TBuf<TPosition::VECCALC> scratch_;
  TBuf<TPosition::VECCALC> activation_;
  Matmul<InputType, WeightType, AccType, BiasType> gate_mm_;
  Matmul<InputType, WeightType, AccType, BiasType> up_mm_;
};

extern "C" __global__ __aicore__ void dense_bf16_pair_tmat(
    GM_ADDR x, GM_ADDR gate_weight, GM_ADDR up_weight, GM_ADDR packed,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  const auto* spec = reinterpret_cast<__gm__ DenseBf16PairTiling*>(tiling);
  DenseBf16GateUpPair<true> op(x, gate_weight, up_weight, packed, spec);
  op.Run();
}
extern "C" __global__ __aicore__ void dense_bf16_pair_tfused(
    GM_ADDR x, GM_ADDR gate_weight, GM_ADDR up_weight, GM_ADDR activation,
    GM_ADDR workspace, GM_ADDR tiling) {
  (void)workspace;
  const auto* spec = reinterpret_cast<__gm__ DenseBf16PairTiling*>(tiling);
  DenseBf16GateUpPair<false> op(x, gate_weight, up_weight, activation, spec);
  op.Run();
}
