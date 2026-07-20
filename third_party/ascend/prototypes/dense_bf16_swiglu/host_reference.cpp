#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

uint16_t FloatToBfloat16(float value) {
  union { float f; uint32_t u; } bits{value};
  const uint32_t rounding_bias = 0x7fffU + ((bits.u >> 16U) & 1U);
  return static_cast<uint16_t>((bits.u + rounding_bias) >> 16U);
}

float Bfloat16ToFloat(uint16_t value) {
  union { uint32_t u; float f; } bits{static_cast<uint32_t>(value) << 16U};
  return bits.f;
}

float Silu(float value) { return value / (1.0F + std::exp(-value)); }

std::vector<float> DenseReference(const std::vector<uint16_t>& x,
                                  const std::vector<uint16_t>& gate,
                                  const std::vector<uint16_t>& up,
                                  const std::vector<uint16_t>& down,
                                  int m, int hidden, int intermediate) {
  std::vector<float> result(static_cast<size_t>(m * hidden), 0.0F);
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < intermediate; ++inner) {
      float gate_sum = 0.0F;
      float up_sum = 0.0F;
      for (int k = 0; k < hidden; ++k) {
        const float input = Bfloat16ToFloat(x[row * hidden + k]);
        gate_sum += input * Bfloat16ToFloat(gate[inner * hidden + k]);
        up_sum += input * Bfloat16ToFloat(up[inner * hidden + k]);
      }
      const float activated = Silu(gate_sum) * up_sum;
      for (int output = 0; output < hidden; ++output) {
        result[row * hidden + output] +=
            activated * Bfloat16ToFloat(down[output * intermediate + inner]);
      }
    }
  }
  return result;
}

std::vector<float> DenseBlocked(const std::vector<uint16_t>& x,
                                const std::vector<uint16_t>& gate,
                                const std::vector<uint16_t>& up,
                                const std::vector<uint16_t>& down,
                                int m, int hidden, int intermediate,
                                int block_m, int block_n, int block_k) {
  std::vector<float> result(static_cast<size_t>(m * hidden), 0.0F);
  for (int row_base = 0; row_base < m; row_base += block_m) {
    for (int output = 0; output < hidden; ++output) {
      for (int row = row_base; row < std::min(row_base + block_m, m); ++row) {
        float output_sum = 0.0F;
        for (int inner_base = 0; inner_base < intermediate; inner_base += block_n) {
          for (int inner = inner_base; inner < std::min(inner_base + block_n, intermediate); ++inner) {
            float gate_sum = 0.0F;
            float up_sum = 0.0F;
            for (int k_base = 0; k_base < hidden; k_base += block_k) {
              for (int k = k_base; k < std::min(k_base + block_k, hidden); ++k) {
                const float input = Bfloat16ToFloat(x[row * hidden + k]);
                gate_sum += input * Bfloat16ToFloat(gate[inner * hidden + k]);
                up_sum += input * Bfloat16ToFloat(up[inner * hidden + k]);
              }
            }
            output_sum += Silu(gate_sum) * up_sum *
                Bfloat16ToFloat(down[output * intermediate + inner]);
          }
        }
        result[row * hidden + output] = output_sum;
      }
    }
  }
  return result;
}

}  // namespace

int main() {
  constexpr int hidden = 8;
  constexpr int intermediate = 12;
  auto make = [](int count, int offset) {
    std::vector<uint16_t> values;
    values.reserve(static_cast<size_t>(count));
    for (int index = 0; index < count; ++index) {
      values.push_back(FloatToBfloat16(static_cast<float>((index + offset) % 13 - 6) / 16.0F));
    }
    return values;
  };
  const auto gate = make(intermediate * hidden, 2);
  const auto up = make(intermediate * hidden, 3);
  const auto down = make(hidden * intermediate, 4);
  for (const int m : {1, 4, 8, 16, 32}) {
    const auto x = make(m * hidden, 1);
    const auto reference = DenseReference(x, gate, up, down, m, hidden, intermediate);
    for (const int block_m : {1, 4, 8, 16, 32}) {
      for (const int block_n : {64, 128}) {
        for (const int block_k : {64, 128}) {
          const auto blocked = DenseBlocked(x, gate, up, down, m, hidden, intermediate,
                                            block_m, block_n, block_k);
          for (size_t index = 0; index < reference.size(); ++index) {
            if (!std::isfinite(blocked[index]) ||
                std::abs(reference[index] - blocked[index]) > 1.0e-5F) {
              std::cerr << "blocked dense BF16 contract mismatch at " << index << '\n';
              return 1;
            }
          }
        }
      }
    }
  }
  for (const auto& shape : {std::pair{3584, 18944}, std::pair{5120, 13824}}) {
    for (const int m : {1, 4, 8, 16, 32}) {
      const uint64_t input_elements = static_cast<uint64_t>(m) * shape.first;
      const uint64_t weight_elements = static_cast<uint64_t>(2) * shape.first * shape.second;
      if (input_elements == 0 || weight_elements <= input_elements ||
          weight_elements > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "real model shape arithmetic contract failed\n";
        return 1;
      }
    }
  }
  std::cout << "dense-bf16-swiglu-host-contract: PASS\n";
  return 0;
}
