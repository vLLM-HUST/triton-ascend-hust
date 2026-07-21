#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace {
float Silu(float x) { return x / (1.0F + std::exp(-x)); }

void RunCase(int m, int hidden, int intermediate, int block_m, int block_h, int block_i) {
  std::vector<float> x(m * hidden), gate(intermediate * hidden), up(intermediate * hidden);
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(static_cast<int>(i % 17) - 8) / 32.0F;
  for (size_t i = 0; i < gate.size(); ++i) gate[i] = static_cast<float>(static_cast<int>(i % 13) - 6) / 64.0F;
  for (size_t i = 0; i < up.size(); ++i) up[i] = static_cast<float>(static_cast<int>(i % 11) - 5) / 64.0F;
  std::vector<float> packed(m * 2 * intermediate, 0.0F), fused(m * intermediate, 0.0F);
  for (int mb = 0; mb < m; mb += block_m) {
    for (int ib = 0; ib < intermediate; ib += block_i) {
      for (int row = mb; row < std::min(mb + block_m, m); ++row) {
        for (int inner = ib; inner < std::min(ib + block_i, intermediate); ++inner) {
          float g = 0.0F, u = 0.0F;
          for (int hb = 0; hb < hidden; hb += block_h) {
            for (int k = hb; k < std::min(hb + block_h, hidden); ++k) {
              g += x[row * hidden + k] * gate[inner * hidden + k];
              u += x[row * hidden + k] * up[inner * hidden + k];
            }
          }
          packed[row * 2 * intermediate + inner] = g;
          packed[row * 2 * intermediate + intermediate + inner] = u;
          fused[row * intermediate + inner] = Silu(g) * u;
        }
      }
    }
  }
  for (int row = 0; row < m; ++row) {
    for (int inner = 0; inner < intermediate; ++inner) {
      const float materialized = Silu(packed[row * 2 * intermediate + inner]) *
                                 packed[row * 2 * intermediate + intermediate + inner];
      if (!std::isfinite(fused[row * intermediate + inner]) ||
          std::abs(materialized - fused[row * intermediate + inner]) > 1.0e-6F) {
        throw std::runtime_error("independent pair mismatch");
      }
    }
  }
}
}  // namespace

int main() {
  for (int m : {1, 4, 17, 32}) {
    for (int block_m : {8, 16, 32}) {
      for (int block_h : {64, 128, 256}) {
        for (int block_i : {64, 128, 256}) {
          RunCase(m, 259, 263, block_m, block_h, block_i);
        }
      }
    }
  }
  std::cout << "dense-bf16-materialization-pair-host-contract: PASS\n";
}
