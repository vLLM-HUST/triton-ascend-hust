# Dense BF16 SwiGLU Stage 0 prototype

This directory is an independent, correctness-first prototype for
`down(silu(gate(x)) * up(x))`. Compiler, kernel, primitive, and direct-test
ownership is in `triton-ascend-hust`; serving integration is deliberately absent.

Stage 0 evidence is limited to a clean host C++ build/test and static source and
ABI validation. The AscendC source has not been compiled with CANN, launched on
an NPU, profiled, or shown to be correct or performant on an accelerator. The
finite schedule space in `contract.json` is frozen by the evidence-redesign
protocol. The corrected package is `REQUEST_ONLY_NOT_AUTHORIZED`: it cannot
perform preflight, generate a reservation request, or authorize execution.
