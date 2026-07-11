get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

include_directories(
    ${PROJECT_SOURCE_DIR}/third_party/ascend/AscendNPU-IR/bishengir/include
    ${PROJECT_BINARY_DIR}/third_party/ascend/AscendNPU-IR/bishengir/include
  )

set(TRITON_DEV_TOOL_EXCLUDE_FROM_ALL)
if(TRITON_BUILD_RUNTIME_TOOLS_ONLY)
  set(TRITON_DEV_TOOL_EXCLUDE_FROM_ALL EXCLUDE_FROM_ALL)
endif()

add_llvm_executable(triton-opt ${TRITON_DEV_TOOL_EXCLUDE_FROM_ALL} triton-opt.cpp PARTIAL_SOURCES_INTENDED)

# TODO: what's this?
llvm_update_compile_flags(triton-opt)
target_link_libraries(triton-opt PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-opt)

add_llvm_executable(triton-reduce ${TRITON_DEV_TOOL_EXCLUDE_FROM_ALL} triton-reduce.cpp PARTIAL_SOURCES_INTENDED)
mlir_check_all_link_libraries(triton-reduce)

llvm_update_compile_flags(triton-reduce)
target_link_libraries(triton-reduce PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRReduceLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-reduce)

add_llvm_executable(triton-lsp ${TRITON_DEV_TOOL_EXCLUDE_FROM_ALL} triton-lsp.cpp PARTIAL_SOURCES_INTENDED)

llvm_update_compile_flags(triton-lsp)
target_link_libraries(triton-lsp PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonAMDGPUTestAnalysis
  TritonTestProton
  # MLIR core
  MLIRLspServerLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
)

mlir_check_all_link_libraries(triton-lsp)


add_llvm_executable(triton-llvm-opt
  ${TRITON_DEV_TOOL_EXCLUDE_FROM_ALL}
  triton-llvm-opt.cpp

  PARTIAL_SOURCES_INTENDED
  DEPENDS
  intrinsics_gen
  SUPPORT_PLUGINS
  )
target_link_libraries(triton-llvm-opt PRIVATE
  TritonLLVMIR

  LLVMAnalysis
  LLVMCore
  LLVMSupport
  LLVMOption
  LLVMCodeGen
  )
export_executable_symbols_for_plugins(triton-llvm-opt)


add_llvm_executable(triton-tensor-layout ${TRITON_DEV_TOOL_EXCLUDE_FROM_ALL} triton-tensor-layout.cpp PARTIAL_SOURCES_INTENDED)
target_link_libraries(triton-tensor-layout PRIVATE
  ${triton_libs}
  # Ascend backend libraries
  DiscreteMaskAccessConversion
  TritonToAnnotation
  TritonToHFusion
  TritonToHIVM
  TritonToLinalg
  TritonToLLVM
  TritonToStructured
  TritonToUnstructure
  # BiShengIR dialects
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
  # tests
  TritonTestAnalysis
  TritonTestDialect
  TritonTestProton
  TritonAMDGPUTestAnalysis
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  )

# triton-mlir-opt: Triton-Ascend optimizer combining MLIR and BishengIR dialects
add_llvm_executable(triton-mlir-opt triton-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-mlir-opt)

set(TRITON_MLIR_OPT_LIBS)
if(NOT TRITON_BUILD_RUNTIME_TOOLS_ONLY)
  list(APPEND TRITON_MLIR_OPT_LIBS ${triton_libs})
endif()

list(APPEND TRITON_MLIR_OPT_LIBS
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

target_link_libraries(triton-mlir-opt PRIVATE ${TRITON_MLIR_OPT_LIBS})

mlir_check_all_link_libraries(triton-mlir-opt)
