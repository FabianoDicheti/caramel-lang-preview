# TableGen the Caramel dialect (lang_009). Generated *.inc files land in
# ${CMAKE_CURRENT_BINARY_DIR} (= build/include/caramel/mlir) and are referenced as
# "caramel/mlir/<name>.inc" because ${PROJECT_BINARY_DIR}/include is on the include
# path (set in the top-level CMakeLists).
set(LLVM_TARGET_DEFINITIONS CaramelOps.td)
mlir_tablegen(CaramelOps.h.inc -gen-op-decls)
mlir_tablegen(CaramelOps.cpp.inc -gen-op-defs)
mlir_tablegen(CaramelOpsDialect.h.inc -gen-dialect-decls -dialect=caramel)
mlir_tablegen(CaramelOpsDialect.cpp.inc -gen-dialect-defs -dialect=caramel)
add_public_tablegen_target(CaramelOpsIncGen)

# FPGA hardware-ISA dialect (lang_012).
set(LLVM_TARGET_DEFINITIONS CaramelFPGAOps.td)
mlir_tablegen(CaramelFPGAOps.h.inc -gen-op-decls)
mlir_tablegen(CaramelFPGAOps.cpp.inc -gen-op-defs)
mlir_tablegen(CaramelFPGAOpsDialect.h.inc -gen-dialect-decls -dialect=caramel_fpga)
mlir_tablegen(CaramelFPGAOpsDialect.cpp.inc -gen-dialect-defs -dialect=caramel_fpga)
add_public_tablegen_target(CaramelFPGAOpsIncGen)
