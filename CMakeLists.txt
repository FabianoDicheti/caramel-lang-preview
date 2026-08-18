# ============================================================================
# Caramel Language - top-level build
# ----------------------------------------------------------------------------
# Two independent halves, selected by CARAMEL_ENABLE_MLIR:
#
#   default (MLIR off)  - caramel_core plus the caramel-run / caramel-lint /
#                         crpk-gen CLIs. Needs nothing but a C++17 compiler,
#                         which is what an end user installing the language
#                         gets:
#                             cmake -S . -B build
#                             cmake --build build -j
#                             sudo cmake --install build
#
#   -DCARAMEL_ENABLE_MLIR=ON - additionally builds the MLIR dialects, the
#                         lowering passes and caramel-opt. This is the
#                         compiler-development half and needs a matching
#                         LLVM/MLIR 18 install:
#                             cmake -S . -B build -G Ninja \
#                               -DCARAMEL_ENABLE_MLIR=ON \
#                               -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir \
#                               -DLLVM_DIR=/usr/lib/llvm-18/lib/cmake/llvm
#
#   ctest --test-dir build --output-on-failure
#
# See INSTALL.md for the user-facing paths and the
# MLIR section there for the compiler-development toolchain.
# ============================================================================
cmake_minimum_required(VERSION 3.20)
project(caramel_lang VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CARAMEL_ENABLE_MLIR "Build the MLIR dialects, passes and caramel-opt (requires LLVM/MLIR 18)" OFF)
option(CARAMEL_BUILD_TESTS "Build the ctest suite" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

# --- Locate MLIR / LLVM (optional half) ------------------------------------
if(CARAMEL_ENABLE_MLIR)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in : ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in : ${LLVM_DIR}")
  message(STATUS "LLVM/MLIR version         : ${LLVM_PACKAGE_VERSION}")

  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
  include(TableGen)
  include(AddLLVM)
  include(AddMLIR)
  include(HandleLLVMOptions)

  # Include paths for generated *.inc (build/include) and LLVM/MLIR.
  include_directories(${PROJECT_BINARY_DIR}/include)
  include_directories(${LLVM_INCLUDE_DIRS})
  include_directories(${MLIR_INCLUDE_DIRS})
  add_definitions(${LLVM_DEFINITIONS})

  # TableGen the Caramel dialect (generates build/include/caramel/mlir/*.inc).
  add_subdirectory(include/caramel/mlir)
else()
  message(STATUS "MLIR half disabled (-DCARAMEL_ENABLE_MLIR=ON to build caramel-opt)")
endif()

# --- Caramel frontend library (no MLIR dependency) -------------------------
add_library(caramel_core STATIC
  src/ast/ast.cpp
  src/types/type.cpp
  src/ops/op_registry.cpp
  src/parse/lexer.cpp
  src/parse/parser.cpp
  src/parse/decorator.cpp
  src/parse/literal_shape.cpp
  src/lint/linter.cpp
  src/ir/dataflow.cpp
  src/ir/binary_ir.cpp
  src/interp/interpreter.cpp
  src/interp/generators.cpp
  src/interp/lambda.cpp
  src/interp/matrix_ops.cpp
  src/interp/int8_emulation.cpp
  # x86_056-059 op-set reference evaluators. src/net/dispatch.cpp calls each of
  # these (activationOpEvaluator / spatialOpEvaluator / shapeOpEvaluator /
  # normOpEvaluator), so leaving them out of the library does not merely drop
  # the ops - it fails to link.
  src/interp/activation.cpp
  src/interp/norm.cpp
  src/interp/shape.cpp
  src/interp/spatial.cpp
  src/analysis/matrix_profile.cpp
  src/quant/quantizer.cpp
  src/proto/crpk.cpp
  src/net/http_client.cpp
  src/net/device.cpp
  src/net/remote_run.cpp
  src/net/jobs.cpp
  src/net/discovery.cpp
  src/net/dispatch.cpp)
target_include_directories(caramel_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_options(caramel_core PRIVATE -Wall -Wextra)
# lang_044: the concurrent dispatcher (src/net/dispatch.cpp) runs flows on
# std::thread, so every caramel_core consumer needs the threads library.
find_package(Threads REQUIRED)
target_link_libraries(caramel_core PUBLIC Threads::Threads)

if(CARAMEL_ENABLE_MLIR)
  # --- MLIR smoke tool (proves the MLIR build integration) -----------------
  add_executable(caramel-mlir-smoke tools/mlir_smoke/caramel_mlir_smoke.cpp)
  target_link_libraries(caramel-mlir-smoke PRIVATE
    MLIRIR
    MLIRParser
    MLIRSupport
    MLIRFuncDialect)

  # --- Caramel MLIR dialect (lang_009) + passes (lang_010) + FPGA (lang_012)
  add_library(CaramelMLIR STATIC
    src/mlir/CaramelDialect.cpp
    src/mlir/LowerToLinalg.cpp
    src/mlir/CaramelFPGADialect.cpp
    src/mlir/LowerToFPGA.cpp
    src/mlir/ProfileMatmul.cpp)
  add_dependencies(CaramelMLIR CaramelOpsIncGen CaramelFPGAOpsIncGen)
  target_link_libraries(CaramelMLIR PUBLIC
    MLIRIR
    MLIRSupport
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRTensorDialect
    MLIRArithDialect)

  # --- caramel-opt driver --------------------------------------------------
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
  add_executable(caramel-opt tools/caramel-opt/caramel_opt.cpp)
  target_link_libraries(caramel-opt PRIVATE
    CaramelMLIR
    MLIROptLib
    MLIRLinalgTransforms
    MLIRBufferizationTransforms
    ${dialect_libs}
    ${conversion_libs})
endif()

# --- caramel-run: interpreter CLI (lang_020, MLIR-free) --------------------
add_executable(caramel-run tools/caramel-run/caramel_run.cpp)
target_link_libraries(caramel-run PRIVATE caramel_core)
target_compile_options(caramel-run PRIVATE -Wall -Wextra)

# --- caramel-lint: static analysis CLI -------------------------------------
add_executable(caramel-lint tools/caramel-lint/caramel_lint.cpp)
target_link_libraries(caramel-lint PRIVATE caramel_core)
target_compile_options(caramel-lint PRIVATE -Wall -Wextra)

# --- crpk-gen: CRPK/CRRS fixture generator (lang_040, MLIR-free) -----------
add_executable(crpk-gen tools/crpk-gen/crpk_gen.cpp)
target_link_libraries(crpk-gen PRIVATE caramel_core)
target_compile_options(crpk-gen PRIVATE -Wall -Wextra)

# --- Unit tests (ctest) ----------------------------------------------------
# FileCheck ships with LLVM. The MLIR half always has it; a plain install may
# not, so the two CLI tests that use it fall back to grep below.
if(CARAMEL_ENABLE_MLIR)
  set(CARAMEL_FILECHECK "${LLVM_TOOLS_BINARY_DIR}/FileCheck")
else()
  find_program(CARAMEL_FILECHECK NAMES FileCheck FileCheck-18 FileCheck-17)
endif()

if(CARAMEL_BUILD_TESTS)
enable_testing()
find_package(Threads REQUIRED)  # test_http_client's stub server uses std::thread

function(caramel_add_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE caramel_core)
  target_compile_options(${name} PRIVATE -Wall -Wextra)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

caramel_add_test(test_type      tests/types/test_type.cpp)
caramel_add_test(test_op_registry tests/ops/test_op_registry.cpp)
caramel_add_test(test_parser    tests/parse/test_parser.cpp)
caramel_add_test(test_decorator tests/parse/test_decorator.cpp)
caramel_add_test(test_literal_shape tests/parse/test_literal_shape.cpp)
caramel_add_test(test_linter tests/lint/test_linter.cpp)
caramel_add_test(test_dataflow tests/ir/test_dataflow.cpp)
caramel_add_test(test_binary_ir tests/ir/test_binary_ir.cpp)
caramel_add_test(test_interpreter tests/interp/test_interpreter.cpp)
caramel_add_test(test_lambda tests/interp/test_lambda.cpp)
caramel_add_test(test_matrix_ops tests/interp/test_matrix_ops.cpp)
caramel_add_test(test_int8_emulation tests/interp/test_int8_emulation.cpp)
caramel_add_test(test_quantizer tests/quant/test_quantizer.cpp)
caramel_add_test(test_matrix_profile tests/analysis/test_matrix_profile.cpp)
caramel_add_test(test_golden_suite tests/interp/test_golden_suite.cpp)
caramel_add_test(test_crpk tests/test_crpk.cpp)
caramel_add_test(test_http_client tests/test_http_client.cpp)
target_link_libraries(test_http_client PRIVATE Threads::Threads)
caramel_add_test(test_device tests/test_device.cpp)
target_link_libraries(test_device PRIVATE Threads::Threads)
# test_remote_execute replays golden CRPK/CRRS bytes from priority/fixtures,
# which lives outside this repository. On a fresh clone the fixtures are absent,
# so the test is only registered when they can be found; regenerate them with
#   crpk-gen examples/matmul_relu.crml --out <dir> --flow layer --in ...
# and point CARAMEL_FIXTURES_DIR at <dir>.
find_path(CARAMEL_FIXTURES_DIR matmul_relu.crrs
  HINTS
    ENV CARAMEL_FIXTURES_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../priority/fixtures
    ${CMAKE_CURRENT_SOURCE_DIR}/priority/fixtures
  NO_DEFAULT_PATH)
if(CARAMEL_FIXTURES_DIR)
  caramel_add_test(test_remote_execute tests/test_remote_execute.cpp)
  target_link_libraries(test_remote_execute PRIVATE Threads::Threads)
  set_tests_properties(test_remote_execute PROPERTIES
    ENVIRONMENT "CARAMEL_FIXTURES_DIR=${CARAMEL_FIXTURES_DIR}")
else()
  message(STATUS "priority/fixtures not found - skipping test_remote_execute")
endif()
caramel_add_test(test_device_syntax tests/test_device_syntax.cpp)
caramel_add_test(test_async_jobs tests/test_async_jobs.cpp)
target_link_libraries(test_async_jobs PRIVATE Threads::Threads)
caramel_add_test(test_async_dispatch tests/test_async_dispatch.cpp)
target_link_libraries(test_async_dispatch PRIVATE Threads::Threads)
caramel_add_test(test_discovery tests/test_discovery.cpp)
target_link_libraries(test_discovery PRIVATE Threads::Threads)

# --- Objectives verification harness (scripts/verify_objectives.sh runs these) ---
caramel_add_test(bench_compile        tests/verify/bench_compile.cpp)
caramel_add_test(bench_parse_scaling  tests/verify/bench_parse_scaling.cpp)
caramel_add_test(verify_quant_accuracy tests/verify/verify_quant_accuracy.cpp)
caramel_add_test(verify_sim_precision tests/verify/verify_sim_precision.cpp)
caramel_add_test(verify_integer_only  tests/verify/verify_integer_only.cpp)
caramel_add_test(verify_zero_overhead tests/verify/verify_zero_overhead.cpp)
caramel_add_test(verify_contract      tests/verify/verify_contract.cpp)

if(CARAMEL_ENABLE_MLIR)
  # Caramel dialect round-trip (lang_009): parse -> verify -> print -> reparse,
  # checked with FileCheck (from the mlir-NN-tools package).
  add_test(NAME caramel_ops_roundtrip
    COMMAND bash -c
      "$<TARGET_FILE:caramel-opt> ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/caramel_ops.mlir \
       | $<TARGET_FILE:caramel-opt> \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/caramel_ops.mlir")

  # Caramel -> Linalg lowering (lang_010).
  add_test(NAME caramel_lower_to_linalg
    COMMAND bash -c
      "$<TARGET_FILE:caramel-opt> --lower-caramel-to-linalg ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_linalg.mlir \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_linalg.mlir")

  # Caramel -> Affine pipeline (lang_011).
  add_test(NAME caramel_lower_to_affine
    COMMAND bash -c
      "$<TARGET_FILE:caramel-opt> --caramel-to-affine ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_affine.mlir \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_affine.mlir")

  # Caramel -> FPGA hardware dialect (lang_012).
  add_test(NAME caramel_lower_to_fpga
    COMMAND bash -c
      "$<TARGET_FILE:caramel-opt> --lower-caramel-to-fpga ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_fpga.mlir \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/lower_to_fpga.mlir")

  add_test(NAME caramel_fpga_verifier_invalid
    COMMAND bash -c
      "out=$(mktemp); trap 'rm -f \"$out\"' EXIT; \
       if $<TARGET_FILE:caramel-opt> ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/fpga_verifier_invalid.mlir > \"$out\" 2>&1; then cat \"$out\"; exit 1; fi; \
       ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/fpga_verifier_invalid.mlir < \"$out\"")

  # Matrix profiling pass (lang_013).
  add_test(NAME caramel_profile_matmul
    COMMAND bash -c
      "$<TARGET_FILE:caramel-opt> --lower-caramel-to-fpga --profile-matmul ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/profile_matmul.mlir \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/mlir/profile_matmul.mlir")
endif()

# Interpreter CLI end-to-end run (lang_020): run a .crml layer and check output.
# Without FileCheck the same two invocations are checked with grep, so the CLI
# stays covered on a bare (MLIR-free) install.
if(CARAMEL_FILECHECK)
  add_test(NAME caramel_run_cli
    COMMAND bash -c
      "$<TARGET_FILE:caramel-run> ${CMAKE_CURRENT_SOURCE_DIR}/examples/matmul_relu.crml \
         --flow layer --in x=2x2:1,2,3,4 --in w=2x2:5,6,7,8 --in bias=2x2:0,0,-100,0 \
       | ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/caramel_run.check")

  add_test(NAME caramel_run_cli_bad_input
    COMMAND bash -c
      "out=$(mktemp); trap 'rm -f \"$out\"' EXIT; \
       if $<TARGET_FILE:caramel-run> ${CMAKE_CURRENT_SOURCE_DIR}/examples/matmul_relu.crml \
            --flow layer --in x=2x2:1,2,3 > \"$out\" 2>&1; then cat \"$out\"; exit 1; fi; \
       ${CARAMEL_FILECHECK} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/caramel_run_bad_input.check < \"$out\"")
else()
  add_test(NAME caramel_run_cli
    COMMAND bash -c
      "$<TARGET_FILE:caramel-run> ${CMAKE_CURRENT_SOURCE_DIR}/examples/matmul_relu.crml \
         --flow layer --in x=2x2:1,2,3,4 --in w=2x2:5,6,7,8 --in bias=2x2:0,0,-100,0 \
       | grep -q 'result : \\[2x2\\] = \\[19, 22, 0, 50\\]'")

  add_test(NAME caramel_run_cli_bad_input
    COMMAND bash -c
      "out=$(mktemp); trap 'rm -f \"$out\"' EXIT; \
       if $<TARGET_FILE:caramel-run> ${CMAKE_CURRENT_SOURCE_DIR}/examples/matmul_relu.crml \
            --flow layer --in x=2x2:1,2,3 > \"$out\" 2>&1; then cat \"$out\"; exit 1; fi; \
       grep -q 'bad --in: x=2x2:1,2,3' \"$out\"")
endif()

add_test(NAME caramel_lint_cli_valid
  COMMAND $<TARGET_FILE:caramel-lint>
          ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/caramel_lint_valid.crml)

add_test(NAME caramel_lint_cli_invalid
  COMMAND bash -c
    "out=$(mktemp); trap 'rm -f \"$out\"' EXIT; \
     if $<TARGET_FILE:caramel-lint> ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli/caramel_lint_invalid.crml > \"$out\" 2>&1; then cat \"$out\"; exit 1; fi; \
     grep -q 'error\\[E003\\].*undefined variable' \"$out\" && \
     grep -q 'warning\\[W001\\].*unused parameter' \"$out\"")

endif()  # CARAMEL_BUILD_TESTS

# --- Installation ----------------------------------------------------------
# `cmake --install build` lays out a normal prefix:
#   <prefix>/bin/{caramel-run,caramel-lint,crpk-gen}
#   <prefix>/lib/libcaramel_core.a
#   <prefix>/include/caramel/**.h
#   <prefix>/share/caramel/examples/*.crml
include(GNUInstallDirs)

install(TARGETS caramel-run caramel-lint crpk-gen
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(CARAMEL_ENABLE_MLIR)
  install(TARGETS caramel-opt RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# The static library and headers let others embed the interpreter in their own
# C++ programs; the MLIR headers are excluded because they are useless without
# an MLIR build.
install(TARGETS caramel_core ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/caramel
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h" PATTERN "mlir" EXCLUDE)

install(DIRECTORY examples/
        DESTINATION ${CMAKE_INSTALL_DATADIR}/caramel/examples
        FILES_MATCHING PATTERN "*.crml")

install(FILES README.md INSTALL.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR}
        OPTIONAL)

# --- Packaging (cpack -G TGZ / -G DEB) -------------------------------------
set(CPACK_PACKAGE_NAME "caramel")
set(CPACK_PACKAGE_VENDOR "Fabiano Dicheti")
# The release workflow passes -DCARAMEL_PACKAGE_VERSION=<tag without the v> so
# the .deb/.tar.gz version tracks the git tag instead of drifting from it.
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
if(CARAMEL_PACKAGE_VERSION)
  set(CPACK_PACKAGE_VERSION "${CARAMEL_PACKAGE_VERSION}")
endif()
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY
    "Caramel - a quantization-first tensor language with an interpreter, linter and MLIR compiler")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/FabianoDicheti/caramel-lang-preview")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "caramel")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr/local")
set(CPACK_STRIP_FILES ON)

# Debian package: `sudo dpkg -i caramel_<version>_amd64.deb`.
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "Fabiano Dicheti <fabiano.dicheti@users.noreply.github.com>")
set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

include(CPack)
