# Copyright (C) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met: 1.
# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer. 2. Redistributions in binary
# form must reproduce the above copyright notice, this list of conditions and
# the following disclaimer in the documentation and/or other materials provided
# with the distribution. 3. Neither the name of the copyright holder nor the
# names of its contributors may be used to endorse or promote products derived
# from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

set(INTERNAL_BENCHMARKS
    bench_internal_transpose_helper
)

set(INTERNAL_INC_DIRS
    ${CMAKE_SOURCE_DIR}/source/include
    ${CMAKE_SOURCE_DIR}/source/core/csv
    ${CMAKE_SOURCE_DIR}/source/core/clustering
    ${CMAKE_SOURCE_DIR}/source/core/optimization
    ${CMAKE_SOURCE_DIR}/source/core/data_management
    ${CMAKE_SOURCE_DIR}/source/core/drivers
    ${CMAKE_SOURCE_DIR}/source/core/factorization
    ${CMAKE_SOURCE_DIR}/source/core/nlls
    ${CMAKE_SOURCE_DIR}/source/core/nearest_neighbors
    ${CMAKE_SOURCE_DIR}/source/core/utilities
    ${CMAKE_SOURCE_DIR}/source/core/decision_forest
    ${CMAKE_SOURCE_DIR}/source/core/decision_forest/tree
    ${CMAKE_SOURCE_DIR}/source/core/metrics
    ${CMAKE_SOURCE_DIR}/source/core/linear_model
    ${CMAKE_SOURCE_DIR}/source/core/basic_statistics
    ${CMAKE_SOURCE_DIR}/source/core/dynamic_dispatch
    ${CMAKE_SOURCE_DIR}/source/core/kernel_functions
    ${CMAKE_SOURCE_DIR}/source/core/svm
    ${CMAKE_SOURCE_DIR}/source/core/interpolation
    ${CMAKE_SOURCE_DIR}/source/core/train_test_split
    ${CMAKE_SOURCE_DIR}/source/core/approximate_neighbors
)

add_executable(bench_internal_transpose_helper
    bench_internal_transpose_helper.cpp
)

foreach(benchmark ${INTERNAL_BENCHMARKS})

  target_include_directories(${benchmark}
    PRIVATE
      ${INTERNAL_INC_DIRS}
  )

  target_link_libraries(${benchmark}
    PRIVATE
      aocl-da
  )

  if(BUILD_SMP)
    target_link_libraries(${benchmark}
      PRIVATE
        OpenMP::OpenMP_CXX
    )
  endif()

  target_compile_definitions(${benchmark}
    PRIVATE
      ${AOCLDA_ILP64}
  )

  target_link_directories(${benchmark}
    PRIVATE
      ${IFORT_LIBS}
      ${FLANG_LIBS}
  )

  target_link_libraries(${benchmark}
    PRIVATE
      ${FORTRAN_RUNTIME}
  )

  # Match the architecture namespace used by internal AOCL-DA source files.
  if(ARCH STREQUAL "dynamic" OR ARCH STREQUAL "generic")
    target_compile_definitions(${benchmark}
      PRIVATE
        ARCH_generic
        ZNVER_MAX=generic
    )

  elseif(ARCH STREQUAL "generic_avx512")
    target_compile_definitions(${benchmark}
      PRIVATE
        ARCH_generic_avx512
        ZNVER_MAX=generic_avx512
    )

  elseif(ARCH STREQUAL "native")
    extract_native_namespace(NATIVE_NAMESPACE)

    if(NATIVE_NAMESPACE STREQUAL "generic")
      target_compile_definitions(${benchmark}
        PRIVATE
          ARCH_generic
          ZNVER_MAX=generic
      )
    else()
      target_compile_definitions(${benchmark}
        PRIVATE
          ARCH_znver=${NATIVE_NAMESPACE}
          ZNVER_MAX=zen${NATIVE_NAMESPACE}
      )
    endif()

  else()
    if(${ARCH} MATCHES "^znver([0-9]+)$")
      set(znver ${CMAKE_MATCH_1})
    endif()

    target_compile_definitions(${benchmark}
      PRIVATE
        ARCH_znver=${znver}
        ZNVER_MAX=zen${znver}
    )
  endif()

endforeach()