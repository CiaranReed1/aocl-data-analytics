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

include_directories(${CMAKE_SOURCE_DIR}/source/include)
include_directories(${CMAKE_SOURCE_DIR}/source)

set(BENCHMARK_EXES 
    bench_internal_transpose_helper
)

add_executable(bench_internal_transpose_helper 
bench_internal_transpose_helper.cpp
)

foreach(target ${BENCHMARK_EXES})
    target_link_libraries(${target} PRIVATE aocl-da)

    if(BUILD_SMP)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()

    target_compile_definitions(${target} PRIVATE ${AOCLDA_ILP64})
    target_link_directories(${target} PRIVATE ${IFORT_LIBS})
    target_link_directories(${target} PRIVATE ${FLANG_LIBS})
    target_link_libraries(${target} PRIVATE ${FORTRAN_RUNTIME})
endforeach()

