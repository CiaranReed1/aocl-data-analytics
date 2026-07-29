/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "aoclda.h"
#include "da_utils.hpp"
#include <immintrin.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>

#ifdef _OPENMP
#include <omp.h>
#endif


namespace {

    void flush_cache(std::size_t size_bytes = 512ULL * 1024ULL * 1024ULL)
    {
        static std::vector<std::uint8_t> buffer(size_bytes, 1);

        const volatile std::uint8_t* data = buffer.data();

        std::uint64_t sum = 0;

        constexpr std::size_t cache_line_size = 64;

        for (std::size_t i = 0; i < buffer.size(); i += cache_line_size)
        {
            sum += data[i];
        }

        asm volatile("" : : "r"(sum) : "memory");
    }

}

static int get_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <typename T>
std::string type_name();

template <>
std::string type_name<float>()
{
    return "float";
}

template <>
std::string type_name<double>()
{
    return "double";
}

struct Shape {
    // m by n matrix: m rows, n columns.
    da_int m;
    da_int n;
};

std::vector<Shape> make_aspect_ratio_shapes(std::size_t num_elems)
{
    const std::vector<int> narrow_dims = {
        4, 5, 6, 7,
        8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24,
        25, 26, 27, 28, 29, 30, 31, 32, 33,
        36, 40,
        47, 48, 49,
        56,
        63, 64, 65,
        80,
        95, 96, 97,
        112,
        127, 128, 129,
        160,
        192,
        256
    };

    std::vector<Shape> shapes;
    std::unordered_set<std::string> seen_shapes;

    auto append_unique_shape =
        [&](int m, int n)
        {
            if (m <= 0 || n <= 0)
            {
                return;
            }

            const std::string key =
                std::to_string(m) + "x" + std::to_string(n);

            if (seen_shapes.insert(key).second)
            {
                shapes.push_back({m, n});
            }
        };

    for (const int narrow : narrow_dims)
    {
        if (narrow <= 0)
        {
            continue;
        }

        const std::size_t narrow_size =
            static_cast<std::size_t>(narrow);

        if (narrow_size > num_elems)
        {
            continue;
        }

        const std::size_t major_size =
            num_elems / narrow_size;

        if (major_size == 0)
        {
            continue;
        }

        if (major_size > static_cast<std::size_t>(INT_MAX))
        {
            std::cerr
                << "# Skipping narrow dimension=" << narrow
                << ": calculated major dimension exceeds INT_MAX\n";
            continue;
        }

        const int major = static_cast<int>(major_size);

        if (major < narrow)
        {
            continue;
        }

        // Tall-skinny.
        append_unique_shape(major, narrow);

        // Short-wide.
        if (major != narrow)
        {
            append_unique_shape(narrow, major);
        }
    }

    return shapes;
}

static const char *direction_name(da_order input_order)
{
    return input_order == row_major ? "row_to_col" : "col_to_row";
}

static const char *kernel_name(da_order input_order)
{
    return input_order == row_major
        ? "copy_transpose_2D_array_row_to_column_major"
        : "copy_transpose_2D_array_column_to_row_major";
}

// Returns a predictable value for a specific i, j input.
template <typename T>
T value_at(da_int i, da_int j)
{
    return static_cast<T>((i % 1000) * 0.25 +
                          (j % 1000) * 0.5 +
                          1.0);
}

// Checks if two matrices have been out-of-place transposed correctly.
template <typename T>
bool check_copy_result(da_order input_order,
                       da_int m,
                       da_int n,
                       const T *X,
                       da_int ldx,
                       const T *Y,
                       da_int ldy)
{
    for (da_int i = 0; i < m; ++i) {
        for (da_int j = 0; j < n; ++j) {
            T expected;
            T got;

            if (input_order == row_major) {
                expected = X[static_cast<size_t>(i) * ldx + j];
                got = Y[static_cast<size_t>(i) +
                        static_cast<size_t>(j) * ldy];
            } else {
                expected = X[static_cast<size_t>(i) +
                             static_cast<size_t>(j) * ldx];
                got = Y[static_cast<size_t>(i) * ldy + j];
            }

            if (expected != got) {
                return false;
            }
        }
    }

    return true;
}

template <typename T>
void call_internal_transpose_helper(da_order input_order,
                                    da_int m,
                                    da_int n,
                                    const T *A,
                                    da_int lda,
                                    T *B,
                                    da_int ldb)
{
    if (input_order == row_major) {
        ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
            m, n, A, lda, B, ldb);
    } else {
        ARCH::da_utils::copy_transpose_2D_array_column_to_row_major<T>(
            m, n, A, lda, B, ldb);
    }
}

template <typename T>
void print_raw_result_csv(da_order input_order,
                          da_int m,
                          da_int n,
                          int threads,
                          int repeats,
                          int repeat_index,
                          double seconds,
                          double gbps,
                          bool correct,
                          da_int lda,
                          da_int ldb)
{
    constexpr int tile_size = 0;
    constexpr bool forced_aligned = false;
    constexpr size_t alignment = 0;

    const char *layout_case = "contiguous_unpadded";
    const char *padding_mode = "none";

    const da_int logical_lda = lda;
    const da_int logical_ldb = ldb;

    std::cout << type_name<T>() << ","
              << kernel_name(input_order) << ","
              << direction_name(input_order) << ","
              << m << ","
              << n << ","
              << threads << ","
              << repeats << ","
              << repeat_index << ","
              << seconds << ","
              << gbps << ","
              << (correct ? "true" : "false") << ","
              << layout_case << ","
              << tile_size << ","
              << lda << ","
              << ldb << ","
              << logical_lda << ","
              << logical_ldb << ","
              << (forced_aligned ? "true" : "false") << ","
              << alignment << ","
              << padding_mode << "\n";
}

template<typename T>
void bench_copy_raw(da_order input_order, da_int m, da_int n, int repeats)
{
    // A is m by n. For contiguous storage, the leading dimension is the
    // number of columns in row-major and the number of rows in column-major.
    const da_int lda = input_order == row_major ? n : m;

    // B stores the transposed output. For row -> column, B is column-major
    // with logical shape m by n, so ldb is m. For column -> row, B is
    // row-major with logical shape m by n, so ldb is n.
    const da_int ldb = input_order == row_major ? m : n;

    const size_t size = static_cast<size_t>(m) * static_cast<size_t>(n);

    T *A = new T[size];
    T *B = new T[size];

    for (da_int i = 0; i < m; ++i) {
        for (da_int j = 0; j < n; ++j) {
            if (input_order == row_major) {
                A[static_cast<size_t>(i) * lda + j] = value_at<T>(i, j);
            } else {
                A[static_cast<size_t>(i) +
                  static_cast<size_t>(j) * lda] = value_at<T>(i, j);
            }
        }
    }

    const double bytes = 2.0 *
                         static_cast<double>(m) *
                         static_cast<double>(n) *
                         static_cast<double>(sizeof(T));

    // Warm up once and check correctness outside the timed loop.
    call_internal_transpose_helper<T>(input_order, m, n, A, lda, B, ldb);

    const bool correct = check_copy_result<T>(
        input_order,
        m,
        n,
        A,
        lda,
        B,
        ldb);

    const int threads = get_threads();

    for (int r = 0; r < repeats; ++r) {

        flush_cache();
        const auto t0 = std::chrono::steady_clock::now();

        call_internal_transpose_helper<T>(input_order, m, n, A, lda, B, ldb);

        _mm_sfence();
        const auto t1 = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(t1 - t0).count();
        const double gbps = bytes / seconds / 1.0e9;

        print_raw_result_csv<T>(
            input_order,
            m,
            n,
            threads,
            repeats,
            r,
            seconds,
            gbps,
            correct,
            lda,
            ldb);
    }

    delete[] A;
    delete[] B;
}

template<typename T>
void run_type_benchmarks(const std::vector<Shape> &shapes, int repeats)
{
    for (const auto &s : shapes) {
        bench_copy_raw<T>(row_major, s.m, s.n, repeats);
        bench_copy_raw<T>(column_major, s.m, s.n, repeats);
    }
}

int main(int argc, char **argv)
{
    int repeats = 10;
    std::uint64_t num_elements = 0;

    try {
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];

            if (argument == "--repeats" && i + 1 < argc) {
                repeats = std::max(1, std::stoi(argv[++i]));
            } else if (argument == "--num-elements" && i + 1 < argc) {
                num_elements = std::stoull(argv[++i]);
            } else {
                std::cerr << "Usage: " << argv[0]
                          << " --num-elements N [--repeats R]\n";
                return 1;
            }
        }

        if (num_elements == 0) {
            throw std::invalid_argument("--num-elements must be greater than zero");
        }

        const std::vector<Shape> shapes = make_aspect_ratio_shapes(num_elements);

        std::cout << "type,kernel,direction,m,n,threads,repeats,repeat,"
                  << "seconds,gbps,correct,"
                  << "layout_case,tile_size,lda,ldb,logical_lda,logical_ldb,"
                  << "forced_aligned,alignment,padding_mode\n";

        run_type_benchmarks<float>(shapes, repeats);
        run_type_benchmarks<double>(shapes, repeats);
    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
