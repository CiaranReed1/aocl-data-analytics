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
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <type_traits>

#ifdef _OPENMP
#include <omp.h>
#endif


static int get_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <typename T>
std::string type_name();

template <>
std::string type_name<float>() {
    return "float";
}

template <>
std::string type_name<double>() {
    return "double";
}


struct Shape{
    // m by n matrix (m rows, n columns)
    da_int m;
    da_int n;
};

//results stats struct
struct Stats{
    double min;
    double max;
    double mean;
    double median;
    double stddev;
};

Stats summarise(std::vector<double> values){
    const size_t len = values.size();
    if (len == 0) {
        std::cerr << "Cannot summarise an empty vector\n";
        std::exit(1);
    }
    
    std::sort(values.begin(),values.end());
    double min = values.front();
    double max = values.back();
    double sum = 0;
    for (double v : values)
    {
        sum += v;
    }

    double mean = sum / static_cast<double>(len);
    double median;
    if (len % 2 == 0) {
        median = 0.5 * (values[len / 2 - 1] + values[len / 2]);
    } else {
        median = values[len / 2];
    }
    
    double stddev = 0;
    for(double v : values)
    {
        double diff = (v - mean) * (v-mean);
        stddev += diff;
    }
    stddev = stddev / (static_cast<double>(len -1));
    stddev = std::sqrt(stddev);

 return {min,max, mean, median, stddev};
}

//returns predictable value for specific i and j input
template <typename T>
T value_at(da_int i, da_int j) {
    return static_cast<T>((i % 1000) * 0.25 + (j % 1000) * 0.5 + 1.0);
}

//checks if two matrices have been out-of-place transposed correctly based upon the input order
template <typename T>
bool check_copy_result(da_order input_order,
                       da_int m,
                       da_int n,
                       const T *X,
                       da_int ldx,
                       const T *Y,
                       da_int ldy) {
    for (da_int i = 0; i < m; ++i) {
        for (da_int j = 0; j < n; ++j) {
            T expected;
            T got;

            if (input_order == row_major) {
                expected = X[static_cast<size_t>(i) * ldx + j];
                got = Y[static_cast<size_t>(i) + static_cast<size_t>(j) * ldy];
            } else {
                expected = X[static_cast<size_t>(i) + static_cast<size_t>(j) * ldx];
                got = Y[static_cast<size_t>(i) * ldy + j];
            }

            if (expected != got)
            {
                return false;
            }
        }
    }

    return true;
}


template<typename T>
void bench_copy(da_order input_order, da_int m, da_int n, int repeats)
{
    //leading dimensions of matrices equal number of columns (row major) or number of rows (column major)
    da_int lda = input_order == row_major ? n : m;
    da_int ldb = input_order == row_major ? m : n;

    //initialise matrices
    const size_t size = static_cast<size_t>(m)*static_cast<size_t>(n);
    T *A = new T[size];
    T *B = new T[size];
    for (da_int i = 0; i < m; ++i) {
        for (da_int j = 0; j < n; ++j) {
            if (input_order == row_major) {
                A[static_cast<size_t>(i) * lda + j] = value_at<T>(i, j);
            } 
            else {
                A[static_cast<size_t>(i) + static_cast<size_t>(j) * lda] = value_at<T>(i, j);
            }
        }
    }

    //initialise results vectors
    std::vector<double> times(repeats);
    std::vector<double> gbps(repeats);

    //note expected bytes to be transferred
    double bytes = 2.0 * static_cast<double>(m) * static_cast<double>(n) * sizeof(T);

    //warmup for the transpose function, and check for correctness
    if(input_order == row_major){
        ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                    m, n, A, lda, B, ldb);
    }
    else 
    {
          ARCH::da_utils::copy_transpose_2D_array_column_to_row_major<T>(
                    m, n, A, lda, B, ldb);
    }
    bool correct = check_copy_result<T>(input_order, m, n, A, lda, B, ldb);

    for(int r = 0; r < repeats; r++)
    {
        auto t0 = std::chrono::steady_clock::now();
        if(input_order == row_major)
        {
            ARCH::da_utils::copy_transpose_2D_array_row_to_column_major<T>(
                    m, n, A, lda, B, ldb);
        }
        else{
            ARCH::da_utils::copy_transpose_2D_array_column_to_row_major<T>(
                    m, n, A, lda, B, ldb);
        }
        auto t1 = std::chrono::steady_clock::now();
        times[r] = std::chrono::duration<double>(t1 - t0).count();
        gbps[r] = bytes / times[r] / 1.0e9;
    }

    //calls summarise function on stats
    Stats times_stats = summarise(times);
    Stats gbps_stats = summarise(gbps);

    //print results
    std::cout << type_name<T>() << ","
              << (input_order == row_major ? "copy_transpose_2D_array_row_to_column_major" : "copy_transpose_2D_array_column_to_row_major") << ","
              << (input_order == row_major ? "row_to_column" : "column_to_row") << ","
              << m << ","
              << n << ","
              << get_threads() << ","
              << repeats << ","
              << times_stats.mean << ","
              <<times_stats.median << ","
              <<times_stats.min << ","
              <<times_stats.max << ","
              <<times_stats.stddev << ","
              <<gbps_stats.mean << ","
              <<gbps_stats.median << ","
              <<gbps_stats.min << ","
              <<gbps_stats.max << ","
              <<gbps_stats.stddev << ","
              << (correct ? "true" : "false") << "\n";

    //frees arrays
    delete[] A;
    delete[] B;
}

template<typename T> 
void run_type_benchmarks(const std::vector<Shape> &shapes, int repeats)
{
    for (const auto &s : shapes) {
        bench_copy<T>(row_major, s.m, s.n, repeats);
        bench_copy<T>(column_major, s.m, s.n, repeats);
    }
}

int main(int argc, char **argv){

    int repeats = 10;

    if (argc >= 2) {
        repeats = std::max(1, std::atoi(argv[1]));
    }
    
    //included shapes, specificially included short fat and tall skinny
    std::vector<Shape> shapes = {
       {64, 64},
        {128, 128},
        {256, 256},
        {512, 512},
        {1024, 1024},
        {1000,1000},
        {2048, 2048},
        {2000,2000},
        {4096, 4096},
        {4000,4000},
        {8192,8192},
        {8000,8000},
        {16384,16384},
        {16000,16000},
        {1024, 64},
        {1000,60},
        {64, 1024},
        {60,1000},
        {4096, 256},
        {4096,64},
        {4000,250},
        {4000,60},
        {256, 4096},
        {64,4096},
        {250,4000},
        {60,4000},
        {8192,64},
        {8000,60},
        {64,8192},
        {60,8000},
        {8192,256},
        {256,8192},
        {8000,250},
        {250,8000},
        {10,20000},
        {15,200000},
        {20000,10},
        {200000,15}
    };

    std::cout << "type,kernel,direction,n,m,threads,repeats,"
          << "mean_seconds,median_seconds,min_seconds,max_seconds,stddev_seconds,"
          << "mean_gbps,median_gbps,min_gbps,max_gbps,stddev_gbps,correct\n";

    run_type_benchmarks<float>(shapes, repeats);
    run_type_benchmarks<double>(shapes, repeats);

    return 0;
}