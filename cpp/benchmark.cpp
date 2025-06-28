#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include "eytzinger.hpp"

class Timer {
    std::chrono::high_resolution_clock::time_point start_time;
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double elapsed_ms() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // Convert to milliseconds
    }
};

template<typename Func>
double benchmark_function(Func&& func, int iterations = 100) {
    Timer timer;
    timer.start();
    
    for (int i = 0; i < iterations; ++i) {
        func();
    }
    
    return timer.elapsed_ms() / iterations;
}

int main() {
    const int num_elements = 6000000;  // Same as Mojo version
    const int num_keys = 60000;
    const int benchmark_iterations = 10;
    
    std::cout << "Generating " << num_elements << " elements and " << num_keys << " search keys...\n";
    
    // Generate random sorted elements
    std::vector<int> elements(num_elements);
    std::random_device rd;
    std::mt19937 gen(42); // Same seed as Mojo version
    std::uniform_int_distribution<> elem_dist(0, num_elements);
    
    for (int& elem : elements) {
        elem = elem_dist(gen);
    }
    std::sort(elements.begin(), elements.end());
    
    // Generate search keys
    std::vector<int> keys(num_keys);
    std::uniform_int_distribution<> key_dist(0, num_keys);
    for (int& key : keys) {
        key = key_dist(gen);
    }
    
    // Create Eytzinger structure
    std::cout << "Building Eytzinger structure...\n";
    Eytzinger eytz(elements);
    
    std::cout << "\nRunning benchmarks...\n";
    std::cout << std::setw(40) << "Algorithm" << std::setw(15) << "Time (ms)" << std::setw(12) << "Relative" << std::endl;
    std::cout << std::string(67, '-') << std::endl;
    
    // Benchmark naive binary search
    double naive_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += naive_binary_search(elements, key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "Naive binary search" 
              << std::setw(15) << std::fixed << std::setprecision(3) << naive_time
              << std::setw(12) << "1.00x" << std::endl;
    
    // Benchmark std::lower_bound
    double std_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += std_lower_bound(elements, key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "std::lower_bound" 
              << std::setw(15) << std::fixed << std::setprecision(3) << std_time
              << std::setw(12) << std::fixed << std::setprecision(2) << (naive_time / std_time) << "x" << std::endl;
    
    // Benchmark Eytzinger original
    double eytz_orig_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += eytz.lower_bound_original(key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "Eytzinger original" 
              << std::setw(15) << std::fixed << std::setprecision(3) << eytz_orig_time
              << std::setw(12) << std::fixed << std::setprecision(2) << (naive_time / eytz_orig_time) << "x" << std::endl;
    
    // Benchmark Eytzinger fixed iterations
    double eytz_fixed_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += eytz.lower_bound_fixed_iter(key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "Eytzinger fixed iterations" 
              << std::setw(15) << std::fixed << std::setprecision(3) << eytz_fixed_time
              << std::setw(12) << std::fixed << std::setprecision(2) << (naive_time / eytz_fixed_time) << "x" << std::endl;
    
    // Benchmark Eytzinger with prefetch
    double eytz_prefetch_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += eytz.lower_bound_prefetch(key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "Eytzinger with prefetch" 
              << std::setw(15) << std::fixed << std::setprecision(3) << eytz_prefetch_time
              << std::setw(12) << std::fixed << std::setprecision(2) << (naive_time / eytz_prefetch_time) << "x" << std::endl;
    
    // Benchmark Eytzinger fixed iterations with prefetch
    double eytz_fixed_prefetch_time = benchmark_function([&]() {
        volatile int dummy = 0;
        for (int key : keys) {
            dummy += eytz.lower_bound_fixed_iter_prefetch(key);
        }
    }, benchmark_iterations);
    
    std::cout << std::setw(40) << "Eytzinger fixed iter + prefetch" 
              << std::setw(15) << std::fixed << std::setprecision(3) << eytz_fixed_prefetch_time
              << std::setw(12) << std::fixed << std::setprecision(2) << (naive_time / eytz_fixed_prefetch_time) << "x" << std::endl;
    
    // Verify correctness by comparing a few results
    std::cout << "\nVerifying correctness (first 10 searches):\n";
    std::cout << std::setw(8) << "Key" << std::setw(10) << "Naive" << std::setw(10) << "Std" 
              << std::setw(12) << "Eytz Orig" << std::setw(12) << "Eytz Fixed" << std::endl;
    std::cout << std::string(52, '-') << std::endl;
    
    for (int i = 0; i < std::min(10, num_keys); ++i) {
        int key = keys[i];
        int naive_result = naive_binary_search(elements, key);
        int std_result = std_lower_bound(elements, key);
        int eytz_orig_result = eytz.lower_bound_original(key);
        int eytz_fixed_result = eytz.lower_bound_fixed_iter(key);
        
        std::cout << std::setw(8) << key 
                  << std::setw(10) << naive_result
                  << std::setw(10) << std_result
                  << std::setw(12) << eytz_orig_result
                  << std::setw(12) << eytz_fixed_result << std::endl;
    }
    
    return 0;
}