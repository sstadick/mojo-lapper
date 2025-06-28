#pragma once
#include <algorithm>
#include <cmath>
#include <vector>

// Portable implementation of std::__lg
inline int lg(int n) {
  if (n == 0)
    return -1;
  return 31 - __builtin_clz(n);
}

class Eytzinger {
private:
  std::vector<int> t;
  int n;
  int iters;

  int eytzinger(std::vector<int> &arr, int i = 0, int k = 1) {
    if (k <= n) {
      i = eytzinger(arr, i, 2 * k);
      t[k] = arr[i++];
      i = eytzinger(arr, i, 2 * k + 1);
    }
    return i;
  }

public:
  Eytzinger(std::vector<int> &sorted_array) : n(sorted_array.size()) {
    t.resize(n + 1);
    t[0] = -1; // sentinel value less than any element

    eytzinger(sorted_array);
    iters = lg(n + 1);
  }

  // Original while-loop version
  int lower_bound_original(int x) {
    int k = 1;
    while (k <= n) {
      k = 2 * k + (t[k] < x);
    }
    k >>= __builtin_ffs(~k);
    return k;
  }

  // Fixed iteration version (removing last branch)
  int lower_bound_fixed_iter(int x) {
    long k = 1;

    // Execute fixed number of iterations
    for (int i = 0; i < iters; i++) {
      k = 2 * k + (t[k] < x);
    }

    // Final comparison with predication
    int *loc = (k <= n ? t.data() + k : t.data());
    k = 2 * k + (*loc < x);

    // Restore actual index
    k >>= __builtin_ffs(~k);
    return k;
  }

  // Version with prefetch
  int lower_bound_prefetch(int x) {
    long k = 1;
    while (k <= n) {
      __builtin_prefetch(t.data() + std::min(k * 16L, (long)t.size() - 1));
      k = 2 * k + (t[k] < x);
    }
    k >>= __builtin_ffs(~k);
    return k;
  }

  // Fixed iteration version with prefetch
  int lower_bound_fixed_iter_prefetch(int x) {
    int k = 1;

    for (int i = 0; i < iters; i++) {
      __builtin_prefetch(t.data() + std::min(k * 16L, (long)t.size() - 1));
      k = 2 * k + (t[k] < x);
    }

    int *loc = (k <= n ? t.data() + k : t.data());
    k = 2 * k + (*loc < x);

    k >>= __builtin_ffs(~k);
    return k;
  }

  // Get the actual value (for comparison with C++ literature)
  int get_value(int index) { return (index < t.size()) ? t[index] : -1; }

  size_t size() const { return n; }
};

// Standard binary search implementations for comparison
int naive_binary_search(const std::vector<int> &arr, int x) {
  if (arr.empty() || arr[0] >= x)
    return 0;

  int low = 0, high = arr.size();
  while (high - low > 1) {
    int mid = (high + low) / 2;
    if (arr[mid] < x) {
      low = mid;
    } else {
      high = mid;
    }
  }
  return high;
}

int std_lower_bound(const std::vector<int> &arr, int x) {
  return std::lower_bound(arr.begin(), arr.end(), x) - arr.begin();
}