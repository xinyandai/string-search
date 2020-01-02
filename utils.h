//
// Created by xinyan on 30/12/2019.
//

#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <chrono>
#include <ctime>
#include <time.h>


using std::string;
using std::min;
using std::pair;
using std::vector;
using size_type = unsigned;


struct Counter {
  struct value_type {
    template<typename T> value_type(const T&) { }
  };
  void push_back(const value_type&) {
    ++count;
  }
  size_t count = 0;
};


template<
  typename _InputIterator1,
  typename _InputIterator2>
size_t intersection_size(
  _InputIterator1 __first1,
  _InputIterator1 __last1,
  _InputIterator2 __first2,
  _InputIterator2 __last2) {
  Counter c;
  set_intersection(
    __first1, __last1,
    __first2, __last2,
    std::back_inserter(c));
  return c.count;
}


int hamming_dist(const string& a, const string& b) {
  int dist = std::abs((int)a.size() - (int)b.size());
  for (int i = 0; i < min(a.size(), b.size()); ++i) {
    if (a[i]!=b[i])
      dist++;
  }
  return dist;
}


int edit_distance(const string & a, const string& b) {
  int na = (int) a.size();
  int nb = (int) b.size();

  vector<int> f(nb+1, 0);
  for (int j = 1; j <= nb; ++j) {
    f[j] = j;
  }

  for (int i = 1; i <= na; ++i) {
    int prev = i;
    for (int j = 1; j <= nb; ++j) {
      int cur;
      if (a[i-1] == b[j-1]) {
        cur = f[j-1];
      }
      else {
        cur = min(min(f[j-1], prev), f[j]) + 1;
      }

      f[j-1] = prev;
      prev = cur;
    }
    f[nb] = prev;
  }
  return f[nb];
}


template <typename T>
vector<size_t> arg_sort(const vector<T> &v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(
    idx.begin(), idx.end(),
    [&v](size_t i1, size_t i2) {
      return v[i1] < v[i2];}
    );
  return idx;
}


template <typename idx_type>
vector<size_t > ed_rank(
  const idx_type* idx,
  size_type  num_prob,
  const string& query,
  const vector<string >& base_strings) {

  auto nb = (size_type)base_strings.size();

  num_prob = std::min(num_prob, nb);
  vector<size_type > dist(num_prob, 0);
  for (int i = 0; i < num_prob; ++i) {
    dist[i] = (size_type)edit_distance(query, base_strings[idx[i]]);
  }

  vector<size_t > res = arg_sort(dist);
  for (int i = 0; i < num_prob; ++i) {
    res[i] = idx[res[i]];
  }
  return res;
}


/** A timer object measures elapsed time,
 * and it is very similar to boost::timer. */
class timer {
 public:
  timer() { restart(); }
  ~timer() = default;
  /** Restart the timer. */
  void restart() {
    t_start = std::chrono::high_resolution_clock::now();
  }
  /** @return The elapsed time */
  double elapsed() {
    auto t_end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  }

 private:
  std::chrono::high_resolution_clock::time_point t_start;
};
