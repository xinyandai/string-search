//
// Created by xinyan on 30/12/2019.
//

#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <time.h>
#include "edit_distance.h"

using std::string;
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
    dist[i] = edit_distance(query, base_strings[idx[i]]);
  }

  vector<size_t > res = arg_sort(dist);
  for (int i = 0; i < num_prob; ++i) {
    res[i] = idx[res[i]];
  }
  return res;
}


int hamming_dist(const string& a, const string& b) {
  int dist = 0;
  if (a.size() != b.size()) {
    dist = std::abs((int)a.size() - (int)b.size());
  }
  for (int i = 0; i < std::min(a.size(), b.size()); ++i) {
    if (a[i]!=b[i])
      dist++;
  }
  return dist;
}


/**
 * A timer object measures elapsed time, and it is very similar to boost::timer.
 */
class timer {
 public:
  timer(): time(static_cast<double>(clock())) {}
  ~timer() {}

  /**
   * Restart the timer.
   */
  void restart() {
    time = static_cast<double>(clock());
  }
  /**
   * Measures elapsed time.
   * @return The elapsed time
   */
  double elapsed() {
    return (static_cast<double>(clock()) - time) / CLOCKS_PER_SEC;
  }

 private:
  double time;
};
