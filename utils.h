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

#include "verification.h"


using std::string;
using std::ifstream;
using std::min;
using std::pair;
using std::vector;
using size_type = unsigned;

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {
/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */
int
sgemm_ (
  const char *transa, const char *transb,
  FINTEGER *m, FINTEGER * n, FINTEGER *k,
  const float *alpha, const float *a,
  FINTEGER *lda, const float *b, FINTEGER *ldb,
  float *beta, float *c, FINTEGER *ldc);

int dgemm_(
  char *transa, char *transb,
  FINTEGER *m, FINTEGER *n, FINTEGER *k,
  const double *alpha, const double *a,
  FINTEGER *lda, const double *b, FINTEGER *ldb,
  const double *beta, const double *c, FINTEGER *ldc);
}


void load_data(
  const string& str_location,
  vector<string>& strings,
  vector<size_type >& signatures,
  size_type& num_dict, size_type& num_str) {

  ifstream  str_reader(str_location);

  num_str = 0;

  string line;
  while (getline(str_reader, line)) {
    // record the string
    strings.push_back(line);
    // record the number of strings
    num_str++;
    // record the signatures and the number of identical characters

    for (char c : line) {
      if (signatures[c] == 1024) {
        signatures[c] = num_dict++;
      }
    }
  }
  str_reader.close();

}


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







template <typename T>
vector<size_t> arg_sort(const T* v, const size_t size_v, const size_t step) {
  // initialize original index locations
  vector<size_t> idx(size_v);
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(
    idx.begin(), idx.end(),
    [&v, step](size_t i1, size_t i2) {
      return v[i1 * step] < v[i2 * step];}
  );
  return idx;
}


template <typename T>
vector<size_t> arg_sort(const vector<T> &v) {
  return arg_sort(v.data(), v.size(), 0);
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
