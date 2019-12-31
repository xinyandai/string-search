//
// Created by xinyan on 30/12/2019.
//

#pragma once
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>

using std::string;
using std::vector;


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