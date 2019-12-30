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