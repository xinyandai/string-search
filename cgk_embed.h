//
// Created by xinyan on 29/12/2019.
//

#pragma once
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <omp.h>
#include <boost/progress.hpp>

#include "utils.h"


using std::string;
using std::vector;
using std::unordered_set;
using std::unordered_map;
using IndexType = unordered_map<string, vector<size_type > >;

class CGKEmbed {
 public:
  CGKEmbed(
    const size_type cgk_l,
    const size_type num_dict,
    const vector<size_type >& signatures)
    :
    cgk_l_(cgk_l),
    signatures_(signatures),
    cgk_randoms(cgk_l, vector<int >(num_dict, 0)) {

    for (int i = 0; i < cgk_randoms.size(); ++i) {
      for (int j = 0; j < num_dict; ++j) {
        cgk_randoms[i][j] = rand() % 2; // 0 or 1
      }
    }
  }

  string embed(const string& x) const {
    string out(cgk_l_, ' ');
    int i = 0;
    int j = 0;

    while (j < cgk_l_ && i < x.size()) {
      assert(j < cgk_randoms.size());
      assert(signatures_[x[i]] >= 0);
      assert(signatures_[x[i]] < cgk_randoms[j].size());
      out[j] = x[i];
      i += cgk_randoms[j][signatures_[x[i]]];
      j += 1;
    }
    return out;
  }

 private:
  const size_type cgk_l_;
  vector<vector<int > > cgk_randoms;
  const vector<size_type >& signatures_;
};


class CGKRanker {

 public:
  CGKRanker(
    const size_type num_bits,
    const size_type num_cgk,
    const size_type cgk_l,
    const size_type num_dict,
    const vector<size_type >& signatures)
    :
    num_bits_(num_bits), num_cgk_(num_cgk), cgk_l_(cgk_l), hash_lsh_(num_cgk) {

    // initialize LSH
    for (int i = 0; i < num_cgk; ++i) {

      std::vector<int> indices(cgk_l);
      std::iota(indices.begin(), indices.end(), 0);
      std::random_device rd;
      std::mt19937 eng(rd());
      std::shuffle(indices.begin(), indices.end(), eng);
      hash_lsh_[i] = vector<size_type >(indices.begin(), indices.begin() + num_bits);
    }

    // construction
    cgk_embed_.reserve(num_cgk);
    for (int i = 0; i < num_cgk; ++i) {
      cgk_embed_.emplace_back(cgk_l, num_dict, signatures);
    }
  }

  string embed(const string& x) const {
    string res;
    res.reserve(num_cgk_ * num_bits_);

    for (int i = 0; i < num_cgk_; i++) {
      string cgk_x = cgk_embed_[i].embed(x);
      for (int j = 0; j < num_bits_; ++j) {
        res += cgk_x[ hash_lsh_[i][j] ];
      }
    }
    return res;
  }

  void add(const vector<string>& xs) {
    size_t max_n = xs.size();
    assert(embed_string_.empty());
    embed_string_ = vector<string >(max_n, "");

    boost::progress_display progress_embed(max_n);

#pragma omp parallel for
    for (int i = 0; i < max_n; ++i) {
      embed_string_[i] = embed(xs[i]);
    }
  }

  vector<int > query(const string& x) const {
    vector<int > dist(embed_string_.size(), 0);
    string cgk_x = embed(x);

    for (int i = 0; i < dist.size(); ++i) {
      dist[i] = hamming_dist(cgk_x, embed_string_[i]);
    }
    return dist;
  }

 private:
  const size_type num_bits_;
  const size_type num_cgk_;
  const size_type cgk_l_;
  vector<string > embed_string_;
  vector<vector<size_type > > hash_lsh_;
  vector<CGKEmbed > cgk_embed_;
};

