//
// Created by xinyan on 29/12/2019.
//

#pragma once
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string.h>
#include <omp.h>

using size_type = unsigned;
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
        cgk_randoms[i][j] = rand() % 1;
      }
    }
  }

  string embed(const string& x) {
    string out(cgk_l_, '\0');
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

class StringLSH {
 public:
  StringLSH(
    const size_type num_hash,
    const size_type num_bits,
    const size_type cgk_l,
    const size_type num_dict,
    const vector<size_type >& signatures)
    :
    num_hash_(num_hash),
    cgk_l_(cgk_l),
    num_bits_(num_bits),
    hash_lsh_(num_hash, vector<size_type >(num_bits_, 0)),
    index_(num_hash, IndexType()),
    cgk_embed_(cgk_l, num_dict, signatures) {
    // initialize LSH
    for (int i = 0; i < num_bits; ++i) {
      for (int j = 0; j < num_bits; ++j) {
        hash_lsh_[i][j] = rand() % cgk_l;
      }
    }
  }
  vector<string > hash(const string& x) {
    string cgk = cgk_embed_.embed(x);
    vector<string > res(num_hash_, string(num_bits_, '\0'));
    for (int i = 0; i < num_hash_; ++i) {
      for (int j = 0; j < num_bits_; ++j) {
        res[i][j] = cgk[hash_lsh_[i][j]];
      }
    }
    return res;
  }

  unordered_set<size_type > query(const string& x) {
    unordered_set<size_type > res;
    vector<string> hash_val = hash(x);

    for (int i = 0; i < num_hash_; ++i) {
      const vector<size_type >& candidates = index_[i][hash_val[i]];
      std::copy(candidates.begin(), candidates.end(), std::inserter(res, res.end()));
    }
    return res;
  }

  void add_hash_val(const vector<string>& hash_val, int id) {
    for (int i = 0; i < num_hash_; ++i) {
      index_[i][hash_val[i]].push_back(id);
    }
  }

  void add(const string& x, int id) {
    add_hash_val(hash(x), id);
  }


  void add(const vector<string>& xs) {
    vector<vector<string> > hash_vals(xs.size(), vector<string>());
//#pragma omp parallel for
    for (int k = 0; k < xs.size(); ++k) {
      hash_vals[k] = hash(xs[k]);
    }
//#pragma omp parallel for
    for (int h = 0; h < num_hash_; ++h) {
      for (int k = 0; k < hash_vals.size(); ++k) {
        index_[h][hash_vals[k][h]].push_back(k);
      }
    }
  }

 private:
  const size_type num_hash_;
  const size_type num_bits_;
  const size_type cgk_l_;
  vector<vector<size_type > > hash_lsh_;

  vector<IndexType> index_;
  CGKEmbed cgk_embed_;

};
