//
// Created by xinyan on 29/12/2019.
//

#pragma once
#include <unordered_set>
#include <unordered_map>
#include <omp.h>

#include "utils.h"

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
    const size_type num_cgk,
    const size_type cgk_l,
    const size_type num_dict,
    const vector<size_type >& signatures)
    :
    num_cgk_(num_cgk), cgk_l_(cgk_l) {

    // construction
    cgk_embed_.reserve(num_cgk);
    for (int i = 0; i < num_cgk; ++i) {
      cgk_embed_.emplace_back(cgk_l, num_dict, signatures);
    }
  }

  string embed(const string& x) const {
    string res;
    for (auto cgk_embed : cgk_embed_) {
      res += cgk_embed.embed(x);
    }
    return res;
  }

  void add(const vector<string>& xs) {
    assert(embed_string_.empty());
    embed_string_ = vector<string >(xs.size(), "");
#pragma omp parallel for
    for (int i = 0; i < xs.size(); ++i) {
      embed_string_[i] = embed(xs[i]);
    }
  }

  vector<int > query(const string& x) const {
    vector<int > dist(embed_string_.size(), 0);
    string cgk_x = embed(x);
#pragma omp parallel for
    for (int i = 0; i < dist.size(); ++i) {
      dist[i] = hamming_dist(cgk_x, embed_string_[i]);
    }
    return dist;
  }

 private:
  const size_type num_cgk_;
  const size_type cgk_l_;
  vector<string > embed_string_;
  vector<CGKEmbed > cgk_embed_;
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

  vector<string > hash(const string& x) const {
    string cgk = cgk_embed_.embed(x);
    vector<string > res(num_hash_, string(num_bits_, '\0'));
    for (int i = 0; i < num_hash_; ++i) {
      for (int j = 0; j < num_bits_; ++j) {
        res[i][j] = cgk[hash_lsh_[i][j]];
      }
    }
    return res;
  }

  unordered_set<size_type > query(const string& x) const {
    unordered_set<size_type > res;
    vector<string> hash_val = hash(x);

    for (int i = 0; i < num_hash_; ++i) {
      auto it = index_[i].find(hash_val[i]);
      if (it != index_[i].end()) {
        const vector<size_type >& candidates = it->second;
        std::copy(candidates.begin(), candidates.end(), std::inserter(res, res.end()));
      }
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
#pragma omp parallel for
    for (int k = 0; k < xs.size(); ++k) {
      hash_vals[k] = hash(xs[k]);
    }
#pragma omp parallel for
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
