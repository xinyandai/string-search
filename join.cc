//
// Created by xinyan on 9/1/2020.
//
#include <algorithm>
#include <cnpy.h>

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFFlat.h>

#include "utils.h"
#include "simd.h"
#include "cgk_embed.h"

using namespace std;
using std::ofstream;

/***
 * \cgk_l: the length of truncation, recommended to be the average length of strings
 * \num_cgk: number of CGK-embedding for each input string
 * \num_hash: number of hash functions for each embedded string
 * \num_bits: number of bits in each hash function
 * \base_location: a set of strings in a file
 * \query_location: a set of strings in a file
 * \ground_truth: knn neighbors in {base} for each {query}
 * @return
 */
int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: ./bin string embedding threshold threshold(l2)\n");
    return 0;
  }

  int i = 0;
  string base_location = argv[++i];
  string embed_location = argv[++i];
  auto threshold_ed = atoi(argv[++i]);
  auto threshold_l2 = atof(argv[++i]);
  string bench_file = argv[++i];
  size_type max_nb = INT_MAX;
  if (argc > ++i) {
    max_nb = (size_type) atoi(argv[i]);
  }
  cout << "threshold_ed " << threshold_ed << endl;
  cout << "threshold_l2 " << threshold_l2 << endl;
  cout << "max_nb " << max_nb << endl;

  size_type num_dict = 0;
  size_type num_base = 0;
  vector<string > base_strings;
  vector<size_type > signatures(256, 1024);;

  cout << "loading base data " << endl;
  load_data(base_location, base_strings, signatures, num_dict, num_base);
  cout << "loaded base data, num dict " << num_dict
       << " nb: " << base_strings.size() << endl;
  auto nb  = (size_type)base_strings.size();
  if (max_nb < nb) {
    base_strings.resize(max_nb);
    nb = max_nb;
    cout << "loaded base data, num dict " << num_dict
         << " nb: " << base_strings.size() << endl;
  }
  vector<string > oridata_modified = base_strings;
  for (int j = 0; j < oridata_modified.size(); j++){
    for(int k = 0;k < 8; k++) oridata_modified[j].push_back(j>>(8*k));
  }

  cout << "loading base_embedding ";
  cnpy::NpyArray np_xb = cnpy::npy_load(embed_location);
  cout << np_xb.shape[0] << "x" << np_xb.shape[1] << endl;

  const float* xb = np_xb.data<float >();
  FINTEGER M = np_xb.shape[0], N = np_xb.shape[0], D = np_xb.shape[1];
  assert(M == nb);
  cout << "Allocating " << M  << "X" << M  << endl;
  float * out = new float[1LL * M * M];
  float alpha = -2.0f, beta = 0.0f;
  FINTEGER LDA = D;
  FINTEGER LDB = D;
  FINTEGER LDC = M;
  cout << "Allocating " << "Allocating result queue" << endl;
  vector<pair<int, int > > res;
  res.reserve(1024);

  timer timer_recorder;
  cout << "calling sgemm " << endl;
  sgemm_("T", "N", &M, &N, &D,
         &alpha, xb, &LDA, xb, &LDB,
         &beta, out, &LDC);
  cout << "calling sqr " << endl;
  float *nsqr = new float[nb];
  for (int i = 0; i < M; ++i) {
    nsqr[i] = fvec_norm_L2sqr(&xb[i * D], D);
  }

  cout << "adding sqr " << endl;
  size_t can = 0;
  for (int i = 0; i < M; ++i) {
    for (int j = i + 1; j < M; ++j) {
      if (out[i * M + j] + nsqr[i] + nsqr[j] <= threshold_l2) {
        int ed = edit_distance(
          oridata_modified[j].data(),
          base_strings[j].size(),
          oridata_modified[i].data(),
          base_strings[i].size(), threshold_ed);

        if (ed != -1 )
          res.emplace_back(i, j);
	    can++;
      }
    }
  }
  cout << "Time  " << timer_recorder.elapsed() << endl;
  cout << "size  " << can << endl;
  cout << "size  " << res.size() << endl;

  ofstream output_sream(bench_file);
  for (auto& p : res) {
    output_sream << p.first << "\t" << p.second << endl;
  }
  output_sream.close();

  delete [] out;
  delete [] nsqr;
  return 0;
}
