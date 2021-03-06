#include <algorithm>
#include <cnpy.h>

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/ProductQuantizer.h>

#include "simd.h"
#include "utils.h"
#include "cgk_embed.h"




using namespace std;
using namespace faiss;


std::pair<double, double>
search(
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const vector<string >& query_modified,
  const vector<string >& base_modified,
  const vector<vector<size_t > >& q_knn,
  const float* xq,
  const float* xb,
  const size_t d,
  const int64_t threshold,
  const float threshold_l2) {

  size_t nq = query_strings.size();
  size_t nb = base_strings.size();


  timer time_recorder;
  vector<vector<size_t > > res(nq);

#pragma omp parallel for
  for (size_t i = 0; i < nq; i++) {
    /* query preparation for asymmetric search: compute look-up tables */
    for (size_t j = 0; j < nb; j++) {
      float dis = fvec_L2sqr(&xq[i * d], &xb[j*d], d);

      if (dis < threshold_l2) {
        if (-1 != edit_distance(
          base_modified[j].c_str(),
          base_strings[j].size(),
          query_modified[i].c_str(),
          query_strings[i].size(),
          threshold)) {
          res[i].push_back(j);
        }
      }
    }
  } // end of for loop

  double search_time =  time_recorder.elapsed();
  double recall = 0.0;
  double size_q_knn = 0;
  for (int k = 0; k < q_knn.size(); ++k) {
    if (!q_knn[k].empty()) {
      size_q_knn+=1.0;
      recall += res[k].size() * 1.0 / q_knn[k].size();
    }
  }
  return {recall / size_q_knn, search_time};
}

void embed_rank(
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const vector<string >& query_modified,
  const vector<string >& base_modified,
  const vector<vector<size_t > >& q_knn,
  const float* xb,
  const float* xq,
  const size_t d,
  const int64_t threshold,
  const float threshold_l2,
  const float step) {

  cout << "searching " << endl;
  for (int i = 0; i < 10; i++) {
    float th_i = threshold_l2 + i * step;
    auto recall_time = search(query_strings, base_strings,
                              query_modified, base_modified,
                              q_knn, xq, xb, d, threshold, th_i);
    std::cout
      << th_i << "\t"
      << recall_time.second << "\t"
      << recall_time.first
      << std::endl;
  }
}

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
  if (argc < 8) {
    fprintf(stderr, "usage: ./bin threshold threshold_l2 base_embedding query_embedding "
                    "base_location query_location ground_truth step\n");
    return 0;
  }

  int64_t threshold = (size_type)atoi(argv[1]);
  float threshold_l2 = atof(argv[2]);
  string base_embedding = argv[3];
  string query_embedding = argv[4];
  string base_location = argv[5];
  string query_location = argv[6];
  string ground_truth = argv[7];
  float step = 0.01f;
  if (argc > 8) {
    step = (float)atof(argv[8]);
  }


  size_type num_dict = 0;
  size_type nb = 0;
  size_type nq = 0;

  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);;
  vector<vector<size_type > > hash_lsh;

  cout << "loading base  data " << base_location  << endl;
  cout << "loading query data " << query_location << endl;
  load_data(base_location, base_strings, signatures, num_dict, nb);
  cout << "loaded base data, num dict " << num_dict << " nb: " << nb << endl;
  load_data(query_location, query_strings, signatures, num_dict, nq);
  cout << "loaded query data, num dict " << num_dict << " nq: " << nq << endl;


  vector<string > base_modified = base_strings;
  for (int j = 0; j < base_modified.size(); j++){
    for(int k = 0;k < 8; k++) base_modified[j].push_back(j>>(8*k));
  }
  vector<string > query_modified = query_strings;
  for (int j = 0; j < query_modified.size(); j++){
    for(int k = 0;k < 8; k++) query_modified[j].push_back(j>>(8*k));
  }

  vector<vector<size_t > > q_knn(nq);
  {
    cout << "loading ground_truth ";
    cnpy::NpyArray np_gt = cnpy::npy_load(ground_truth);
    cout <<  np_gt.shape[0] << "x" << np_gt.shape[1] << endl;
    const int64_t* qd = np_gt.data<int64_t >();
    size_t nd = np_gt.shape[1];
    if (np_gt.fortran_order) {
#pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        q_knn[i] = where_smaller(&qd[i], std::min(nd, (size_t)nb), nq, threshold);
      }
    } else {
#pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        q_knn[i] = where_smaller(&qd[i*nd], std::min(nd, (size_t)nb), 0, threshold);
      }
    }
  }

  cout << "loading base_embedding ";
  cnpy::NpyArray np_xb = cnpy::npy_load(base_embedding);
  cout << np_xb.shape[0] << "x" << np_xb.shape[1] << endl;

  cout << "loading query_embedding " ;
  cnpy::NpyArray np_xq = cnpy::npy_load(query_embedding);
  cout << np_xq.shape[0] << "x" << np_xq.shape[1] << endl;
  assert(np_xb.shape[1] == np_xq.shape[1]);

  const float* xb = np_xb.data<float >();
  const float* xq = np_xq.data<float >();
  const size_t d = np_xb.shape[1];

  embed_rank( query_strings, base_strings,
              query_modified, base_modified,
              q_knn, xb, xq, d, threshold, threshold_l2, step);
  return 0;
}
