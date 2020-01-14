#include <algorithm>
#include <cnpy.h>

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFFlat.h>

#include "utils.h"
#include "cgk_embed.h"


using namespace std;
const vector<size_type > top_k = {
  1, 10, 50, 100
};
const vector<size_type > probed = {
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
  2048, 4096, 8192, 16384, 32768, 65536
};


template <class INDEX_TYPE>
std::pair<double, double>
search(
  INDEX_TYPE& index,
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const vector<string >& query_modified,
  const vector<string >& base_modified,
  const vector<vector<size_t > >& q_knn,
  const float* xq,
  size_t num_prob,
  size_t topk,
  size_t alpha) {

  size_t nq = query_strings.size();
  long *I = new long[num_prob * nq];
  float *D = new float[num_prob * nq];

  if constexpr (!std::is_same<faiss::IndexFlatL2, INDEX_TYPE>::value) {
    index.nprobe = num_prob * alpha;
  }
  timer time_recorder;
  index.search(nq, xq, num_prob, D, I);

  vector<vector<size_t > > re_rank_idx(nq);
#pragma omp parallel for
  for (int i = 0; i < nq; ++i) {
    re_rank_idx[i] = ed_rank(
      &I[i * num_prob], num_prob, topk,
      query_modified[i], query_strings[i].size(),
      base_strings, base_modified);
  }

  double search_time =  time_recorder.elapsed();
  double recall = calculate_recalls(q_knn, re_rank_idx, nq, topk);
  delete [] I;
  delete [] D;
  return {recall, search_time};
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
  size_t alpha) {

  auto nb = (size_type)base_strings.size();
  auto nt = std::min(nb, (size_type )50000);

  faiss::IndexFlatL2 index(d);
  /*
  size_t n_list = 1000;
  faiss::IndexFlatL2 q(d);
  size_t m = 16;
  assert (d % m == 0);
  faiss::IndexIVFPQ index(&q, d, n_list, m, 8);
  */
  /*
  size_t nhash = 2;
  size_t nbits_subq = int (log2 (nb+1) / 2);        // good choice in general
  size_t ncentroids = 1 << (nhash * nbits_subq);  // total # of centroids

  faiss::MultiIndexQuantizer coarse_quantizer (d, nhash, nbits_subq);
  faiss::MetricType metric = faiss::METRIC_L2; // can be METRIC_INNER_PRODUCT
  faiss::IndexIVFFlat index (&coarse_quantizer, d, ncentroids, metric);
  index.quantizer_trains_alone = true;
  */
  cout << "training index" << endl;
  index.train(nt, xb);
  cout << "add base to index" << endl;
  index.add(nb, xb);
  cout << "done adding items" << endl;

  for (size_t k  : top_k) {

    for (size_t i = 0; i < probed.size(); i++) {
      auto recall_time = search(index,
                                query_strings, base_strings,
                                query_modified, base_modified,
                                q_knn, xq, probed[i], k, alpha);
      std::cout <<  k << "\t"
                << probed[i] << "\t"
                <<  recall_time.second << "\t"
                <<  recall_time.first  << std::endl;
    }
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
  if (argc < 6) {
    fprintf(stderr, "usage: ./bin base_embedding query_embedding  "
                    "base_location query_location ground_truth alpha\n");
    return 0;
  }


  string base_embedding = argv[1];
  string query_embedding = argv[2];
  string base_location = argv[3];
  string query_location = argv[4];
  string ground_truth = argv[5];
  int alpha = 2;
  if (argc > 6) {
    alpha = atoi(argv[6]);
  }

  size_type num_dict = 0;
  size_type nb = 0;
  size_type nq = 0;

  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);;
  vector<vector<size_type > > hash_lsh;

  cout << "loading base and query data " << endl;
  load_data(base_location, base_strings, signatures, num_dict, nb);
  cout << "loaded base data, num dict " << num_dict
       << " nb: " << base_strings.size() << endl;
  load_data(query_location, query_strings, signatures, num_dict, nq);
  cout << "loaded query data, num dict " << num_dict
       << " nq: " << query_strings.size() << endl;

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
        q_knn[i] = arg_sort(&qd[i], std::min(nd, (size_t)nb), nq);
      }
    } else {
#pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        q_knn[i] = arg_sort(&qd[i*nd], std::min(nd, (size_t)nb), 0);
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

  embed_rank(
    query_strings, base_strings,
    query_modified, base_modified,
    q_knn,
    xb, xq, d, alpha);
  return 0;
}
