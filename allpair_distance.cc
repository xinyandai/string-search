//
// Created by xinyan on 12/1/2020.
//

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

  const float threshold_l2 = 0.001;
  const string embed_location_x = argv[0];
  const string embed_location_q = argv[1];

  cout << "loading query_embedding ";
  cnpy::NpyArray np_xq = cnpy::npy_load(embed_location_q);
  cout << np_xq.shape[0] << "x" << np_xq.shape[1] << endl;
  const float* xq = np_xq.data<float >();

  cout << "loading base_embedding ";
  cnpy::NpyArray np_xb = cnpy::npy_load(embed_location_x);
  cout << np_xb.shape[0] << "x" << np_xb.shape[1] << endl;
  const float* xb = np_xb.data<float >();

  size_t nb = std::min(np_xb.shape[0], (size_t)50000);
  size_t nq = np_xq.shape[0];
  size_t D = np_xq.shape[1];

  timer timer_recorder;

  vector<float> xb_nsqr(nb, .0);
  vector<float> xq_nsqr(nq, .0);
  for (int i = 0; i < nb; ++i) {
    xb_nsqr[i] = fvec_norm_L2sqr(&xb[i * D], D);
  }
  for (int i = 0; i < nq; ++i) {
    xq_nsqr[i] = fvec_norm_L2sqr(&xq[i * D], D);
  }


  boost::progress_display progress(nb);
  size_t can = 0;
//#pragma omp parallel for
  for (size_type i = 0; i < nq; ++i) {
//#pragma omp critical
    {
      ++progress;
    }
    for (int j = 0; j < nb; ++j) {
      if (-2.0f * fvec_inner_product(&xb[i * D], &xb[j * D], D) + xq_nsqr[i] + xb_nsqr[j] <= threshold_l2) {
        can++;
      }
    }
  }

  cout << "Time  " << timer_recorder.elapsed() << endl;
  return 0;
}
