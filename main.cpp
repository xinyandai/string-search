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
  1, 10, 20, 50, 100, 1000
};
const vector<size_type > probed = {
  1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,
  2048, 4096, 8192, 16384, 32768, 65536
};

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

}


void embed_rank(
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const float* xb, const float* xq,
  const size_t d, const int* ed, const bool re_rank) {

  auto nb  = (size_type)base_strings.size();
  auto nq  = (size_type)query_strings.size();
  auto nt = std::min((size_type)10000, nb);

  size_t n_list = 100;
  faiss::IndexFlatL2 q(d);
  size_t m = 8;
  assert (d % m == 0);
  faiss::IndexIVFPQ index(&q, d, n_list, m, 8);
  timer time_recorder;

  cout << "training index" << endl;
  index.train(nt, xb);
  cout << "add base to index" << endl;
  index.add(nb, xb);
  cout << "done adding items" << endl;
  for (int j = 0; j < probed.size(); ++j) {
    const size_type num_prob = probed[j];

    long *I = new long[num_prob * nq];
    float *D = new float[num_prob * nq];


    index.nprobe = num_prob;
    time_recorder.restart();
    index.search(nq, xq, num_prob, D, I);
    double embed_rank_time = time_recorder.elapsed();

    vector<double > recalls(top_k.size(), 0);
    double re_rank_time = 0.0;

    if (re_rank) {
      time_recorder.restart();
      vector<vector<size_t > > re_rank_idx(nq);

#pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        re_rank_idx[i] = ed_rank(&I[i * num_prob], num_prob, query_strings[i], base_strings);
      }
      re_rank_time = time_recorder.elapsed();

#pragma omp parallel for
      for (int k = 0; k < top_k.size(); ++k) {
        for (int i = 0; i < nq; ++i) {
          recalls[k] += intersection_size(
            re_rank_idx[i].begin(), re_rank_idx[i].end(),
            &ed[i * nb], &ed[i * nb + top_k[k]]);
        }
      } // end computing recalls of ks
    } // end if (re_rank)
    else {
#pragma omp parallel for
      for (int k = 0; k < top_k.size(); ++k) {
        for (int i = 0; i < nq; ++i) {
          recalls[k] += std::count_if(
            &I[i * num_prob], &I[(i + 1) * num_prob],
            [nb, ed, i, k] (const long id) {
              for (int l = 0; l < top_k[k]; ++l) {
                if (id == (size_t)ed[i * nb + l])
                  return true;
              }
              return false;
            });
        } // end of computing k recalls
      } // end of calling omp parallel
    } // end else [if (re_rank)]

    std::cout << probed[j] << "\t" << embed_rank_time << "\t" << re_rank_time ;
    for (int k = 0; k < top_k.size(); ++k) {
      std::cout << "\t" << recalls[k] / nq / top_k[k];
    }
    std::cout << std::endl;

    delete [] I;
    delete [] D;
  }
}


void cgk_rank(
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const size_type  num_cgk,
  const size_type  num_hash,
  const size_type  num_bits,
  const size_type  cgk_l,
  const size_type  num_dict,
  const vector<size_type > & signatures,
  const int* ed,
  bool re_rank) {

  auto nb  = (size_type)base_strings.size();
  auto nq  = (size_type)query_strings.size();

  CGKRanker ranker(num_bits, num_cgk, cgk_l,  num_dict, signatures);

  cout << "add base items to tables" << endl;
  timer time_recorder;
  ranker.add(base_strings);
  cout << "add time " << time_recorder.elapsed() << endl;

  time_recorder.restart();
  vector<vector<size_t > > idx(nq, vector<size_t >());
  {
    // dist should be cleared after sorting
    vector<vector<int> > dist(nq, vector<int>());
    cout << "compute hamming distance" << endl;
#pragma omp parallel for
    for (int i = 0; i < nq; ++i) {
      dist[i] = ranker.query(query_strings[i]);
    }


#pragma omp parallel for
    for (int i = 0; i < nq; ++i) {
      idx[i] = arg_sort(dist[i]);
    }
  }
  double embed_rank_time = time_recorder.elapsed();

  cout << "compute recalls" << endl;

  for (int j = 0; j < probed.size(); ++j) {
    const size_type num_prob = probed[j];

    vector<double > recalls(top_k.size(), 0);
    double re_rank_time = 0.0;
    if (re_rank) {
      time_recorder.restart();
      vector<vector<size_t > > re_rank_idx(nq);

#pragma omp parallel for
      for (int i = 0; i < nq; ++i) {
        re_rank_idx[i] = ed_rank(idx[i].data(), num_prob, query_strings[i], base_strings);
      }

      re_rank_time = time_recorder.elapsed();

#pragma omp parallel for
      for (int k = 0; k < top_k.size(); ++k) {
        for (int i = 0; i < nq; ++i) {
          recalls[k] += intersection_size(
            re_rank_idx[i].begin(), re_rank_idx[i].end(),
            &ed[i * nb], &ed[i * nb + top_k[k]]);
        }
      }
    }
    // end if (re_rank)
    else {

#pragma omp parallel for
      for (int k = 0; k < top_k.size(); ++k) {
        for (int i = 0; i < nq; ++i) {
          recalls[k] += std::count_if(
            idx[i].begin(), probed[j] < nb ? idx[i].begin() + probed[j] : idx[i].end(),
            [nb, ed, i, k] (const size_t id) {
              for (int l = 0; l < top_k[k]; ++l) {
                if (id == ed[i * nb + l])
                  return true;
              }
              return false;
            });
        }
      }

    }

    std::cout << probed[j] << "\t" << embed_rank_time << "\t" << re_rank_time ;
    for (int k = 0; k < top_k.size(); ++k) {
      std::cout << "\t" << recalls[k] / nq / top_k[k];
    }
    std::cout << std::endl;
  }
  // end else [if (re_rank)]

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
    fprintf(stderr, "usage: ./bin cgk_l num_cgk num_hash num_bits "
                    "base_location query_location ground_truth\n");
    return 0;
  }

  auto cgk_l = (size_type)atoi(argv[1]);
  auto num_cgk = (size_type)atoi(argv[2]);
  auto num_hash = (size_type)atoi(argv[3]);
  auto num_bits = (size_type)atoi(argv[4]);

  string base_location = argv[5];
  string query_location = argv[6];
  string ground_truth = argv[7];
  string base_embedding = argv[8];
  string query_embedding = argv[9];
  bool re_rank = (argc == 11);

  size_type num_dict = 0;
  size_type num_base = 0;
  size_type num_query = 0;

  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);;
  vector<vector<size_type > > hash_lsh;

  cout << "loading base and query data " << endl;
  load_data(base_location, base_strings, signatures, num_dict, num_base);
  cout << "loaded base data, num dict " << num_dict
       << " nb: " << base_strings.size() << endl;
  load_data(query_location, query_strings, signatures, num_dict, num_query);
  cout << "loaded query data, num dict " << num_dict
       << " nq: " << query_strings.size() << endl;

  cout << "loading ground_truth";
  cnpy::NpyArray np_gt = cnpy::npy_load(ground_truth);
  cout <<  np_gt.shape[0] << "x" << np_gt.shape[1] << endl;

  cout << "loading base_embedding" << endl;
  cnpy::NpyArray np_xb = cnpy::npy_load(base_embedding);
  cout << np_xb.shape[0] << "x" << np_xb.shape[1] << endl;
  cout << "loading query_embedding" ;
  cnpy::NpyArray np_xq = cnpy::npy_load(query_embedding);
  cout << np_xq.shape[0] << "x" << np_xq.shape[1] << endl;
  const int* ed = np_gt.data<int >();
  const float* xb = np_xb.data<float >();
  const float* xq = np_xq.data<float >();
  const size_t d = np_xb.shape[1];

  embed_rank(query_strings, base_strings,
             xb, xq, d, ed, re_rank);

  cgk_rank(query_strings, base_strings,
           num_cgk, num_hash, num_bits,
           cgk_l, num_dict, signatures, ed, re_rank);
  return 0;
}
