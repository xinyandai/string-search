#include <algorithm>
#include <cnpy.h>
#include <boost/progress.hpp>

#include "utils.h"
#include "cgk_embed.h"

using namespace std;
using size_type = unsigned;


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


void cgk_rank(
  const vector<string >& query_strings,
  const vector<string >& base_strings,
  const size_type  num_cgk,
  const size_type  num_hash,
  const size_type  num_bits,
  const size_type  cgk_l,
  const size_type  num_dict,
  const vector<size_type > & signatures,
  const int* ed) {

  auto nb  = (size_type)base_strings.size();
  auto nq  = (size_type)query_strings.size();

  CGKRanker ranker(num_bits, num_cgk, cgk_l,  num_dict, signatures);

  cout << "add base items to tables" << endl;
  ranker.add(base_strings);


  vector<vector<size_t > > idx(nq, vector<size_t >());
  {
    // dist should be cleared after sorting
    vector<vector<int> > dist(nq, vector<int>());
    cout << "compute hamming distance" << endl;
    boost::progress_display progress_dist(nq);
#pragma omp parallel for
    for (int i = 0; i < nq; ++i) {
#pragma omp critical
      {
        ++progress_dist;
      }
      dist[i] = ranker.query(query_strings[i]);
    }

    cout << "rank by hamming distance" << endl;
    boost::progress_display progress_sort(nq);

#pragma omp parallel for
    for (int i = 0; i < nq; ++i) {

#pragma omp critical
      {
        ++progress_sort;
      }
      idx[i] = arg_sort(dist[i]);
    }
  }

  cout << "compute recalls" << endl;
  vector<size_type > top_k = {1, 10, 20, 50, 100, 1000};
  vector<size_type > probed;
  size_type num_prob = 20;
  probed.reserve(num_prob);
  for (int i = 0; i < num_prob; ++i) {
    probed.push_back((size_type)1 << i);
  }

  for (int j = 0; j < probed.size(); ++j) {
    vector<double > recalls(top_k.size(), 0);

#pragma omp parallel for
    for (int k = 0; k < top_k.size(); ++k) {
      for (int i = 0; i < nq; ++i) {

        recalls[k] += std::count_if(
          idx[i].begin(), probed[j] < nb ? idx[i].begin() + probed[j] : idx[i].end(),
          [nb, &top_k, ed, i, k] (const size_t id) {
            for (int l = 0; l < top_k[k]; ++l) {
              if (id == ed[i * nb + l])
                return true;
            }
            return false;
          });
      }
      recalls[k] /= nq * top_k[k];
    }

    std::cout << probed[j];
    for (int k = 0; k < top_k.size(); ++k) {
      std::cout << "\t" << recalls[k];
    }
    std::cout << std::endl;
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
  if (argc != 8) {
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

  cnpy::NpyArray load_np = cnpy::npy_load(ground_truth);
  const int* ed = load_np.data<int >();

  cgk_rank(query_strings, base_strings,
      num_cgk, num_hash, num_bits, cgk_l, num_dict, signatures, ed);

  return 0;
}
