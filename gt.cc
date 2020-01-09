//
// Created by xinyan on 6/1/2020.
//

#include <algorithm>
#include <cnpy.h>

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFFlat.h>
#include <boost/progress.hpp>

#include "utils.h"
#include "cgk_embed.h"


using namespace std;


void write_ivecs(const string& bench_file, const vector<vector<int > >& ed) {
  ofstream fout(bench_file, ios::binary);
  if (!fout) {
    cout << "cannot create output file " << bench_file << endl;
    assert(false);
  }
  int D = (int)ed[0].size();
  for (const vector<int >& line : ed) {
    assert(line.size() == D);
    fout.write((char*)&D, sizeof(int));
    fout.write((char*)line.data(), sizeof(int) * D);
  }
  fout.close();
  cout << "ivecs groundtruth are written into " << bench_file << endl;
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
  if (argc < 4) {
    fprintf(stderr, "usage: ./bin query_location base_location ground_truth max_nb\n");
    return 0;
  }

  string query_location = argv[1];
  string base_location = argv[2];
  string ground_truth = argv[3];
  size_type max_nb = INT_MAX;
  if (argc > 4) {
    max_nb = (size_type) atoi(argv[4]);
  }

  size_type num_dict = 0;
  size_type num_base = 0;
  size_type num_query = 0;
  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);;

  cout << "loading base and query data " << endl;
  load_data(base_location, base_strings, signatures, num_dict, num_base);
  cout << "loaded base data, num dict " << num_dict
       << " nb: " << base_strings.size() << endl;
  load_data(query_location, query_strings, signatures, num_dict, num_query);
  cout << "loaded query data, num dict " << num_dict
       << " nq: " << query_strings.size() << endl;

  auto nq  = (size_type)query_strings.size();
  auto nb  = (size_type)base_strings.size();
  if (max_nb < nb) {
    base_strings.resize(max_nb);
    nb = max_nb;
    cout << "loaded base data, num dict " << num_dict
         << " nb: " << base_strings.size() << endl;
  }

  if (max_nb < nq) {
    query_strings.resize(max_nb);
    nq = max_nb;
    cout << "loaded query data, num dict " << num_dict
         << " nq: " << query_strings.size() << endl;
  }

  vector<vector<int > > ed(nq, vector<int > (nb, 0));
  boost::progress_display progress(nb);

#pragma omp parallel for
  for (int j = 0; j < nb; ++j) {

#pragma omp critical
    {
      ++progress;
    }


    for (int i = 0; i < nq; ++i) {
      ed[i][j] = edit_distance(query_strings[i], base_strings[j]);
    }
  }

  write_ivecs(ground_truth, ed);


  return 0;
}
