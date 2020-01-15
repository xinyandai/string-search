//
// Created by xinyan on 12/1/2020.
//
#include <omp.h>
#include <boost/progress.hpp>

#include "utils.h"
#include "heap.h"


int main(int argc, char **argv) {
  string query_location = argv[1];
  string base_location = argv[2];
  int topk = atoi(argv[3]);
  string bench_file = argv[4];

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

  vector<string > base_modified = base_strings;
#pragma omp parallel for
  for (int j = 0; j < base_modified.size(); j++){
    for(int k = 0;k < 8; k++) base_modified[j].push_back(j>>(8*k));
  }
  vector<string > query_modified = query_strings;
#pragma omp parallel for
  for (int j = 0; j < query_modified.size(); j++){
    for(int k = 0;k < 8; k++) query_modified[j].push_back(j>>(8*k));
  }

  timer timer_recorder;
  vector<vector<size_t > > res(num_query, vector<size_t >());
  vector<size_t> idx(num_base);
  iota(idx.begin(), idx.end(), 0);

  boost::progress_display progress(num_query);
#pragma omp parallel for
  for (size_t i = 0; i < num_query; ++i) {
#pragma omp critical
    {
      ++progress;
    }

    res[i] = ed_rank(
      idx.data(), num_base,
      topk, query_modified[i],
      query_strings[i].size(),
      base_strings, base_modified);
  }

  std::cout <<  "# elapsed time \t"
            << timer_recorder.elapsed()
            << std::endl;

  ofstream output_stream(bench_file);
  for (auto& ids : res) {
    for (auto& id : ids) {
      output_stream << id << "\t";
    }
    output_stream << endl;
  }
  output_stream.close();

  return 0;
}
