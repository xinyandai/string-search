#include <cnpy.h>
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

/***
 * \str_location: a set of strings in a file
 * \num_cgk: number of CGK-embedding for each input string
 * \num_hash: number of hash functions for each embedded string
 * \num_bits: number of bits in each hash function
 * \max_l: the length of truncation, recommended to be the average length of strings
 * @return
 */
int main(int argc, char **argv) {
  if (argc == 1) {
    fprintf(stderr, "usage: ./bin max_l num_cgk num_hash num_bits base_location query_location \n");
    return 0;
  }

  auto max_l = (size_type)atoi(argv[1]);
  auto num_cgk = (size_type)atoi(argv[2]);
  auto num_hash = (size_type)atoi(argv[3]);
  auto num_bits = (size_type)atoi(argv[4]);

  string base_location = argv[5];
  string query_location = argv[6];
  string ground_truth = argv[7];

  size_type cgk_l = 3 * max_l;
  size_type num_dict = 0;
  size_type num_base = 0;
  size_type num_query = 0;

  vector<string > base_strings;
  vector<string > query_strings;
  vector<size_type > signatures(256, 1024);;
  vector<vector<size_type > > hash_lsh;

  cout << "loading base and query data " << endl;
  load_data(base_location, base_strings, signatures, num_dict, num_base);
  cout << "loaded base data, num dict " << num_dict << endl;
  load_data(query_location, query_strings, signatures, num_dict, num_query);
  cout << "loaded query data, num dict " << num_dict << endl;

  size_type nb  = base_strings.size();
  size_type nq  = query_strings.size();

  cout << "construction tables" << endl;
  vector<StringLSH > tables;
  for (int i = 0; i < num_cgk; ++i) {
    tables.emplace_back(num_hash,num_bits,cgk_l,num_dict,signatures);
  }

  cout << "add base items to tables" << endl;
#pragma omp parallel for
  for (int i = 0; i < tables.size(); i++) {
    tables[i].add(base_strings);
  }

  cout << "query" << endl;
  vector<unordered_set<size_type > > res(
    query_strings.size(), unordered_set<size_type >());
#pragma omp parallel for
  for (int i = 0; i < query_strings.size(); ++i) {
    for (int k = 0; k < tables.size(); ++k) {
      unordered_set<size_type > candidates = tables[k].query(query_strings[i]);
      std::copy(candidates.begin(), candidates.end(), std::inserter(res[i], res[i].end()));
    }
  }

  cnpy::NpyArray load_np = cnpy::npy_load(ground_truth);
  const int* ed = load_np.data<int >();

  size_type topk = 10;

  double recall = 0;
#pragma omp parallel for reduction(+ : recall)
  for (int i = 0; i < nq; ++i) {
    for (int j = 0; j < topk; ++j) {
      int id = ed[i * nb + j];
      if (res[i].find(id) != res[i].end()) {
        recall+=1;
      }
    }
  }
  std::cout << "recall : " << recall / res.size() / topk << endl;

  return 0;
}
