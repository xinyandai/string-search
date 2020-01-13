#pragma once

#include <algorithm>
#include <utility>
#include <vector>

using std::vector;
using std::pair;

template<typename DataT, typename DistType>
class DistanceElement {
 protected:
  DistType    dist_;
  DataT       data_;

 public:
  DistanceElement(DistType dist, const DataT& data)
  : dist_(dist), data_(data) {}

  explicit DistanceElement(const pair<DistType, DataT>& p)
  : dist_(p.first), data_(p.second) {}

  DistType dist() const {
      return dist_;
  }

  const DataT& data() const {
      return data_;
  }
};


// min heap element
template<typename DataT, typename DistType>
class MinHeapElement : public DistanceElement<DataT, DistType> {
 public:

  MinHeapElement(DistType dist, const DataT& data)
  : DistanceElement<DataT, DistType>(dist, data) {}

  bool operator<(const MinHeapElement& other) const  {
      return this->dist_ > other.dist_;
  }
};


/// max heap element
template<typename DataT, typename DistType=float>
class MaxHeapElement : public DistanceElement<DataT, DistType> {
 public:

  MaxHeapElement(DistType dist, const DataT& data)
  : DistanceElement<DataT, DistType>(dist, data) {}

  bool operator<(const MaxHeapElement& other) const  {
      return this->dist_ < other.dist_;
  }
};


template <
  typename DataT,
  typename DistType,
  typename HeapElement=MaxHeapElement<DataT, DistType>
>
class Heap {
 private:
  int                  K_;
  vector<HeapElement > heap_;
 public:
  explicit Heap(int K) {
      K_ = K;
  }

  bool insert(const HeapElement& pair) {
    if (heap_.size() < K_) {
      heap_.push_back(pair);
      std::push_heap(heap_.begin(), heap_.end());
      return true;
    } else {
      if (pair < heap_[0]) {
        std::pop_heap(heap_.begin(), heap_.end());
        heap_[K_-1] =  pair;/// pop the max one and swap it
        std::push_heap(heap_.begin(), heap_.end());
        return true;
      }
    }
    return false;
  }


  bool insert(DistType dist, DataT data) {
    return insert(HeapElement(dist, data));
  }

  const HeapElement& top() const {
      return heap_[0];
  }

  const size_t size() const {
    return heap_.size();
  }

  vector<HeapElement> topk() const {
      vector<HeapElement > heap = heap_;
      std::sort(heap.begin(), heap.end());
      return heap;
  }
};

