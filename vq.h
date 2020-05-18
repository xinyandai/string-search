//
// Created by xydai on 2020/5/18.
//

#ifndef HNSW_LIB_UTIL_H
#define HNSW_LIB_UTIL_H

#include <cstdint>
#include <stdexcept>
#include "simd.h"


int vq(const float *centroids, const float *x, int ks, int dsub) {
    int id = 0;
    float min_dist = fvec_L2sqr(centroids, x, dsub);

    for (int i = 1; i < ks; ++i) {
        centroids += dsub;
        float dist = fvec_L2sqr(centroids, x, dsub);
        if (dist < min_dist) {
            id = i;
        }
    }
    return id;
}

/**
 * @param centroids M * ks * dsub
 * @param codes nx * M
 * @return
 */
void vq(uint8_t* codes, const float *centroids, const float *x,
        const int ks, const int nx, const int d, const int M) {
    if (d % M != 0) {
        throw std::runtime_error("d is not dividable by M.");
    }
    const int dsub = d / M;

#pragma omp parallel for
    for (int i = 0; i < nx; ++i) {
        for (int m = 0; m < M; ++m) {
            codes[i * M + m] = vq(centroids + m * ks * dsub,
                                  x + i * d + m * dsub, ks, dsub);
        }
    }
}

/**
 * @param dt  M * ks
 */
void compute_distance_table(float* dt, const float *centroids,
                            const float *x, int ks, int d, int M) {
    if (d % M != 0) {
        throw std::runtime_error("d is not dividable by M.");
    }
    int dsub = d / M;
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < ks; ++k) {
            dt[m * ks + k] = fvec_L2sqr(x + m * dsub,
                                        centroids + m * ks * dsub + k * dsub, dsub);
        }
    }
}

#endif //HNSW_LIB_UTIL_H
