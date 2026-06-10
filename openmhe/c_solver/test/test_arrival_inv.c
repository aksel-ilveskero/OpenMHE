/**
 * Unit tests for ``openmhe_invert_arrival_covariance`` and
 * ``openmhe_arrival_weight_block``.  Reference values come from Python
 * ``openmhe.invert_arrival_covariance`` (not plain ``numpy.linalg.inv`` when
 * null-space directions are present).
 */
#include <stdio.h>
#include <string.h>

#include "../filter_arrival.h"

static int approx_eq(double a, double b, double tol)
{
    if (a > b) {
        return (a - b) <= tol;
    }
    return (b - a) <= tol;
}

static int test_dense_3x3(void)
{
    /* Reference from Python invert_arrival_covariance on a dense SPD matrix. */
    const double P[] = {4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0};
    const double ref[] = {
        0.2738095238095239, -0.09523809523809548, 0.02380952380952398,
        -0.09523809523809545, 0.3809523809523815, -0.09523809523809548,
        0.02380952380952398, -0.09523809523809545, 0.5238095238095235};
    double W[9];
    openmhe_invert_arrival_covariance(
        P, 3, W, OPENMHE_ARRIVAL_INV_TOL, OPENMHE_ARRIVAL_MAX_WEIGHT);
    for (int i = 0; i < 9; ++i) {
        if (!approx_eq(W[i], ref[i], 1e-10)) {
            fprintf(stderr, "dense mismatch at %d: got %g want %g\n", i, W[i], ref[i]);
            return 1;
        }
    }
    return 0;
}

static int test_sparse_diagonal(void)
{
    const double P[] = {0.01, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    const double ref[] = {100.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double W[16];
    openmhe_invert_arrival_covariance(
        P, 4, W, OPENMHE_ARRIVAL_INV_TOL, OPENMHE_ARRIVAL_MAX_WEIGHT);
    for (int i = 0; i < 16; ++i) {
        if (!approx_eq(W[i], ref[i], 1e-8)) {
            fprintf(stderr, "sparse mismatch at %d: got %g want %g\n", i, W[i], ref[i]);
            return 1;
        }
    }
    return 0;
}

static int test_weight_block_submatrix(void)
{
    const double P_full[] = {0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02};
    const int idx[] = {0, 2};
    const double ref[] = {100.0, 0.0, 0.0, 50.0};
    double W[4];
    openmhe_arrival_weight_block(P_full, 3, idx, 2, W);
    for (int i = 0; i < 4; ++i) {
        if (!approx_eq(W[i], ref[i], 1e-8)) {
            fprintf(stderr, "block mismatch at %d: got %g want %g\n", i, W[i], ref[i]);
            return 1;
        }
    }
    return 0;
}

int main(void)
{
    if (test_dense_3x3() != 0) {
        return 1;
    }
    if (test_sparse_diagonal() != 0) {
        return 1;
    }
    if (test_weight_block_submatrix() != 0) {
        return 1;
    }
    printf("arrival_inv ok\n");
    return 0;
}
