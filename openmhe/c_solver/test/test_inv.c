#include <stdio.h>
#include <string.h>

#include "../filter_arrival.h"

int main(void)
{
    const int n = 3;
    double P[] = {4.0, 1.0, 0.0, 1.0, 3.0, 0.5, 0.0, 0.5, 2.0};
    double P_inv[9];
    double I[9];

    if (openmhe_symmetric_inv(P, n, P_inv) != 0) {
        fprintf(stderr, "openmhe_symmetric_inv failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < n; ++k) {
                s += P[i * n + k] * P_inv[k * n + j];
            }
            I[i * n + j] = s;
        }
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            const double want = (i == j) ? 1.0 : 0.0;
            if (I[i * n + j] - want > 1e-8 || want - I[i * n + j] > 1e-8) {
                fprintf(stderr, "P*P_inv mismatch at (%d,%d) = %g\n", i, j, I[i * n + j]);
                return 1;
            }
        }
    }

    printf("symmetric_inv ok\n");
    return 0;
}
