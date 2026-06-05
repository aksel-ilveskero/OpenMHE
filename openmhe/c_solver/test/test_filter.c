#include <stdio.h>
#include "filter_arrival.h"

int main(void)
{
    const int n = 3;
    const int ny = 2;
    const int nu = 2;
    double A[] = {0.9, 0.1, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.9};
    double B[] = {0.01, 0.0, 0.01, 0.0, 0.01, 0.0};
    double C[] = {1, 0, 0, 0, 1, 0};
    double D[] = {0, 0, 0, 0};
    double Q[] = {0.01, 0, 0, 0, 0.01, 0, 0, 0, 0.01};
    double R[] = {0.1, 0, 0, 0.1};
    openmhe_filter_config_t cfg = {
        OPENMHE_FILTER_UKF, n, ny, nu, 1e-3, 2.0, 0.0,
        A, B, C, D, Q, R};
    double x[3], P[9];
    double chi[21], chi_pred[21], gamma[14];
    openmhe_ukf_workspace_t ws = {chi, chi_pred, gamma};
    openmhe_filter_state_t st;
    openmhe_filter_init(&st, &cfg, x, P, &ws);
    double y0[] = {1.0, 0.5};
    double u0[] = {0.1, 0.2};
    double xb[3], Pp[9];
    openmhe_filter_prior(&st, u0, xb, Pp);
    printf("prior0:");
    for (int i = 0; i < n; ++i) {
        printf(" %g", xb[i]);
    }
    printf("\n");
    openmhe_filter_assimilate(&st, y0, u0);
    printf("post:");
    for (int i = 0; i < n; ++i) {
        printf(" %g", st.x[i]);
    }
    printf("\n");
    double u1[] = {0.2, 0.3};
    openmhe_filter_prior(&st, u1, xb, Pp);
    printf("prior1:");
    for (int i = 0; i < n; ++i) {
        printf(" %g", xb[i]);
    }
    printf("\n");
    return 0;
}
