#include <math.h>
#include <string.h>

#include "filter_arrival.h"

#define CHOL_REG 1e-9

int openmhe_symmetric_inv(const double *P, int n, double *P_inv);

static void mat_vec(
    const double *A, int m, int n, const double *x, double *y)
{
    for (int i = 0; i < m; ++i) {
        double s = 0.0;
        for (int j = 0; j < n; ++j) {
            s += A[i * n + j] * x[j];
        }
        y[i] = s;
    }
}

static void outer_add(double *P, int n, const double *a, double w)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            P[i * n + j] += w * a[i] * a[j];
        }
    }
}

static void subtract_kpk(double *P, int n, const double *K, const double *P_yy, int ny)
{
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double s = 0.0;
            for (int k = 0; k < ny; ++k) {
                for (int l = 0; l < ny; ++l) {
                    s += K[i * ny + k] * P_yy[k * ny + l] * K[j * ny + l];
                }
            }
            P[i * n + j] -= s;
        }
    }
}

static int cholesky_lower(const double *A, int n, double *L)
{
    for (int i = 0; i < n * n; ++i) {
        L[i] = 0.0;
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i * n + j];
            for (int k = 0; k < j; ++k) {
                s -= L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                if (s <= 0.0) {
                    return -1;
                }
                L[i * n + j] = sqrt(s);
            } else {
                L[i * n + j] = s / L[j * n + j];
            }
        }
    }
    return 0;
}

static void ekf_predict(
    const openmhe_filter_config_t *cfg, double *x, double *P, const double *u_t)
{
    const int n = cfg->nx;
    const int nu = cfg->nu;
    double x_new[n];
    double AP[n * n];
    double APAt[n * n];

    mat_vec(cfg->A, n, n, x, x_new);
    double Bu[n];
    mat_vec(cfg->B, n, nu, u_t, Bu);
    for (int i = 0; i < n; ++i) {
        x[i] = x_new[i] + Bu[i];
    }

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            AP[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                AP[i * n + j] += cfg->A[i * n + k] * P[k * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            APAt[i * n + j] = cfg->Q[i * n + j];
            for (int k = 0; k < n; ++k) {
                APAt[i * n + j] += AP[i * n + k] * cfg->A[j * n + k];
            }
        }
    }
    memcpy(P, APAt, (size_t)(n * n) * sizeof(double));
}

static void ekf_update(
    const openmhe_filter_config_t *cfg,
    double *x,
    double *P,
    const double *y_t,
    const double *u_t)
{
    const int n = cfg->nx;
    const int ny = cfg->ny;
    const int nu = cfg->nu;

    double y_pred[ny];
    mat_vec(cfg->C, ny, n, x, y_pred);
    double Du[ny];
    mat_vec(cfg->D, ny, nu, u_t, Du);
    for (int j = 0; j < ny; ++j) {
        y_pred[j] += Du[j];
    }

    double CP[ny * n];
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < n; ++j) {
            CP[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                CP[i * n + j] += cfg->C[i * n + k] * P[k * n + j];
            }
        }
    }

    double S[ny * ny];
    memcpy(S, cfg->R, (size_t)(ny * ny) * sizeof(double));
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < ny; ++j) {
            for (int k = 0; k < n; ++k) {
                S[i * ny + j] += CP[i * n + k] * cfg->C[j * n + k];
            }
        }
    }

    double S_inv[ny * ny];
    if (openmhe_symmetric_inv(S, ny, S_inv) != 0) {
        return;
    }

    double PCT[n * ny];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ny; ++j) {
            PCT[i * ny + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                PCT[i * ny + j] += P[i * n + k] * cfg->C[j * n + k];
            }
        }
    }

    double K[n * ny];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ny; ++j) {
            K[i * ny + j] = 0.0;
            for (int k = 0; k < ny; ++k) {
                K[i * ny + j] += PCT[i * ny + k] * S_inv[k * ny + j];
            }
        }
    }

    double innov[ny];
    for (int j = 0; j < ny; ++j) {
        innov[j] = y_t[j] - y_pred[j];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ny; ++j) {
            x[i] += K[i * ny + j] * innov[j];
        }
    }

    double IKC[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            IKC[i * n + j] = (i == j) ? 1.0 : 0.0;
            for (int k = 0; k < ny; ++k) {
                IKC[i * n + j] -= K[i * ny + k] * cfg->C[k * n + j];
            }
        }
    }

    double tmp[n * n];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            tmp[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                tmp[i * n + j] += IKC[i * n + k] * P[k * n + j];
            }
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            P[i * n + j] = 0.0;
            for (int k = 0; k < n; ++k) {
                P[i * n + j] += tmp[i * n + k] * IKC[j * n + k];
            }
            for (int r = 0; r < ny; ++r) {
                for (int c = 0; c < ny; ++c) {
                    P[i * n + j] += K[i * ny + r] * cfg->R[r * ny + c] * K[j * ny + c];
                }
            }
        }
    }
}

void openmhe_filter_init(
    openmhe_filter_state_t *state,
    const openmhe_filter_config_t *cfg,
    double *x,
    double *P,
    openmhe_ukf_workspace_t *ws)
{
    const int n = cfg->nx;
    state->cfg = *cfg;
    state->posterior_t = -1;
    state->x = x;
    state->P = P;
    state->ws = ws;
    state->lambda = cfg->alpha * cfg->alpha * (n + cfg->kappa) - n;
    memset(x, 0, (size_t)n * sizeof(double));
    memset(P, 0, (size_t)(n * n) * sizeof(double));
    for (int i = 0; i < n; ++i) {
        P[i * n + i] = cfg->Q[i * n + i];
    }
}

static void ukf_sigma_points(
    const double *x,
    const double *P,
    int n,
    double lam,
    double *chi)
{
    double scale = n + lam;
    double L[n * n];
    double P_scaled[n * n];
    for (int i = 0; i < n * n; ++i) {
        P_scaled[i] = scale * P[i];
    }
    if (cholesky_lower(P_scaled, n, L) != 0) {
        for (int i = 0; i < n; ++i) {
            P_scaled[i * n + i] += CHOL_REG;
        }
        cholesky_lower(P_scaled, n, L);
    }
    memcpy(chi, x, (size_t)n * sizeof(double));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            chi[(1 + i) * n + j] = x[j] + L[j * n + i];
            chi[(1 + i + n) * n + j] = x[j] - L[j * n + i];
        }
    }
}

static void ukf_unscented_predict(
    const openmhe_filter_state_t *state,
    const double *u_t,
    double *x_pred,
    double *P_pred)
{
    const openmhe_filter_config_t *cfg = &state->cfg;
    const int n = cfg->nx;
    const int nu = cfg->nu;
    const double lam = state->lambda;
    const double alpha = cfg->alpha;
    const double beta = cfg->beta;

    double Wm[2 * n + 1];
    double Wc[2 * n + 1];
    const double denom = n + lam;
    for (int i = 0; i < 2 * n + 1; ++i) {
        Wm[i] = 0.5 / denom;
        Wc[i] = 0.5 / denom;
    }
    Wm[0] = lam / denom;
    Wc[0] = lam / denom + (1.0 - alpha * alpha + beta);

    double *chi = state->ws->chi;
    double *chi_pred = state->ws->chi_pred;
    ukf_sigma_points(state->x, state->P, n, lam, chi);

    double Bu[n];
    mat_vec(cfg->B, n, nu, u_t, Bu);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const double *pt = &chi[i * n];
        for (int j = 0; j < n; ++j) {
            chi_pred[i * n + j] = Bu[j];
            for (int k = 0; k < n; ++k) {
                chi_pred[i * n + j] += cfg->A[j * n + k] * pt[k];
            }
        }
    }

    for (int j = 0; j < n; ++j) {
        x_pred[j] = 0.0;
        for (int i = 0; i < 2 * n + 1; ++i) {
            x_pred[j] += Wm[i] * chi_pred[i * n + j];
        }
    }

    memcpy(P_pred, cfg->Q, (size_t)(n * n) * sizeof(double));
    for (int i = 0; i < 2 * n + 1; ++i) {
        double dx[n];
        for (int j = 0; j < n; ++j) {
            dx[j] = chi_pred[i * n + j] - x_pred[j];
        }
        outer_add(P_pred, n, dx, Wc[i]);
    }

}

static void ukf_unscented_update(
    const openmhe_filter_state_t *state,
    double *x_pred,
    double *P_pred,
    const double *y_t,
    const double *u_t)
{
    const openmhe_filter_config_t *cfg = &state->cfg;
    const int n = cfg->nx;
    const int ny = cfg->ny;
    const int nu = cfg->nu;
    const double lam = state->lambda;
    const double alpha = cfg->alpha;
    const double beta = cfg->beta;

    double Wm[2 * n + 1];
    double Wc[2 * n + 1];
    const double denom = n + lam;
    for (int i = 0; i < 2 * n + 1; ++i) {
        Wm[i] = 0.5 / denom;
        Wc[i] = 0.5 / denom;
    }
    Wm[0] = lam / denom;
    Wc[0] = lam / denom + (1.0 - alpha * alpha + beta);

    double *chi = state->ws->chi;
    double *gamma_pred = state->ws->gamma_pred;
    ukf_sigma_points(x_pred, P_pred, n, lam, chi);

    double Du[ny];
    mat_vec(cfg->D, ny, nu, u_t, Du);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const double *pt = &chi[i * n];
        for (int j = 0; j < ny; ++j) {
            gamma_pred[i * ny + j] = Du[j];
            for (int k = 0; k < n; ++k) {
                gamma_pred[i * ny + j] += cfg->C[j * n + k] * pt[k];
            }
        }
    }

    double y_mean[ny];
    for (int j = 0; j < ny; ++j) {
        y_mean[j] = 0.0;
        for (int i = 0; i < 2 * n + 1; ++i) {
            y_mean[j] += Wm[i] * gamma_pred[i * ny + j];
        }
    }

    double P_yy[ny * ny];
    double P_xy[n * ny];
    for (int i = 0; i < ny * ny; ++i) {
        P_yy[i] = cfg->R[i];
    }
    for (int i = 0; i < n * ny; ++i) {
        P_xy[i] = 0.0;
    }

    for (int i = 0; i < 2 * n + 1; ++i) {
        double dy[ny];
        double dx[n];
        for (int j = 0; j < ny; ++j) {
            dy[j] = gamma_pred[i * ny + j] - y_mean[j];
        }
        for (int j = 0; j < n; ++j) {
            dx[j] = chi[i * n + j] - x_pred[j];
        }
        outer_add(P_yy, ny, dy, Wc[i]);
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < ny; ++c) {
                P_xy[r * ny + c] += Wc[i] * dx[r] * dy[c];
            }
        }
    }

    double P_yy_inv[ny * ny];
    if (openmhe_symmetric_inv(P_yy, ny, P_yy_inv) != 0) {
        return;
    }

    double K[n * ny];
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ny; ++j) {
            K[i * ny + j] = 0.0;
            for (int k = 0; k < ny; ++k) {
                K[i * ny + j] += P_xy[i * ny + k] * P_yy_inv[k * ny + j];
            }
        }
    }

    double innov[ny];
    for (int j = 0; j < ny; ++j) {
        innov[j] = y_t[j] - y_mean[j];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < ny; ++j) {
            x_pred[i] += K[i * ny + j] * innov[j];
        }
    }
    subtract_kpk(P_pred, n, K, P_yy, ny);
}

void openmhe_filter_assimilate(
    openmhe_filter_state_t *state, const double *y_t, const double *u_t)
{
    if (state->cfg.kind == OPENMHE_FILTER_EKF) {
        ekf_predict(&state->cfg, state->x, state->P, u_t);
        ekf_update(&state->cfg, state->x, state->P, y_t, u_t);
        state->posterior_t++;
        return;
    }
    if (state->cfg.kind != OPENMHE_FILTER_UKF) {
        return;
    }
    const int n = state->cfg.nx;
    double x_pred[n];
    double P_pred[n * n];
    ukf_unscented_predict(state, u_t, x_pred, P_pred);
    ukf_unscented_update(state, x_pred, P_pred, y_t, u_t);
    memcpy(state->x, x_pred, (size_t)n * sizeof(double));
    memcpy(state->P, P_pred, (size_t)(n * n) * sizeof(double));
    state->posterior_t++;
}

void openmhe_filter_prior(
    const openmhe_filter_state_t *state,
    const double *u_t,
    double *x_bar,
    double *P_prior)
{
    if (state->cfg.kind == OPENMHE_FILTER_EKF) {
        const int n = state->cfg.nx;
        double x_tmp[n];
        double P_tmp[n * n];
        memcpy(x_tmp, state->x, (size_t)n * sizeof(double));
        memcpy(P_tmp, state->P, (size_t)(n * n) * sizeof(double));
        ekf_predict(&state->cfg, x_tmp, P_tmp, u_t);
        memcpy(x_bar, x_tmp, (size_t)n * sizeof(double));
        memcpy(P_prior, P_tmp, (size_t)(n * n) * sizeof(double));
        return;
    }
    if (state->cfg.kind != OPENMHE_FILTER_UKF) {
        return;
    }
    ukf_unscented_predict(state, u_t, x_bar, P_prior);
}

int openmhe_symmetric_inv(const double *P, int n, double *P_inv)
{
    double L[n * n];
    if (cholesky_lower(P, n, L) != 0) {
        double Preg[n * n];
        memcpy(Preg, P, (size_t)(n * n) * sizeof(double));
        for (int i = 0; i < n; ++i) {
            Preg[i * n + i] += CHOL_REG;
        }
        if (cholesky_lower(Preg, n, L) != 0) {
            return -1;
        }
    }

    for (int col = 0; col < n; ++col) {
        double b[n];
        double y[n];
        double x[n];
        memset(b, 0, (size_t)n * sizeof(double));
        b[col] = 1.0;
        for (int i = 0; i < n; ++i) {
            y[i] = b[i];
            for (int k = 0; k < i; ++k) {
                y[i] -= L[i * n + k] * y[k];
            }
            y[i] /= L[i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            x[i] = y[i];
            for (int k = i + 1; k < n; ++k) {
                x[i] -= L[k * n + i] * x[k];
            }
            x[i] /= L[i * n + i];
        }
        for (int row = 0; row < n; ++row) {
            P_inv[row * n + col] = x[row];
        }
    }
    return 0;
}
