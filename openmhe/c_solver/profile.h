#ifndef OPENMHE_PROFILE_H_
#define OPENMHE_PROFILE_H_

#include <time.h>

typedef struct {
    double setup;
    double solve;
    double shift;
} openmhe_profile_t;

#define OPENMHE_PROF_DECL struct timespec _omhe_ta, _omhe_tb

#ifdef OPENMHE_PROFILE

#include <stdio.h>

#define OPENMHE_PROF_NOW(t) clock_gettime(CLOCK_MONOTONIC, (t))

static inline void openmhe_profile_add(
    double *acc, const struct timespec *a, const struct timespec *b)
{
    *acc += (b->tv_sec - a->tv_sec) + 1e-9 * (b->tv_nsec - a->tv_nsec);
}

static inline void openmhe_profile_report(
    const openmhe_profile_t *p, int n_windows, const char *tag)
{
    fprintf(stderr,
        "[openmhe %s] windows=%d setup=%.1fms solve=%.1fms shift=%.1fms"
        " | per-window setup=%.4f solve=%.4f shift=%.4f ms\n",
        tag, n_windows, p->setup * 1e3, p->solve * 1e3, p->shift * 1e3,
        p->setup / n_windows * 1e3, p->solve / n_windows * 1e3,
        p->shift / n_windows * 1e3);
}

#else /* OPENMHE_PROFILE */

#define OPENMHE_PROF_NOW(t) ((void)0)

static inline void openmhe_profile_add(
    double *acc, const struct timespec *a, const struct timespec *b)
{
    (void)acc;
    (void)a;
    (void)b;
}

static inline void openmhe_profile_report(
    const openmhe_profile_t *p, int n_windows, const char *tag)
{
    (void)p;
    (void)n_windows;
    (void)tag;
}

#endif /* OPENMHE_PROFILE */

#endif /* OPENMHE_PROFILE_H_ */
