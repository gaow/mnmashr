// Gao Wang and Wei Wang (c) 2016
#ifndef _M2ASH_HPP
#define _M2ASH_HPP

#include <armadillo>
#include <map>
#include <omp.h>

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double INV_SQRT_2PI_LOG = -0.91893853320467267;

inline double normal_pdf(double x, double m, double s)
{
  double a = (x - m) / s;
  return INV_SQRT_2PI / s * std::exp(-0.5 * a * a);
};

inline double normal_pdf_log(double x, double m, double s)
{
  double a = (x - m) / s;
  return INV_SQRT_2PI_LOG - std::log(s) -0.5 * a * a;
};

class M2ASH {
public:
  M2ASH(double * cX, double * cY, double * cU, double *cOmega, double * pi_0,
        int N, int P, int J, int K, int L):
    // mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
    X(cX, N, P, false, true), Y(cY, N, J, false, true),
    U(cU, J, J, K, false, true), omega(cOmega, L, false, true),
    pi(pi_0, K * L, false, true)
  {
    S.set_size(K, J, J);
    mu.set_size(K, J);
    alpha.set_size(K, J);
    n_threads = 1;
    n_updates = 1;
  }
  ~M2ASH() {}

  void print(std::ostream& out, int info) {
    if (info == 0) {
      // debug
      X.print(out, "X Matrix:");
      Y.print(out, "Y Matrix:");
      omega.print(out, "Grid vector:");
      U.print(out, "U tensor:");
      pi.print(out, "pi vector");
    }
  }

  void set_threads(int n) {
    n_threads = n;
  }

  double get_loglik() {
    return loglik;
  }
  void update() {}

private:
  arma::mat X;
  arma::mat Y;
  arma::mat P;
  arma::cube U;
  arma::vec omega;
  arma::vec pi;
  // updated quantities
  // K slices, J X J each
  arma::cube S;
  // K columns, J rows
  arma::mat mu;
  // K columns, J rows
  arma::mat alpha;
  // loglik
  double loglik;
  // number of threads
  int n_threads;
  // updates on the model
  int n_updates;
};
#endif
