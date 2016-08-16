// Gao Wang and Wei Wang (c) 2016
#ifndef _M2ASH_HPP
#define _M2ASH_HPP

#include <armadillo>
#include <map>
#include <omp.h>
#include <iostream>

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
    U(cU, J, J, K, false, true), omega(cOmega, L, false, true),
    pi(pi_0, K * L, false, true), P(P), J(J), K(K), L(L)
  {
    S.set_size(P * J, J, K * L);
    mu.set_size(P, J, K * L);
    alpha.set_size(P, K * L);
    n_threads = 1;
    n_updates = 1;
    // initialize data matrices
    arma::mat X(cX, N, P, false, true);
    arma::mat Y(cY, N, J, false, true);
    tXX = X.t() * X;
    tYX = Y.t() * X;
    // initialize alpha with uniform weights
    alpha.ones();
    alpha = alpha / double(P);
    // initialize mu
    mu.zeros();
    // initialize S_{tp}
    for (size_t k = 0; k < K; k++) {
      for (size_t l = 0; l < L; l++) {
        // for given prior w_l * U_k
        arma::mat Vi = arma::inv(omega.at(l) * U.slice(k));
        for (size_t p = 0; p < P; p++) {
          S.slice(k * L + l).rows(p * J , p * J + J - 1) = arma::inv(tXX.at(p, p) * arma::eye<arma::mat>(J, J) + Vi);
        }
      }
    }
  }
  ~M2ASH() {}

  void print(std::ostream& out, int info) {
    if (info == 0) {
      // debug
      tXX.print(out, "X'X matrix:");
      tYX.print(out, "Y'X matrix:");
      omega.print(out, "Grid vector:");
      U.print(out, "U tensor:");
      pi.print(out, "pi vector:");
      alpha.print(out, "alpha matrix:");
      S.print(out, "S tensor:");
    }
  }

  void set_threads(int n) {
    n_threads = n;
  }

  double get_loglik() {
    return loglik;
  }
  void update() {
    // R is a P by J matrix like a slice of mu
    arma::mat R(P, J, arma::fill::zeros);
    for (size_t p = 0; p < mu.n_rows; p++) {
      for (size_t k = 0; k < U.n_slices; k++) {
        for (size_t l = 0; l < omega.n_elem; l++) {
        }
      }
    }

    for (size_t k = 0; k < U.n_slices; k++) {
      for (size_t l = 0; l < omega.n_elem; l++) {
        // for given prior w_l * U_k
        arma::mat V = omega.at(l) * U.slice(k);
        for (size_t p = 0; p < mu.n_rows; p++) {
          arma::vec r();
          double xx = arma::accu(tXX.col(p)) - tXX.at(p, p);
          
          mu.slice(k * L + l).row(p) = S.slice(k * L + l).rows(p * J , p * J + J - 1) * (tYX.col(p) - xx * r)
        }
      }
    }
  }

private:
  arma::mat tXX;
  arma::mat tYX;
  arma::cube U;
  arma::vec omega;
  arma::vec pi;
  // updated quantities
  // K * L slices, P * J X J each
  arma::cube S;
  // K * L slices, J columns, P rows
  arma::cube mu;
  // K * L columns, P rows
  arma::mat alpha;
  // loglik
  double loglik;
  // number of threads
  int n_threads;
  // updates on the model
  int n_updates;
  // data dimensions
  int P, K, L, J;
};
#endif
