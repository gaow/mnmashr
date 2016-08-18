// Gao Wang and Wei Wang (c) 2016
#ifndef _M2ASH_HPP
#define _M2ASH_HPP

#include <armadillo>
#include <map>
#include <omp.h>
#include <iostream>
#include <cmath>

static const double INV_SQRT_2PI = 0.3989422804014327;
static const double INV_SQRT_LOG_2PI = -0.91893853320467267;
static const double LOG_2PI = 1.8378770664093456;

inline double normal_pdf(double x, double m, double s)
{
	double a = (x - m) / s;

	return INV_SQRT_2PI / s * std::exp(-0.5 * a * a);
};

inline double normal_pdf_log(double x, double m, double s)
{
	double a = (x - m) / s;

	return INV_SQRT_LOG_2PI - std::log(s) - 0.5 * a * a;
};

class M2ASH
{
public:
	M2ASH(double * cX, double * cY, double * cU, double * cOmega, double * pi_0,
		int N, int P, int J, int K, int L) :
		// mat(aux_mem*, n_rows, n_cols, copy_aux_mem = true, strict = true)
		U(cU, J, J, K, false, true), omega(cOmega, L, false, true),
		pi(pi_0, K * L, false, true), P(P), J(J), K(K), L(L), N(N)
	{
		S.set_size(J, J * P, K * L);
		SI.set_size(J, J * P, K * L);
		VI.set_size(J, J, K * L);
		arma::cube V(J, J, K * L, arma::fill::zeros);
		mu.set_size(J, P, K * L);
		R.set_size(J, P);
		alpha.set_size(P, K * L);
		DSV.set_size(P, K * L);
		log_det_S.set_size(P, K * L);
		log_det_V.set_size(K * L);
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
				double t = k * L + l;
				V.slice(t) = omega.at(l) * U.slice(k);
				VI.slice(t) = arma::inv(V.slice(t));
				double ldet_v, sign_v;
				arma::log_det(ldet_v, sign_v, V.slice(t));
				log_det_V.at(t) = ldet_v;
				for (size_t p = 0; p < P; p++) {
					SI.slice(t).cols(p * J, p * J + J -
						1) =
					    tXX.at(p, p) * arma::eye<arma::mat>(J, J) + VI.slice(t);
					S.slice(t).cols(p * J, p * J + J -
						1) = arma::inv(SI.slice(t).cols(p * J, p * J + J - 1));
					double ldet_s, sign_s;
					arma::log_det(ldet_s, sign_s,
						S.slice(t).cols(p * J, p * J + J - 1));
					log_det_S.at(p, t) = ldet_s;
					DSV.at(p, t) = std::sqrt(std::exp(
							ldet_s) * sign_s / std::exp(ldet_v) / sign_v);
				}
			}
		}
		C1 = -1 * N * J * 0.5 * LOG_2PI;
		C2 = J * 0.5 * LOG_2PI;
	}


	~M2ASH() {}

	void print(std::ostream & out, int info)
	{
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


	void set_threads(int n)
	{
		n_threads = n;
	}


	void update()
	{
		R.zeros();
		for (size_t p = 0; p < P; p++) {
			for (size_t k = 0; k < K; k++) {
				for (size_t l = 0; l < L; l++) {
					R.col(p) = R.col(p) + alpha.at(k * L + l, p) * mu.slice(
						k * L + l).col(p);
				}
			}
		}
		//
		for (size_t k = 0; k < K; k++) {
			for (size_t l = 0; l < L; l++) {
				// for given prior w_l * U_k
				double t = k * L + l;
				pi.at(t) = arma::accu(alpha.col(t));
				for (size_t p = 0; p < P; p++) {
					mu.slice(t).col(p) =
					    S.slice(t).cols(p * J, p * J + J -
							1) *
					    (tYX.col(p) -
					     (arma::accu(tXX.col(p)) - tXX.at(p, p)) * R.col(p));
					double kernel = arma::dot(mu.slice(t).col(p).t() * SI.slice(
							t).cols(p * J, p * J + J - 1), mu.slice(t).col(p));
					alpha.at(p, t) = pi.at(t) * DSV.at(p, t) * std::exp(
						0.5 * kernel);
				}
			}
		}
		//
		alpha = arma::normalise(alpha, 1, 1);
		pi = arma::normalise(pi);
	}


	double get_logKL()
	{
		// K-L divergence
		// constant terms are dropped for now
		arma::mat tr1m(J, J, arma::fill::zeros);
		arma::mat tr2m(J, J, arma::fill::zeros);
		arma::mat tr3m(J, J, arma::fill::zeros);
		arma::mat tr4m(J, J, arma::fill::zeros);
		double pbeta = 0, qbeta = 0;

		for (size_t p1 = 0; p1 < P; p1++) {
			for (size_t p2 = 0; p2 < P; p2++) {
				if (p1 == p2) {
					tr1m = tr1m + R.col(p1) * tYX.col(p1).t();
					tr2m = tr2m + tXX.at(p1, p1) * (R.col(p1) * R.col(p1).t());
					arma::mat tmp1(J, J, arma::fill::zeros);
					for (size_t t = 0; t < K * L; t++) {
						arma::mat mu_outer_s = mu.slice(t).col(p1) *
						                       mu.slice(t).col(p1).t() +
						                       S.slice(t).cols(p1 * J,
							p1 * J + J - 1);
						tmp1 = tmp1 + alpha.at(p1, t) * mu_outer_s;
						pbeta +=
						    alpha.at(p1,
								t) *
						    (std::log(pi.at(t) - C2 - 0.5 * log_det_V.at(t) -
								 0.5 *
								 arma::trace(VI.slice(t) * mu_outer_s)));
						qbeta +=
						    alpha.at(p1,
								t) *
						    (std::log(alpha.at(p1,
									 t) - C2 - 0.5 *
								 log_det_S.at(p1, t) - J * 0.5));
					}
					tr3m = tr3m + tmp1 * tXX.at(p1, p1);
				} else {
					tr4m = tr4m + tXX.at(p1, p2) * (R.col(p1) * R.col(p2).t());
				}
			}
		}
		logKL = qbeta - pbeta -
		        (C1 + arma::trace(tr1m) - 0.5 * arma::trace(tr2m) + 0.5 *
		         arma::trace(tr3m) - 0.5 * arma::trace(tr4m));
		return logKL;
	}


private:
	arma::mat tXX;
	arma::mat tYX;
	arma::cube U;
	arma::vec omega;
	arma::vec pi;
	// updated quantities (intermediate)
	// K * L slices, J * (J * P) each
	arma::cube S;
	arma::cube SI;
	// K * L slices, J * J each
	arma::cube VI;
	// K * L slices, J * P
	arma::cube mu;
	// J * P matrix like a slice of mu
	arma::mat R;
	// K * L columns, P rows
	arma::mat alpha;
	arma::mat DSV;
	arma::mat log_det_S;
	// K * L vector, log |V_t|
	arma::vec log_det_V;
	// logKL
	double logKL;
	// number of threads
	int n_threads;
	// updates on the model
	int n_updates;
	// some constants
	int N, P, K, L, J;
	double C1, C2;
};
#endif
