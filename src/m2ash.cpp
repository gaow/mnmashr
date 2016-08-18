// Gao Wang and Wei Wang (c) 2016
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
#include "m2ash.hpp"

//! VB algorithm for m2ash
// @param X [N, P] X data matrix
// @param Y [N, J] Y data matrix
// @param U [K, J, J] prior matrices
// @param omega [L, 1] grids for priors
// @param pi_0 [K * L, 1] weights for priors
// @param N [int_pt] number of rows of matrix X and Y
// @param P [int_pt] number of columns of matrix X
// @param J [int_pt] number of columns of matrix Y and nrow & ncol of each U
// @param K [int_pt] number of U's
// @param L [int_pt] number of grids
// @param tol [double_pt] tolerance for convergence
// @param maxiter [int_pt] maximum number of iterations
// @param niter [int_pt] number of iterations
// @param loglik [maxiter, 1] log likelihood, track of convergence (return)
// @param status [int_pt] return status, 0 for good, 1 for error (return)
// @param logfn_1 [int_pt] log file 1 name as integer converted from character array
// @param nlf_1 [int_pt] length of above
// @param logfn_2 [int_pt] log file 2 name as integer converted from character array
// @param nlf_2 [int_pt] length of above
// @param n_threads [int_pt] number of threads for parallel processing

extern "C" int m2ash_vb(double *, double *, double *, double *, double *,
	int *, int *, int *, int *, int *,
	double *, int *, int *, double *, int *,
	int *, int *, int *, int *, int *);

int m2ash_vb(double * X, double * Y, double * U, double * omega, double * pi_0,
             int * N, int * P, int * J, int * K, int * L,
             double * tol, int * maxiter, int * niter, double * loglik,
             int * status,
             int * logfn_1, int * nlf_1, int * logfn_2, int * nlf_2,
             int * n_threads)
{
	//
	// Set up logfiles
	//
	bool keeplog = (*nlf_1 > 0) ? true : false;
	std::fstream f1;
	std::fstream f2;

	if (keeplog) {
		char f1_log[(*nlf_1) + 1];
		char f2_log[(*nlf_2) + 1];
		for (int i = 0; i < *nlf_1; i++)
			f1_log[i] = (char)*(logfn_1 + i);
		for (int i = 0; i < *nlf_2; i++)
			f2_log[i] = (char)*(logfn_2 + i);
		f1_log[*nlf_1] = '\0';
		f2_log[*nlf_2] = '\0';
		f1.open(f1_log, std::fstream::out);
		f2.open(f2_log, std::fstream::out);
		time_t now;
		time(&now);
		f1 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f2 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f1.close();
		f2.close();
		f1.open(f1_log, std::fstream::app);
		f2.open(f2_log, std::fstream::app);
	}
	//
	// Fit model via VB
	//
	*niter = 0;
	M2ASH model(X, Y, U, omega, pi_0, *N, *P, *J, *K, *L);
	model.set_threads(*n_threads);
	model.print(f2, 0);
	while (*niter <= *maxiter) {
		loglik[*niter] = model.get_logKL() * -1;
		(*niter)++;
		// check convergence
		if (*niter > 1) {
			double diff = loglik[(*niter) - 1] - loglik[(*niter) - 2];
			// check monotonicity
			if (diff < 0.0) {
				std::cerr <<
				"[ERROR] likelihood decreased in variational approximation!" <<
				std::endl;
				*status = 1;
				break;
			}
			// converged
			if (diff < *tol)
				break;
		}
		if (*niter == *maxiter) {
			// did not converge
			*status = 1;
			break;
		}
		// continue with more iterations
		model.update();
		if (keeplog) {
			f1 << "#----------------------------------\n";
			f1 << "# Iteration " << *niter << "\n";
			f1 << "#----------------------------------\n";
			model.print(f1, 1);
			f2 << "#----------------------------------\n";
			f2 << "# Iteration " << *niter << "\n";
			f2 << "#----------------------------------\n";
			model.print(f2, 2);
		}
	}
	if (*status)
		std::cerr <<
		"[WARNING] Variational inference failed to converge after " <<
		*niter <<
		" iterations!" << std::endl;
	if (keeplog) {
		f1.close();
		f2.close();
	}
	return 0;
}


