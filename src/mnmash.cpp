// Gao Wang and Wei Wang (c) 2016
#include <iostream>
#include <fstream>
#include <cstdio>
#include <ctime>
#include <cmath>
#include <vector>
#include <string>
#include <mnmash.hpp>

//! VB algorithm for mnmash
// @param X [N, P] X data matrix
// @param Y [N, J] Y data matrix
// @param U [K, J, J] prior matrices
// @param omega [L, 1] grids for priors
// @param pi_0 [K * L, 1] weights for priors
// @param tol [double] tolerance for convergence
// @param maxiter [int] maximum number of iterations
// @param n_threads [int] number of threads for parallel processing
// @param f1_log [string] log file 1 name as integer converted from character array
// @param f2_log [string] log file 2 name as integer converted from character array

MNMASH mnmash_vb(arma::mat X, arma::mat Y, arma::cube U, arma::vec omega, arma::vec pi_0,
                 double tol, int maxiter, int n_threads, std::string f1_log, std::string f2_log)
{
	//
	// Set up logfiles
	//
	bool keeplog = (f1_log.size() > 0 && f2_log.size() > 0) ? true : false;
	std::fstream f1;
	std::fstream f2;

	if (keeplog) {
		f1.open(f1_log.c_str(), std::fstream::out);
		f2.open(f2_log.c_str(), std::fstream::out);
		time_t now;
		time(&now);
		f1 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f2 << "#\n# " << asctime(localtime(&now)) << "#\n\n";
		f1.close();
		f2.close();
		f1.open(f1_log.c_str(), std::fstream::app);
		f2.open(f2_log.c_str(), std::fstream::app);
	}
	//
	// Fit model via VB
	//
	MNMASH model(X, Y, U, omega, pi_0);
	model.set_threads(n_threads);
	model.print(f2, 0);
  std::vector<double> logKL(0);
	while (model.get_niter() <= maxiter) {
		logKL.push_back(model.get_logKL());
		// check convergence
		if (model.get_niter() > 1) {
			double diff = -(logKL[model.get_niter() - 1] - logKL[model.get_niter() - 2]);
			// check monotonicity
			// FIXME: not converging right now!
			// if (diff < 0.0) {
			//  std::cerr <<
			//    "[ERROR] likelihood decreased in variational approximation!" <<
			//    std::endl;
			//  model.set_status(1);
			//  break;
			// }
			// converged
			// FIXME: should not have std::abs
			if (std::abs(diff) < tol)
				break;
		}
		if (model.get_niter() == maxiter) {
			// did not converge
			model.set_status(1);
			break;
		}
		// continue with more iterations
		model.update();
		if (keeplog) {
			f1 << "#----------------------------------\n";
			f1 << "# Iteration " << model.get_niter() << "\n";
			f1 << "#----------------------------------\n";
			model.print(f1, 1);
			f2 << "#----------------------------------\n";
			f2 << "# Iteration " << model.get_niter() << "\n";
			f2 << "#----------------------------------\n";
			model.print(f2, 2);
		}
	}
	if (model.get_status())
		std::cerr <<
		  "[WARNING] Variational inference failed to converge after " <<
		  model.get_niter() <<
		  " iterations!" << std::endl;
  model.set_logKL_all(logKL);
  if (keeplog) {
		f1.close();
		f2.close();
	}
	return model;
}
