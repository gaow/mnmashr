#include <RcppArmadillo.h>
#include <mnmash.hpp>

// [[Rcpp::export]]
Rcpp::List rcpp_mnmash_vb(Rcpp::NumericMatrix X, Rcpp::NumericMatrix Y,
                     Rcpp::NumericVector U_, Rcpp::NumericVector omega,
                     Rcpp::NumericVector pi_0, double tol, int maxiter,
                     int n_threads, Rcpp::StringVector filenames) {
  Rcpp::IntegerVector dimU = U_.attr("dim");
  arma::cube U(U_.begin(), dimU[0], dimU[1], dimU[2], false, true);
  MNMASH model = mnmash_vb(Rcpp::as<arma::mat>(X),
                           Rcpp::as<arma::mat>(Y),
                           U,
                           Rcpp::as<arma::vec>(omega),
                           Rcpp::as<arma::vec>(pi_0),
                           tol, maxiter, n_threads,
                           Rcpp::as<std::string>(filenames[0]),
                           Rcpp::as<std::string>(filenames[1]));
}
