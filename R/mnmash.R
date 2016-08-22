#' @title multivariate, multiple regression extension of Adaptive SHrinkage (ASH)
#' @description ...
#' @param X [N, P] X data matrix
#' @param Y [N, J] Y data matrix
#' @param U [K, J, J] prior matrices
#' @param omega [L, 1] grids for priors
#' @param pi_0 [K * L, 1] weights for priors
#' @param control \{tol = 1E-5, maxiter = 10000, logfile = NULL, n_cpu = 1\} list of runtime variables
#' @return ...
#' @details ...
#' @author Gao Wang and Wei Wang
#' @references ...
#' @examples ...
#' @useDynLib mnmashr
#' @exportPattern ^[[:alpha:]]+
#' @importFrom Rcpp evalCpp
#' @export

mnmash <- function(X, Y, U, omega, pi_0 = NULL, control = NULL) {
  ## Here Y is residue and scaled by Sigma^{-1/2}
  ## Initialize data
  N <- nrow(X)
  P <- ncol(X)
  J <- ncol(Y)
  K <- dim(U)[3]
  L <- length(omega)
  if (is.null(pi_0)) {
    pi_0 <- rep(1 / (K * L), K * L)
  }
  tol <- as.double(control$tol)
  if (length(tol) == 0 || tol <= 0) {
    tol <- 1E-4
  }
  maxiter <- as.integer(control$maxiter)
  if (length(maxiter) == 0 || maxiter <= 0) {
    maxiter <- 10000
  }
  n_cpu <- as.integer(control$n_cpu)
  if (length(n_cpu) == 0 || n_cpu <= 0) {
    n_cpu <- 1
  }
  logfile <- control$logfile
  if (is.null(logfile)) {
    f1 <- f2 <- ""
  } else {
    f1 <- paste(logfile, "updates.log", sep = ".")
    f2 <- paste(logfile, "debug.log", sep = ".")
  }
  ## sanity check
  stopifnot(nrow(X) == nrow(Y))
  stopifnot(K * L == length(pi_0))
  for (i in 1:K) {
    stopifnot(nrow(U[,,i]) == ncol(U[,,i]) && ncol(U[,,i]) == J)
  }
  ## analysis
  rcpp_mnmash_vb(X, Y, U, omega, pi_0, tol, maxiter, n_cpu, c(f1, f2))
}
