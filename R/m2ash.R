#' @title multivariate, multiple regression extension of Adaptive SHrinkage (ASH)
#' @description ...
#' @param X [N, P] X data matrix
#' @param Y [N, J] Y data matrix
#' @param U [K, J, J] prior matrices
#' @param omega [L, 1] grids for priors
#' @param pi_0 [K * L, 1] weights for priors
#' @param control \{tol = 1E-5, maxiter = 10000, logfile = NULL, n_cpu = 1\} list of runtime variables
#' @return
#' @details ...
#' @author Gao Wang and Wei Wang
#' @references ...
#' @examples
#' @useDynLib m2ashr
#' @export

m2ash <- function(X, Y, U, omega, pi_0 = NULL, control = NULL) {
  ## Here Y is residue and scaled by Sigma^{-1/2}
  ## Initialize data
  N <- nrow(X)
  P <- ncol(X)
  J <- ncol(Y)
  K <- length(U)
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
    f1 <- n_f1 <- f2 <- n_f2 <- 0
  } else {
    f1 <- charToRaw(paste(logfile, "updates.log", sep = "."))
    f2 <- charToRaw(paste(logfile, "debug.log", sep = "."))
    n_f1 <- length(f1)
    n_f2 <- length(f2)
  }
  ## sanity check
  stopifnot(nrow(X) == nrow(Y))
  stopifnot(length(U) * length(omega) == length(pi_0))
  for (i in 1:length(U)) {
    stopifnot(nrow(U[[i]]) == ncol(U[[i]]) && ncol(U[[i]]) == ncol(Y))
  }
  ## analysis
  logKL <- rep(-999, maxiter)
  niter <- 0
  status <- 0
  res <- .C("m2ash_vb",
            as.double(as.vector(X)),
            as.double(as.vector(Y)),
            as.double(as.vector(unlist(U))),
            as.double(as.vector(omega)),
            pi_0 = as.double(as.vector(pi_0)),
            as.integer(N),
            as.integer(P),
            as.integer(J),
            as.integer(K),
            as.integer(L),
            as.double(tol),
            as.integer(maxiter),
            niter = as.integer(niter),
            logKL = as.double(as.vector(logKL)),
            status = as.integer(status),
            as.integer(as.vector(f1)),
            as.integer(n_f1),
            as.integer(as.vector(f2)),
            as.integer(n_f2),
            as.integer(n_cpu),
            PACKAGE = "m2ashr")
  logKL <- res$logKL[1:res$niter]
  return(list(pi = res$pi_0, logKL = logKL))
}
