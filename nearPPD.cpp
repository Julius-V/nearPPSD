#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;
using namespace arma;

// [[Rcpp::export]]
List nearPPD(mat X, const float eigenTol = 1e-06, const float convTol = 1e-07, 
                const float psdTol = 1e-08, const int maxit = 100) {
  const int n = X.n_cols;
  const mat X0 = X;
  mat DS(n, n, fill::zeros);
  mat Y, R;
  vec eigval;
  mat eigvec;
  int iter = 0;
  bool converged;
  uvec idx;
  vec un = unique(X0);
  do {
    Y = X;
    R = Y - DS;
    eig_sym(eigval, eigvec, R);
    uvec p = find(eigval > eigenTol * eigval(n - 1));
    if (p.size() == 0)
      stop("Matrix seems to be negative definite");
    eigvec = eigvec.cols(p);
    X = eigvec * diagmat(eigval(p)) * eigvec.t();
    DS = X - R;
    for (int i = 0; i<un.size(); i++) {
      idx = find(X0 == un(i));
      X.elem(idx).fill(mean(X.elem(idx)));
    }
    X.elem(find(X0 == 0)).fill(0);
    converged = (norm(Y - X) / norm(Y)) <= convTol;
    iter++;
  } while (iter < maxit && (!converged || min(eig_sym(X)) < 0));
  
  if (!converged)
    warning("'nearPPD' did not converge in %d iterations", iter);
}