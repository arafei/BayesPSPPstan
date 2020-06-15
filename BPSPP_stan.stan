
functions {
  // Spline function
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order);
  vector build_b_spline(real[] t, real[] ext_knots, int ind, int order) {
    // INPUTS:
    //    t:          the points at which the b_spline is calculated
    //    ext_knots:  the set of extended knots
    //    ind:        the index of the b_spline
    //    order:      the order of the b-spline
    vector[size(t)] b_spline;
    vector[size(t)] w1 = rep_vector(0, size(t));
    vector[size(t)] w2 = rep_vector(0, size(t));
    if (order==1)
      for (i in 1:size(t)) // B-splines of order 1 are piece-wise constant
        b_spline[i] = (ext_knots[ind] <= t[i]) && (t[i] < ext_knots[ind+1]);
    else {
      if (ext_knots[ind] != ext_knots[ind+order-1])
        w1 = (to_vector(t) - rep_vector(ext_knots[ind], size(t))) /
             (ext_knots[ind+order-1] - ext_knots[ind]);
      if (ext_knots[ind+1] != ext_knots[ind+order])
        w2 = 1 - (to_vector(t) - rep_vector(ext_knots[ind+1], size(t))) /
                 (ext_knots[ind+order] - ext_knots[ind+1]);
      // Calculating the B-spline recursively as linear interpolation of two lower-order splines
      b_spline = w1 .* build_b_spline(t, ext_knots, ind, order-1) +
                 w2 .* build_b_spline(t, ext_knots, ind+1, order-1);
    }
    return b_spline;
  }

  //quantile function
  vector quantile(vector unordered_x, int num_quants);
  vector quantile(vector unordered_x, int num_quants) {
    real Q[num_quants];
    real frac;
    Q[1] = min(unordered_x);
    Q[num_quants] = max(unordered_x);
    frac = (Q[num_quants] - Q[1])/(num_quants-1.0);
    for(id in 2:(num_quants-1))
       Q[id] = Q[id-1] + frac;
    return(to_vector(Q)); 
  }
}

data {
  int<lower=0> k;            // num of knots
  int<lower=0> d;            // the degree of spline (is equal to order - 1)
  int<lower=0> n0;           // number of data points
  int<lower=0> n1;           // number of data points
  int<lower=0> pz;           // X design matrix for modeling E(Z=1|X)
  int<lower=0> py;           // X design matrix for modeling E(Y|X)
  real Y[n1];
  real Wt[n1+n0];
  matrix[n1+n0,pz] Xz;
  matrix[n1+n0,py] Xy;
  int<lower=0,upper=1> Z[n1+n0];
}

transformed data {
  int num_basis = k + d - 1; // total number of B-splines
  int<lower=0> n;
  matrix[n1, pz] Xz1;
  matrix[n0, pz] Xz0;
  matrix[n1, py] Xy1;
  matrix[n0, py] Xy0;
  n = n0 + n1;
  Xz1 = Xz[1:n1,:];
  Xz0 = Xz[(n1+1):n,:];
  Xy1 = Xy[1:n1,:];
  Xy0 = Xy[(n1+1):n,:];
}

parameters {
  row_vector[num_basis] a_raw;
  real a0;  // intercept
  vector[py] a1;  
  real<lower=0> sigma;
  real<lower=0> tau;
  real b0;
  vector[pz] b;
}

transformed parameters {
  row_vector[num_basis] a;
  vector[n1] Y_hat1;
  vector[n] Z_hat;
  vector[n] PS;
  matrix[num_basis, n] B;  // matrix of B-splines
  vector[d + k] ext_knots_temp;
  vector[2*d + k] ext_knots; // set of extended knots
  vector[k] knots;           // the sequence of knots    
  Z_hat = b0 + Xz*b;
  PS = to_vector(Wt) .* exp(-Z_hat);
  PS = logit(inv(PS));
  knots = quantile(PS, k);
  ext_knots_temp = append_row(rep_vector(knots[1], d), knots);
  ext_knots = append_row(ext_knots_temp, rep_vector(knots[k], d));
  a[1] = a_raw[1];
  B[1,:] = to_row_vector(build_b_spline(to_array_1d(PS), to_array_1d(ext_knots), 1, d + 1));
  for (ind in 2:num_basis){
    a[ind] = a[ind-1] + a_raw[ind]*tau;
    B[ind,:] = to_row_vector(build_b_spline(to_array_1d(PS), to_array_1d(ext_knots), ind, d + 1));
  }
  B[k + d - 1, n] = 1;

  Y_hat1 = a0*to_vector(PS[1:n1]) + to_vector(a*B[:,1:n1]) + Xy1*a1; //
}

model {
  // Priors
  b0 ~ normal(0, 1);
  b ~ normal(0, 1);
  a_raw ~ normal(0, 1);
  a0 ~ normal(0, 1);
  a1 ~ normal(0, 1);
  tau ~ normal(0, 1);
  sigma ~ normal(0, 1);

  //Likelihood
  Z ~ bernoulli_logit(Z_hat);
  Y ~ normal(Y_hat1, sigma);
}

generated quantities {
  vector[n0] Y0;
  Y0 = to_vector(normal_rng(a0*PS[(n1+1):n] + to_vector(a*B[:,(n1+1):n]) + Xy0*a1, sigma)); // 
}
