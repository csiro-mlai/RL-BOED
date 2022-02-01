##############################################################
# CONDUCTING THE SMC (NO INPUT REQUIRED)
##############################################################

# Initialise other required quantities (Input not needed for these values)
data <- matrix(0, I, 2) # Set up data
log_Z = matrix(0, 1, K) # initialise log estimate of evidence
loglik = matrix(0, N, K) # initialise vector of log likelihood
logpri = matrix(0, N, K) # initialise vector of log prior
px = matrix(0, N, K) # initialise vector of the log posterior
w = matrix(0, N, K) # initialise vector of unnormalised weights of particles
ESS = matrix(0, 1, K) # initialise vector of effective sample size of each model

# Draw theta from prior distribution, set intial weighting, set covariance matrices

W <- rep(1/N, N * K) %>% matrix(nrow = N)

all_cov_matrices <- list()
for(M in 1:K){
  all_cov_matrices[[M]] = cov(theta[,,M])
}
