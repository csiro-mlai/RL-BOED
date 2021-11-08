## DESIGN NEXT EXPERIMENT

Nt <- Nmin:Nmax # possible designs

utility <- matrix(0, length(Nt), 1)
u1 <- matrix(0, K, 1)

for (j in 1:length(Nt)){
  for (mod in 1:K){
    parameters <- find_parameters(theta[,,mod], Nt[j], time, models[mod]) # Determine the mu and lambda parameters for all particles
    y <- 0:Nt[j] # define possible responses
    llh <- log_lik(y, Nt[j], parameters[,1], parameters[,2], models[mod]) # Calculate the log likelihood of observing the datapoint [Nt(j), y]
    Z_next = t(exp(llh)) %*% W[,mod] # calculate the marginal likelihood
    W_hat = exp(llh) * W[,mod]  # determine updated unnormalised weights
    w_hat = scale(W_hat, center=FALSE, scale=colSums(W_hat)) # normalise updated weights
    A = llh * w_hat # multiply log likelihood by normalised weights
    A[is.nan(A)] <- 0
    u = colSums(A) - log(colSums(W_hat)) # Determine utility for design point Nt(j), observation y and model M
    u[u==Inf]<- 0
    u1[mod] = u %*% Z_next # Average utility over all responses
    rm(parameters, y, llh, Z_next, W_hat, w_hat, u)
  }
  log_Z_n = log_Z - logsumexp(log_Z,0) # Normalise evidence
  utility[j] = exp(log_Z_n) %*% u1 # Average utility over all models
}

# Display the optimal design point
idx = which(utility == max(utility))
cat(paste0('Optimal design point at ', Nt[idx]), sep ="\n")
data[i,1] <- Nt[idx]
cat(" ", sep ="\n")
