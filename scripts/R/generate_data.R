#-----------------------------------------------------------------------------------------------#
# Function Name: generate_data                                                                  #
# Author: Hayden Moffat                                                                         #
# email: hayden.moffat@hdr.qut.edu.au                                                           #
#                                                                                               #
# This function generates random observations at specified design points from either the       #
# beta-binomial distribution or the binomial distribution (depending on the model of interest). #
#                                                                                               #
# Inputs:                                                                                       #
# design - design at which we will generate the data                                            #
# theta -  true model parameters                                                                #
# observation_time - length of time that predator has access to prey in hours                   #
# model - model of interest                                                                     #
#                                                                                               #
# Ouputs:                                                                                       #
# data - the response generated from the true model                                             #
#                                                                                               #                                                                                            #
#-----------------------------------------------------------------------------------------------#


generate_data <- function(design, theta, observation_time, model){

  # Binomial distribution
  v <- solve_ode_Hollings(exp(theta), N0= design, observation_time = observation_time, model= model) # solve ode

  design_matrix <- matrix(rep(design, times = nrow(theta)), byrow = T, nrow = nrow(theta))
  mu <- (design_matrix - v)/design_matrix # determine parameter

  # Generate data
  final_data <- list()
  for (i in 1:nrow(theta)){
    data <- rep(NA, length(design))
    for (j in 1:length(design)){
      data[j] <- rbinom(1, design[j], mu[i,j])
    }
    final_data[[i]] <- data
    rm(data)
  }

  return(final_data)
}

###########################################################################
# Function name: log_prior
# This function computes the log prior probability of a set of parameters.
# Here we consider a log normal prior on each of the parameters where
# the mean and standard deviation of the natural logarithm of the variables
# are -1.4 and 1.35, respectively.
#
# Input:
# x - transformed parameters
# model - model of interest
#
# Output:
# p - log prior probability for each set of transformed parameters
#
###########################################################################

# This function computes the log prior probability of a set of parameters.
log_prior <- function(x, model){
  if (model %in% 1:2){
    # Prior probabilities for Models 1 and 2
    p <- dnorm(x[,1], -1.4, 1.35, log =T ) + dnorm(x[,2], -1.4, 1.35, log =T ) + dnorm(x[,3], -1.4, 1.35, log =T )
  } else if (model %in% 3:4){
    # Prior probabilities for Models 3 and 4
    p <- dnorm(x[,1], -1.4, 1.35, log =T ) + dnorm(x[,2], -1.4, 1.35, log =T )
  }
}

###########################################################
# Function name: ode_Hollings
# This function defines Holling's Type II and Type III differential equations.
#
# Inputs:
# tau - time
# N - prey population
# phi - original parameters
# model - model of interest
#
# Outputs:
# f - numerical value
#
#
###########################################################


ode_Hollings <- function(tau, N, phi, model) {

  # Equation for Holling's Type III model
  f <- - phi[1]*N^2/(1+phi[1]*phi[2]*N^2)
  return(list(f))
}



#############################################################################################
# Function name: solve_ode_Hollings
# Solves the ode for a range of parameters and initial
# conditions for a given model and time
#
# Inputs:
# phi - a matrix of the non-transformed parameter values.
# N0 - initial prey population.
# observation_time - time at which the experiment finishes and the data is observed.
#
# Outputs:
# solution - a numeric vector
#
############################################################################################


solve_ode_Hollings <- function(phi, N0, observation_time, model) {

  solution <- matrix(rep(NA, times = length(N0)*nrow(phi)), nrow= nrow(phi))
  Hol.t <- seq(0, observation_time, len = 10) # define the time range in which the ODE will be solved

  for (i in 1:nrow(phi)){

    res <- ode(N0, Hol.t, ode_Hollings, phi[i,], model = model)  # solve the ODE
    solution[i,] <- res[nrow(res),2:ncol(res), drop = FALSE]# only look at the final end point since this is the information that is relevant to the obervational data

  }

  solution[solution < 0] <- 0
  return(solution)

}


#######################################################################################################################
# Function name: loglikelihood_Hollings
# This function determines the log likelihood of observing the data for a given set of non_transformed parameters.
#
# Inputs:
# phi - a set of non_transformed parameters
# data - experiment data (designs are in the first column and the corresponding responses are in the second)
# observation_time - total exposure time of predator and prey
# model - model of interest
#
# Outputs:
# loglik - a numerical vector with the with a length equal to the number of particles
#
######################################################################################################################


loglikelihood_Hollings <- function(phi, data, observation_time, model) {

    N <- unique(data[,1])
    ic <- match(data[,1], N)
    v <- solve_ode_Hollings(phi, N0=N, observation_time = observation_time, model = model) %>% matrix(ncol = length(N))
    V <- v[,ic, drop = FALSE]

    # determine probability of prey being eaten
    ratio <- t(V) * 1/N[ic]
    prob <- 1-ratio

    n.eaten <- data[,2]
    loglik <- colSums(log_binpdf(data[,1], n.eaten, prob))

  return(loglik)
}


###########################################################################
# Function name: find_parameters
# For each particle in the given particle set and an initial condition,
# this function determines the parameter for the expected proportion
# of prey eaten and over-dispersion parameter for a specified
# time and model
#
# Inputs:
# theta - current particle set
# N - initial condition
# time - length of time that predator has access to prey in hours
# M - model of interest
#
# Outputs:
# para - numeric matrix containing the complement to 1 of the expected proportion
# and the overdispersion parameter for each particle
#
#
###########################################################################


find_parameters <- function(theta, N, observation_time, model){

  phi <- exp(theta) # transform the particle values
  V <- solve_ode_Hollings(phi, N0=N, observation_time = observation_time, model = model) %>% matrix(ncol = 1) # solve ode

  # Calculate mu and lambda
  mu_v <- V/N
  lambda = phi[,3]
  para = cbind(mu_v, lambda)

  return(para)

}

################################################################################################
# Function name: log_lik
# Given the expected proportion of prey eaten and over-dispersion
# parameter for each particle, this function determines the
# log likelihood of observing data
#
# Inputs:
# y - responses to each experiment
# N - intial prey populations for each experiment
# mu_v - the complement to 1 of the expected proportion of prey eaten (mu_v = 1-mu)
# lambda - over-dispersion parameter
# mod - model of interest
#
# Outputs:
# loglikelihoods - a numerical vector with the with a length equal to the number of particles
#
###############################################################################################

log_lik <- function(y, N, mu_v, lambda, model){

  loglikelihoods <- matrix(NA, nrow = length(mu_v), ncol = length(y))

  for (particle in 1:length(mu_v)){

    # If the expected proportion of prey eaten is 1 then we say it is certain that all of the prey are eaten
    if (log(mu_v[particle])==-Inf){
      loglikelihoods[particle,] = c(rep(-Inf,length(y)-1), 0)

    } else{
      loglikelihoods[particle,] = log_binpdf(rep(N, length(y)), y, 1-mu_v[particle])

    }

  }
  return(loglikelihoods)

}


###########################################################################
# Function name: log_binpdf
# Performs a stable calculation of the logarithm of the binomial pdf
#
# Input:
# N - number of trials
# n - number of successes
# p - probability of success
#
# Output:
# y - logarithm of the density
#
###########################################################################

log_binpdf <- function(N,n,p){
  y <- (lgamma(N + 1)-lgamma(n + 1)-lgamma(N - n + 1) + (n*log(p)) + ((N-n) * log(1-p)))
  y[is.nan(y)] = -Inf
  return(y)
}


#######################################################################################
# Function name: logsumexp
# Performs a stable calculation of log(sum(exp(x)))
#
# Inputs:
# x - numeric vector or matrix that does not exceed 2 dimensions
# margin - dimension to apply summation
#
# Outputs:
# f - numeric vector of the same columns or rows (depending on margin) as x
#
#######################################################################################

logsumexp <- function(x, margin = 0){

  y <- t(x)

  if (margin ==0){
    my_max <- max(y)
    y <- y - my_max
    f <- my_max + log(sum(exp(y)))

  } else{
    my_max <- apply(y, margin, max)
    y <- sweep(y, margin, my_max, FUN = '-')
    f <- my_max + log(apply(exp(y), margin, sum))

  }
  return(f)
}

###########################################################################
# Function name: multinomial_resampling
# This function conducts multinomial resampling for when ESS becomes too low
#
# Input:
# theta - current particle set
# W - current weights for each particle
# N - number of particles
# n - number of samples we require
#
# Output:
# th - a numeric vector or matrix containing n samples from theta
#
###########################################################################

# This function conducts multinomial resampling for when ESS becomes too low
multinomial_resampling <- function(W, N, n, theta){
  idx <- rep.int(1:N, rmultinom(1, n, W))
  th <- theta[idx,]
  return(th)
}

###########################################################################
# Function name: residual_resampling
# This function conducts residual resampling for when ESS becomes too low
#
# Inputs:
# theta - current particle set
# W - current weights for each particle
# N - number of particles
#
# Outputs:
# th - a numeric vector or matrix containing N samples from theta
#
###########################################################################

# This function conducts residual resampling for when ESS becomes too low
residual_resampling <- function(W, N, theta){
  th <- matrix(rep(0, nrow(theta)*ncol(theta)), nrow = nrow(theta), ncol = ncol(theta))
  R <- floor(W*N)
  sumR <- sum(R)
  idx <- rep.int(1:N, R)
  th[1:sumR,] <- theta[idx,]
  wt <- W*N - R;
  th[(sumR+1):N, ] <- multinomial_resampling((wt)/(sum(wt)),N,N-sumR,theta)
  return(th)
}