##############################################################
# WORKSPACE SETUP
##############################################################

# Clear workspace
rm(list = ls())

# Set seed for reproducibility
 set.seed(100)

# Load required packages
library(dplyr)
library(deSolve)
library(mvtnorm)
library(pracma)
library(reshape)
library(ggplot2)
library(tidyr)
library(ggthemes)
library(latex2exp)


##############################################################
# PROBLEM AND SMC SET UP (INPUT REQUIRED)
##############################################################

# Specify method of determining next design point
R <- 1

# Specify models to be considered
models <- 1:1 %>% matrix(nrow = 1)
modeltype <- list("Binomial type 3 functional response model")

# Assign values to parameters (Input required for these values)
I <- 10 # number of data points
K <- length(models) # number of models
N <- 500 # number of particles for SMC
E <- N/2 # threshold for ESS for SMC
Nmin <- 1 # min value in the discrete design space
Nmax <- 300 # max value in the discrete design space
time <- 24 # length of time that predator has access to prey in hours
tol <- 2 # tolerance for ESS which indicates when the move step might fail (see move step script for more detail)
i <- 1 # loop index

# Load required functions
source('scripts/R/generate_data.R')
source('scripts/R/prior_sampling.R')