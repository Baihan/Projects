# The generative zombie drug model

# Set parameters
prop_success <- 0.42
n_zombies <- 100

# Simulating data
data <- c()
for(zombie in 1:n_zombies) {
  data[zombie] <- runif(1, min = 0, max = 1) < prop_success
}

# Count cured
data <- sum(data)
data

# Change the parameters
rbinom(n = 200, size = 100, prob = 0.42)