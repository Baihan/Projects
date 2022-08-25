n_samples <- 100000
n_ads_shown <- 100
proportion_clicks <- runif(n_samples, min = 0.0, max = 0.2)
n_visitors <- rbinom(n = n_samples, size = n_ads_shown, prob = proportion_clicks)

# Visualize proportion clicks
hist(proportion_clicks)

# Visualize n_visitors
hist(n_visitors)

# Create the prior data frame
prior <- data.frame(proportion_clicks, n_visitors)

# Examine the prior data frame
head(prior)

# Create the posterior data frame
posterior <- prior[prior$n_visitors == 13, ]

# Visualize posterior proportion clicks
hist(posterior$proportion_clicks)