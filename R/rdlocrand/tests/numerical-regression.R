library(rdlocrand)

quiet <- function(expr) {
  value <- NULL
  invisible(capture.output(value <- suppressWarnings(force(expr))))
  value
}
equal <- function(actual, expected, tolerance = 1e-10) {
  stopifnot(isTRUE(all.equal(unname(actual), unname(expected),
                             tolerance = tolerance, check.attributes = FALSE)))
}

# Window boundaries must describe the observations used in the tests.
R <- c(-(1:40), (0:39) + 0.5)
set.seed(1)
X <- matrix(rnorm(80), 80, 1)
out <- quiet(rdwinselect(R, X, nwindows = 3, approx = TRUE))
equal(out$results[1, 4:7], c(10, 10, -10, 10))
for (wmin in list(10, c(-10, 10))) {
  out <- quiet(rdwinselect(R, X, wmin = wmin, wasymmetric = TRUE,
                           nwindows = 3, approx = TRUE))
  for (j in seq_len(nrow(out$results))) {
    lo <- out$results[j, 6]; hi <- out$results[j, 7]
    equal(out$results[j, 4:5], c(sum(R >= lo & R < 0), sum(R >= 0 & R <= hi)))
  }
}
R <- c(rep(-(1:10), each = 3), rep((0:9) + 0.5, each = 2))
set.seed(3)
X <- matrix(rnorm(50), 50, 1)
out <- quiet(rdwinselect(R, X, wmasspoints = TRUE, nwindows = 3, approx = TRUE))
equal(out$results[, 4:7], cbind(3*(1:3), 2*(1:3), -(1:3), (0:2)+0.5))
stopifnot(is.finite(out$results[1, 1]))

# KS must agree with the empirical CDF definition, including mixed and binary ties.
ks <- getFromNamespace('ksmirnov.statistic', 'rdlocrand')
cases <- list(list(c(0,0,0,1,1), c(0,0,0,1,1)),
              list(c(0,0,0,1), c(0,1,1,1)),
              list(c(-1,-1,0,2), c(-1,0,0,2,2)),
              list(c(-1.2,-.4,.1,.7,1.5), c(-1.1,-.8,.2,.4,1,1.7)))
set.seed(41)
for (i in 1:40) cases[[length(cases)+1]] <- list(sample(0:3, 7, TRUE), sample(0:3, 9, TRUE))
for (pair in cases) {
  ref <- suppressWarnings(ks.test(pair[[1]], pair[[2]])$statistic)
  equal(ks(pair[[1]], pair[[2]]), ref)
  equal(ks(pair[[2]], pair[[1]]), ref)
}
set.seed(5)
R <- runif(60, -1, 1)
Y <- rbinom(60, 1, .3 + .2*(R >= 0))
D <- as.numeric(R >= 0)
ref_stat <- function(y, d) suppressWarnings(unname(ks.test(y[d == 0], y[d == 1])$statistic))
for (seed in c(1, 3, 10)) {
  set.seed(seed)
  observed <- ref_stat(Y, D)
  simulated <- replicate(199, ref_stat(Y, sample(D)))
  expected <- mean(simulated >= observed)
  out <- quiet(rdrandinf(Y, R, wl = -1, wr = 1, statistic = 'ksmirnov', reps = 199, seed = seed))
  equal(out$p.value, expected)
  out <- quiet(rdrandinf(Y, R, wl = -1, wr = 1, statistic = 'all', reps = 199, seed = seed))
  equal(out$p.value[2], expected)
}

# Approximate polynomial balance tests agree with an independent full regression.
set.seed(9)
R <- runif(400, -5, 5)
X <- cbind(sample(1:3, 400, TRUE), rpois(400, 2), rpois(400, 1))
for (p in c(0, 1, 2)) for (kernel in c('uniform', 'triangular', 'epan')) {
  out <- quiet(rdwinselect(R, X, wmin = .5, wstep = .25, nwindows = 4,
                           approx = TRUE, p = p, kernel = kernel, vce = "HC2"))
  for (j in 1:4) {
    w <- out$results[j, 7]
    inside <- abs(R) <= w
    r <- R[inside]; d <- as.numeric(r >= 0)
    weights <- switch(kernel, uniform = rep(1, length(r)),
                      triangular = 1 - abs(r/w), epan = .75*(1-(r/w)^2))
    weights[weights == 0] <- .Machine$double.eps
    ref <- sapply(1:ncol(X), function(k) {
      y <- X[inside, k]
      if (p == 0) fit <- lm(y ~ d, weights = weights)
      else {
        powers <- outer(r, seq_len(p), '^')
        fit <- lm(y ~ d * powers, weights = weights)
      }
      se <- sqrt(sandwich::vcovHC(fit, type = 'HC2')['d','d'])
      2*pnorm(-abs(coef(fit)['d']/se))
    })
    equal(out$results[j, 1], min(ref))
  }
}

# Nonzero sharp nulls use the same regression and HC2 variance as zero nulls.
set.seed(2)
R <- runif(80, -1, 1)
D <- as.numeric(R >= 0)
Y <- 2*D + rnorm(80)
for (p in c(1, 2)) for (kernel in c('uniform', 'triangular', 'epan')) {
  weights <- switch(kernel, uniform = rep(1, length(R)),
                    triangular = 1-abs(R), epan = .75*(1-R^2))
  powers <- outer(R, seq_len(p), '^')
  fit <- lm(Y ~ D * powers, weights = weights)
  se <- sqrt(sandwich::vcovHC(fit, type = 'HC2')['D','D'])
  for (tau in c(0, 2, unname(coef(fit)['D']))) {
    out <- quiet(rdrandinf(Y, R, wl = -1, wr = 1, p = p, kernel = kernel,
                          nulltau = tau, reps = 29, vce = "HC2"))
    equal(out$asy.pvalue, 2*pnorm(-abs((coef(fit)['D']-tau)/se)))
  }
}

# Calls with fixed seeds remain reproducible and preserve the caller's RNG state.
set.seed(902)
before <- .Random.seed
one <- quiet(rdrandinf(Y, R, wl = -1, wr = 1, reps = 29, seed = 10))
stopifnot(identical(before, .Random.seed))
two <- quiet(rdrandinf(Y, R, wl = -1, wr = 1, reps = 29, seed = 10))
equal(one$p.value, two$p.value)
cat('Numerical regression checks passed.\n')
