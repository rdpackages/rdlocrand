make_regression_data <- function(n = 60) {
  idx <- seq_len(n)
  R <- seq(-1.5, 1.5, length.out = n)
  D <- as.numeric(R >= 0)
  T <- 0.15 + 0.65 * D + 0.05 * cos(idx)
  Y <- 1 + 0.8 * R - 0.2 * R^2 + 1.5 * (R >= 0) + sin(idx / 3)
  Y_fuzzy <- 1 + 0.8 * R - 0.2 * R^2 + 1.5 * T + sin(idx / 3)
  X <- cbind(x1 = cos(idx), x2 = sin(idx))

  list(Y = Y, Y_fuzzy = Y_fuzzy, R = R, X = X, T = T)
}

test_that("rdwinselect fixed-seed output is stable", {
  d <- make_regression_data()
  out <- rdwinselect(
    d$R, d$X,
    wmin = 0.45, wstep = 0.2, nwindows = 4,
    reps = 40, seed = 123, quietly = TRUE
  )

  expect_named(out, c("w_left", "w_right", "wlist_left", "wlist_right", "results", "summary"))
  expect_equal(c(out$w_left, out$w_right), c(-1.05, 1.05), tolerance = 1e-12)
  expect_equal(
    unname(out$results),
    matrix(
      c(
        0.25, 0.975, 0.675, 0.7,
        1, 2, 1, 1,
        1, 1, 1, 1,
        9, 13, 17, 21,
        9, 13, 17, 21,
        -0.45, -0.65, -0.85, -1.05,
        0.45, 0.65, 0.85, 1.05
      ),
      nrow = 4
    ),
    tolerance = 1e-12
  )
})

test_that("rdwinselect polynomial and asymmetric window paths are stable", {
  d <- make_regression_data()
  adjusted <- rdwinselect(
    d$R, d$X,
    wmin = 0.45, wstep = 0.2, nwindows = 4,
    p = 1, reps = 30, seed = 123, quietly = TRUE
  )
  asymmetric <- rdwinselect(
    d$R, d$X,
    obsmin = 6, wobs = 3, wasymmetric = TRUE, nwindows = 4,
    reps = 30, seed = 123, quietly = TRUE
  )

  expect_equal(c(adjusted$w_left, adjusted$w_right), c(-0.45, 0.45), tolerance = 1e-12)
  expect_equal(
    unname(adjusted$results),
    matrix(
      c(
        0.266666666666667, 0, 0.133333333333333, 0.366666666666667,
        1, 1, 2, 1,
        1, 1, 1, 1,
        9, 13, 17, 21,
        9, 13, 17, 21,
        -0.45, -0.65, -0.85, -1.05,
        0.45, 0.65, 0.85, 1.05
      ),
      nrow = 4
    ),
    tolerance = 1e-12
  )

  expect_equal(
    c(asymmetric$w_left, asymmetric$w_right),
    c(-0.686440677966102, 0.737288135593221),
    tolerance = 1e-12
  )
  expect_equal(
    asymmetric$wlist_left,
    c(-0.228813559322034, -0.38135593220339, -0.533898305084746, -0.686440677966102),
    tolerance = 1e-12
  )
  expect_equal(
    asymmetric$wlist_right,
    c(0.279661016949153, 0.432203389830509, 0.584745762711865, 0.737288135593221),
    tolerance = 1e-12
  )
  expect_equal(
    unname(asymmetric$results),
    matrix(
      c(
        0.666666666666667, 0.366666666666667, 0.7, 0.366666666666667,
        1, 1, 1, 2,
        1, 1, 1, 1,
        5, 8, 11, 14,
        6, 9, 12, 15,
        -0.228813559322034, -0.38135593220339, -0.533898305084746, -0.686440677966102,
        0.279661016949153, 0.432203389830509, 0.584745762711865, 0.737288135593221
      ),
      nrow = 4
    ),
    tolerance = 1e-12
  )
})

test_that("rdwinselect mass-point windows are stable", {
  R <- rep(c(-1.2, -0.8, -0.4, 0, 0.4, 0.8, 1.2), each = 6)
  idx <- seq_along(R)
  X <- cbind(x1 = sin(idx / 2), x2 = cos(idx / 3))
  out <- NULL
  invisible(capture.output(
    out <- rdwinselect(
      R, X,
      wmasspoints = TRUE, nwindows = 4,
      reps = 30, seed = 123, quietly = TRUE
    )
  ))

  expect_equal(c(out$w_left, out$w_right), c(0, 0), tolerance = 1e-12)
  expect_equal(out$wlist_left, c(0, -0.4, -0.8, -1.2), tolerance = 1e-12)
  expect_equal(out$wlist_right, c(0, 0.4, 0.8, 1.2), tolerance = 1e-12)
  expect_equal(
    unname(out$results),
    matrix(
      c(
        NA, 0.0333333333333333, 0.4, 0.166666666666667,
        NA, 1, 1, 1,
        NA, 0.237884521484375, 0.361594608053565, 0.440799067342597,
        0, 6, 12, 18,
        6, 12, 18, 24,
        0, -0.4, -0.8, -1.2,
        0, 0.4, 0.8, 1.2
      ),
      nrow = 4
    ),
    tolerance = 1e-12
  )
})

test_that("rdwinselect missing-covariate filtering is stable", {
  d <- make_regression_data()
  X <- d$X
  X[c(20, 24, 37), 1] <- NA
  X[c(25, 41), 2] <- NA
  default <- NULL
  dropmissing <- NULL
  invisible(capture.output(
    default <- rdwinselect(
      d$R, X,
      wmin = 0.45, wstep = 0.2, nwindows = 4,
      reps = 30, seed = 123, quietly = TRUE
    )
  ))
  invisible(capture.output(
    dropmissing <- rdwinselect(
      d$R, X,
      wmin = 0.45, wstep = 0.2, nwindows = 4,
      reps = 30, seed = 123, quietly = TRUE, dropmissing = TRUE
    )
  ))

  expect_equal(c(default$w_left, default$w_right), c(NA, NA))
  expect_equal(c(dropmissing$w_left, dropmissing$w_right), c(NA, NA))
  expect_equal(
    unname(default$results),
    matrix(
      c(
        0.133333333333333, 0.333333333333333, 0.166666666666667, 0.366666666666667,
        1, 1, 1, 1,
        1, 1, 1, 1,
        7, 10, 14, 18,
        8, 11, 15, 19,
        -0.45, -0.65, -0.85, -1.05,
        0.45, 0.65, 0.85, 1.05
      ),
      nrow = 4
    ),
    tolerance = 1e-12
  )
  expect_equal(unname(dropmissing$results), unname(default$results), tolerance = 1e-12)
  expect_equal(
    unname(default$summary),
    matrix(c(30, 0, 1, 3, 5, 30, 0, 1, 3, 5), nrow = 5),
    tolerance = 1e-12
  )
  expect_equal(
    unname(dropmissing$summary),
    matrix(c(27, 0, 1, 3, 6, 28, 0, 1, 3, 5), nrow = 5),
    tolerance = 1e-12
  )
})

test_that("rdrandinf fixed-seed sharp RD outputs are stable", {
  d <- make_regression_data()
  out <- rdrandinf(
    d$Y, d$R,
    wl = -0.85, wr = 0.85, statistic = "all",
    reps = 40, seed = 123, quietly = TRUE
  )

  expect_equal(
    out$obs.stat,
    c(2.1434017332736, 0.823529411764706, -4.73598564813259),
    tolerance = 1e-12
  )
  expect_equal(out$p.value, c(0, 0, 0), tolerance = 1e-12)
  expect_equal(
    out$asy.pvalue,
    c(9.04837451098905e-13, 5.12854306670472e-06, 2.17993015072857e-06),
    tolerance = 1e-12
  )
})

test_that("rdrandinf Bernoulli assignment and interference CI outputs are stable", {
  d <- make_regression_data()
  out <- rdrandinf(
    d$Y, d$R,
    wl = -0.85, wr = 0.85,
    bernoulli = rep(0.55, length(d$R)),
    interfci = 0.1,
    reps = 40, seed = 123, quietly = TRUE
  )

  expect_equal(out$obs.stat, 2.1434017332736, tolerance = 1e-12)
  expect_equal(out$p.value, 0, tolerance = 1e-12)
  expect_equal(out$asy.pvalue, 9.04837451098905e-13, tolerance = 1e-12)
  expect_equal(out$interf.ci, c(1.35731887129149, 2.85658648104015), tolerance = 1e-12)
})

test_that("rdrandinf confidence interval inversion output is stable", {
  d <- make_regression_data()
  out <- rdrandinf(
    d$Y, d$R,
    wl = -0.85, wr = 0.85,
    ci = c(0.1, 0, 1, 2),
    reps = 40, seed = 123, quietly = TRUE
  )

  expect_equal(out$obs.stat, 2.1434017332736, tolerance = 1e-12)
  expect_equal(out$p.value, 0, tolerance = 1e-12)
  expect_equal(out$ci, matrix(c(2, 2), nrow = 1), tolerance = 1e-12)
})

test_that("rdrandinf polynomial adjustment and fuzzy TSLS outputs are stable", {
  d <- make_regression_data()
  adjusted <- rdrandinf(
    d$Y, d$R,
    wl = -0.85, wr = 0.85, p = 1,
    reps = 40, seed = 123, quietly = TRUE
  )
  fuzzy <- suppressWarnings(rdrandinf(
    d$Y_fuzzy, d$R,
    wl = -0.85, wr = 0.85, fuzzy = c(d$T, "tsls"),
    reps = 40, seed = 123, quietly = TRUE
  ))

  expect_equal(adjusted$obs.stat, -0.136158326342138, tolerance = 1e-12)
  expect_equal(adjusted$p.value, 0.4, tolerance = 1e-12)
  expect_equal(unname(fuzzy$obs.stat), 2.48048963071659, tolerance = 1e-12)
  expect_true(is.na(fuzzy$p.value))
  expect_equal(fuzzy$asy.pvalue, 5.53260826029061e-08, tolerance = 1e-12)
})

test_that("rdrandinf fuzzy ITT outputs are stable", {
  d <- make_regression_data()
  fuzzy_itt <- rdrandinf(
    d$Y_fuzzy, d$R,
    wl = -0.85, wr = 0.85, fuzzy = c(d$T, "itt"),
    reps = 40, seed = 123, quietly = TRUE
  )
  fuzzy_default <- rdrandinf(
    d$Y_fuzzy, d$R,
    wl = -0.85, wr = 0.85, fuzzy = d$T,
    reps = 40, seed = 123, quietly = TRUE
  )

  expect_equal(fuzzy_itt$obs.stat, 1.62770852212262, tolerance = 1e-12)
  expect_equal(fuzzy_itt$p.value, 0, tolerance = 1e-12)
  expect_equal(fuzzy_itt$asy.pvalue, 6.80322928718854e-08, tolerance = 1e-12)
  expect_equal(fuzzy_default$obs.stat, fuzzy_itt$obs.stat, tolerance = 1e-12)
  expect_equal(fuzzy_default$p.value, fuzzy_itt$p.value, tolerance = 1e-12)
  expect_equal(fuzzy_default$asy.pvalue, fuzzy_itt$asy.pvalue, tolerance = 1e-12)
})

test_that("rdsensitivity fixed-seed output is stable", {
  d <- make_regression_data()
  out <- rdsensitivity(
    d$Y, d$R,
    wlist = c(0.65, 0.85), tlist = c(0, 1, 2),
    ci = c(-0.85, 0.85),
    reps = 40, seed = 123, nodraw = TRUE, quietly = TRUE
  )

  expect_equal(
    out$results,
    matrix(c(0, 0.125, 0.1, 0, 0, 0.675), nrow = 3),
    tolerance = 1e-12
  )
  expect_equal(out$ci, matrix(c(2, 2), nrow = 1), tolerance = 1e-12)
})

test_that("confidence interval helper output is stable", {
  expect_equal(
    rdlocrand:::find_CI(c(0.1, 0.01, 0.2, 0.3, 0.01, 0.5), 0.05, 1:6),
    matrix(c(1, 1, 3, 4, 6, 6), ncol = 2, byrow = TRUE),
    tolerance = 1e-12
  )
  expect_equal(
    rdlocrand:::find_CI(c(0.1, 0.2), 0.05, c(-1, 1)),
    matrix(c(-1, 1), nrow = 1),
    tolerance = 1e-12
  )
  expect_equal(
    rdlocrand:::find_CI(c(0.01, 0.02), 0.05, c(-1, 1)),
    matrix(NA, nrow = 1, ncol = 2)
  )
})

test_that("direct Kolmogorov-Smirnov statistic helper matches ks.test", {
  x <- c(-1.2, -0.4, 0.1, 0.7, 1.5)
  y <- c(-1.1, -0.8, 0.2, 0.4, 1.0, 1.7)
  x_ties <- c(-1, -1, 0, 0.5, 1)
  y_ties <- c(-0.5, 0, 0, 1, 1.5)
  d <- make_regression_data()
  in_window <- d$R >= -0.85 & d$R <= 0.85
  D <- as.numeric(d$R[in_window] >= 0)

  expect_equal(
    rdlocrand:::ksmirnov.statistic(x, y),
    as.numeric(ks.test(x, y)$statistic),
    tolerance = 1e-12
  )
  expect_equal(
    rdlocrand:::ksmirnov.statistic(x_ties, y_ties),
    suppressWarnings(as.numeric(ks.test(x_ties, y_ties)$statistic)),
    tolerance = 1e-12
  )
  expect_equal(
    rdlocrand:::rdrandinf.model(d$Y[in_window], D, "ksmirnov", kweights = rep(1, length(D)))$statistic,
    rdlocrand:::rdrandinf.model(d$Y[in_window], D, "ksmirnov", pvalue = TRUE, kweights = rep(1, length(D)))$statistic,
    tolerance = 1e-12
  )
  expect_equal(
    rdlocrand:::rdrandinf.model(d$Y[in_window], D, "all", kweights = rep(1, length(D)))$statistic[2],
    rdlocrand:::rdrandinf.model(d$Y[in_window], D, "all", pvalue = TRUE, kweights = rep(1, length(D)))$statistic[2],
    tolerance = 1e-12
  )
})

test_that("fast Bernoulli rank-sum helper matches rdrandinf", {
  d <- make_regression_data()
  ww <- d$R >= -0.85 & d$R <= 0.85
  Yw <- d$Y[ww]
  Rw <- d$R[ww]
  prob <- seq(0.25, 0.75, length.out = length(Rw))
  ref <- rdrandinf(
    Yw, Rw,
    wl = -0.85, wr = 0.85, bernoulli = prob,
    statistic = "ranksum", reps = 50, seed = 666, quietly = TRUE
  )
  fast <- rdlocrand:::rdrandinf.bernoulli.ranksum.pvalue(
    Yw, Rw, prob,
    reps = 50, nulltau = 0, seed = 666
  )

  expect_equal(fast, ref$p.value, tolerance = 1e-12)
})

test_that("rdrbounds fixed-seed output is stable", {
  d <- make_regression_data()
  out <- NULL
  invisible(capture.output(
    out <- rdrbounds(
      d$Y, d$R,
      expgamma = 1.5, wlist = 0.65,
      reps = 20, seed = 123
    )
  ))

  expect_equal(out$p.values, 0, tolerance = 1e-12)
  expect_equal(out$lower.bound, matrix(0, nrow = 1, ncol = 1), tolerance = 1e-12)
  expect_equal(out$upper.bound, matrix(0, nrow = 1, ncol = 1), tolerance = 1e-12)
})

test_that("rdrbounds one-sided bound outputs are stable", {
  d <- make_regression_data()
  upper <- NULL
  lower <- NULL
  invisible(capture.output(
    upper <- rdrbounds(
      d$Y, d$R,
      expgamma = 1.5, wlist = 0.65, bound = "upper",
      reps = 12, seed = 123
    )
  ))
  invisible(capture.output(
    lower <- rdrbounds(
      d$Y, d$R,
      expgamma = 1.5, wlist = 0.65, bound = "lower",
      reps = 12, seed = 123
    )
  ))

  expect_equal(upper$p.values, 0, tolerance = 1e-12)
  expect_equal(upper$upper.bound, matrix(0, nrow = 1, ncol = 1), tolerance = 1e-12)
  expect_equal(lower$p.values, 0, tolerance = 1e-12)
  expect_equal(lower$lower.bound, matrix(0, nrow = 1, ncol = 1), tolerance = 1e-12)
})

expect_rng_unchanged <- function(expr) {
  set.seed(1900)
  before <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  force(expr)
  expect_identical(get(".Random.seed", envir = .GlobalEnv, inherits = FALSE), before)
}

test_that("seeded calls restore the caller RNG state", {
  d <- make_regression_data()

  expect_rng_unchanged(
    rdwinselect(
      d$R, d$X,
      wmin = 0.45, wstep = 0.2, nwindows = 2,
      reps = 5, seed = 123, quietly = TRUE
    )
  )

  expect_rng_unchanged(
    rdrandinf(
      d$Y, d$R,
      wl = -0.85, wr = 0.85,
      reps = 5, seed = 123, quietly = TRUE
    )
  )

  expect_rng_unchanged(
    rdsensitivity(
      d$Y, d$R,
      wlist = 0.65, tlist = c(0, 1),
      reps = 5, seed = 123, nodraw = TRUE, quietly = TRUE
    )
  )

  expect_rng_unchanged(
    invisible(capture.output(
      rdrbounds(
        d$Y, d$R,
        expgamma = 1.5, wlist = 0.65,
        reps = 5, seed = 123
      )
    ))
  )

  set.seed(1901)
  before <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
  invisible(
    rdrandinf(
      d$Y, d$R,
      wl = -0.85, wr = 0.85,
      reps = 5, seed = -1, quietly = TRUE
    )
  )
  expect_identical(get(".Random.seed", envir = .GlobalEnv, inherits = FALSE), before)
})

test_that("choice validation gives direct errors for invalid options", {
  d <- make_regression_data()

  expect_error(
    rdwinselect(d$R, d$X, statistic = "bad", quietly = TRUE),
    "bad not a valid statistic",
    fixed = TRUE
  )
  expect_error(
    rdrandinf(d$Y, d$R, wl = -0.85, wr = 0.85, kernel = "bad", quietly = TRUE),
    "bad not a valid kernel",
    fixed = TRUE
  )
  expect_error(
    invisible(capture.output(rdrbounds(d$Y, d$R, wlist = 0.65, bound = "bad"))),
    "bound option incorrectly specified",
    fixed = TRUE
  )
})
