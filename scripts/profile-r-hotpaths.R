#!/usr/bin/env Rscript

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
keep_profiles <- "--keep" %in% args
quick <- "--quick" %in% args

find_repo_root <- function(path = getwd()) {
  path <- normalizePath(path, winslash = "/", mustWork = TRUE)
  repeat {
    if (file.exists(file.path(path, "R", "rdlocrand", "DESCRIPTION"))) return(path)
    parent <- dirname(path)
    if (identical(parent, path)) stop("Could not find rdlocrand repository root.", call. = FALSE)
    path <- parent
  }
}

repo_root <- find_repo_root()
profile_dir <- if (keep_profiles) {
  file.path(repo_root, "tmp", "r-profiles")
} else {
  tempfile("rdlocrand-r-profiles-")
}
dir.create(profile_dir, recursive = TRUE, showWarnings = FALSE)

load_rdlocrand_source <- function() {
  pkg_r <- file.path(repo_root, "R", "rdlocrand", "R")
  source(file.path(pkg_r, "rdlocrand_fun.R"), local = .GlobalEnv)
  source(file.path(pkg_r, "rdwinselect.R"), local = .GlobalEnv)
  source(file.path(pkg_r, "rdrandinf.R"), local = .GlobalEnv)
  source(file.path(pkg_r, "rdsensitivity.R"), local = .GlobalEnv)
  source(file.path(pkg_r, "rdrbounds.R"), local = .GlobalEnv)
}

quiet <- function(expr) {
  value <- NULL
  invisible(capture.output(value <- force(expr)))
  value
}

make_regression_data <- function(n = 100) {
  idx <- seq_len(n)
  R <- seq(-1.5, 1.5, length.out = n)
  D <- as.numeric(R >= 0)
  T <- 0.15 + 0.65 * D + 0.05 * cos(idx)
  Y <- 1 + 0.8 * R - 0.2 * R^2 + 1.5 * (R >= 0) + sin(idx / 3)
  Y_fuzzy <- 1 + 0.8 * R - 0.2 * R^2 + 1.5 * T + sin(idx / 3)
  X <- cbind(x1 = cos(idx), x2 = sin(idx), x3 = cos(idx / 4))

  list(Y = Y, Y_fuzzy = Y_fuzzy, R = R, X = X, T = T)
}

top_table <- function(table, rows = 8) {
  if (is.null(table) || !nrow(table)) return(table)
  table[seq_len(min(rows, nrow(table))), , drop = FALSE]
}

run_profile <- function(name, fun, interval = 0.005) {
  profile_file <- file.path(profile_dir, paste0(name, ".Rprof"))

  invisible(fun())
  gc()

  value <- NULL
  elapsed <- system.time({
    Rprof(profile_file, interval = interval)
    tryCatch(
      value <- fun(),
      finally = Rprof(NULL)
    )
  })

  summary <- summaryRprof(profile_file)

  cat("\n== ", name, " ==\n", sep = "")
  cat("Elapsed seconds: ", sprintf("%.3f", unname(elapsed[["elapsed"]])), "\n", sep = "")
  cat("Profile samples: ", summary$sampling.time / summary$sample.interval, "\n", sep = "")
  cat("\nTop by self time:\n")
  print(top_table(summary$by.self))
  cat("\nTop by total time:\n")
  print(top_table(summary$by.total))

  invisible(value)
}

load_rdlocrand_source()
d <- make_regression_data(if (quick) 80 else 120)

ri_reps <- if (quick) 80 else 200
rdw_reps <- if (quick) 50 else 120
bounds_reps <- if (quick) 8 else 20
sensitivity_reps <- if (quick) 30 else 70

cat("Repository: ", repo_root, "\n", sep = "")
cat("Profile traces: ", profile_dir, "\n", sep = "")
if (!keep_profiles) {
  cat("Profile traces are temporary; pass --keep to save them under tmp/r-profiles.\n")
}

run_profile("rdrandinf_diffmeans", function() quiet(rdrandinf(
  d$Y, d$R,
  wl = -0.85, wr = 0.85,
  reps = ri_reps, seed = 123, quietly = TRUE
)))

run_profile("rdrandinf_all", function() quiet(rdrandinf(
  d$Y, d$R,
  wl = -0.85, wr = 0.85, statistic = "all",
  reps = ri_reps, seed = 123, quietly = TRUE
)))

run_profile("rdwinselect_balance", function() quiet(rdwinselect(
  d$R, d$X,
  wmin = 0.35, wstep = 0.15, nwindows = 5,
  reps = rdw_reps, seed = 123, quietly = TRUE
)))

run_profile("rdsensitivity_grid", function() quiet(rdsensitivity(
  d$Y, d$R,
  wlist = c(0.65, 0.85), tlist = c(0, 1, 2),
  reps = sensitivity_reps, seed = 123, nodraw = TRUE, quietly = TRUE
)))

run_profile("rdrbounds_both", function() quiet(rdrbounds(
  d$Y, d$R,
  expgamma = 1.5, wlist = 0.65,
  reps = bounds_reps, seed = 123
)))

if (!keep_profiles) unlink(profile_dir, recursive = TRUE, force = TRUE)
