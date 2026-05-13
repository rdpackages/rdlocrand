#!/usr/bin/env Rscript

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
record <- "--record" %in% args

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
replication_root <- Sys.getenv("RDLOCRAND_REPLICATION_ROOT")
if (!nzchar(replication_root)) {
  replication_root <- file.path(dirname(dirname(repo_root)), "rdpackages-replication")
}
replication_root <- normalizePath(replication_root, winslash = "/", mustWork = FALSE)

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

round_numeric <- function(x, digits = 14) {
  if (is.numeric(x)) return(round(x, digits))
  if (is.list(x)) return(lapply(x, round_numeric, digits = digits))
  x
}

run_cases <- function() {
  ctv_dir <- file.path(replication_root, "CTV_2017_JPAM")
  cit_dir <- file.path(replication_root, "CIT_2024_CUP")
  required <- c(
    file.path(ctv_dir, "headstart.csv"),
    file.path(cit_dir, "CIT_2024_CUP_locrand.csv"),
    file.path(cit_dir, "CIT_2024_CUP_fuzzy.csv")
  )
  missing <- required[!file.exists(required)]
  if (length(missing)) {
    stop("Missing replication file(s):\n", paste(missing, collapse = "\n"), call. = FALSE)
  }

  ctv <- read.csv(file.path(ctv_dir, "headstart.csv"))
  cutoff <- 59.1984
  Y <- ctv$mort_age59_related_postHS
  R <- ctv$povrate60 - cutoff
  X60 <- cbind(
    ctv$census1960_pop,
    ctv$census1960_pctsch1417,
    ctv$census1960_pctsch534,
    ctv$census1960_pctsch25plus,
    ctv$census1960_pop1417,
    ctv$census1960_pop534,
    ctv$census1960_pop25plus,
    ctv$census1960_pcturban,
    ctv$census1960_pctblack
  )
  Xrdw <- cbind(ctv$mort_age59_related_preHS, X60)

  ctv_rdwinselect <- quiet(rdwinselect(
    R, Xrdw,
    reps = 30, statistic = "ksmirnov",
    wmin = 0.3, wstep = 0.2, nwindows = 5,
    level = 0.2, seed = 666, quietly = TRUE
  ))
  ctv_rdrandinf <- rdrandinf(
    Y, R,
    wl = -1.1, wr = 1.1,
    reps = 50, seed = 666, quietly = TRUE
  )
  ctv_rdrandinf_p1 <- rdrandinf(
    Y, R,
    wl = -1.1, wr = 1.1, p = 1,
    reps = 50, seed = 666, quietly = TRUE
  )

  cit_locrand <- read.csv(file.path(cit_dir, "CIT_2024_CUP_locrand.csv"))
  Z_locrand <- cit_locrand[, c(
    "presdemvoteshlag1", "demvoteshlag1", "demvoteshlag2",
    "demwinprv1", "demwinprv2", "dmidterm", "dpresdem", "dopen"
  )]
  cit_locrand_ri <- rdrandinf(
    cit_locrand$Y, cit_locrand$X,
    wl = -2.5, wr = 2.5,
    reps = 50, seed = 50, quietly = TRUE
  )
  cit_locrand_rdw <- quiet(rdwinselect(
    cit_locrand$X, Z_locrand,
    seed = 50, wobs = 2, reps = 50,
    nwindows = 10, quietly = TRUE
  ))

  cit_fuzzy <- read.csv(file.path(cit_dir, "CIT_2024_CUP_fuzzy.csv"))
  cit_fuzzy_tsls <- suppressWarnings(rdrandinf(
    cit_fuzzy$Y, cit_fuzzy$X1,
    wl = -0.13000107, wr = 0.13000107,
    fuzzy = c(cit_fuzzy$D, "tsls"),
    reps = 50, seed = 50, quietly = TRUE
  ))

  round_numeric(list(
    CTV_2017_JPAM = list(
      rdwinselect = list(
        window = c(ctv_rdwinselect$w_left, ctv_rdwinselect$w_right),
        first_rows = ctv_rdwinselect$results[1:3, , drop = FALSE]
      ),
      rdrandinf = list(
        obs = ctv_rdrandinf$obs.stat,
        p = ctv_rdrandinf$p.value,
        asy = ctv_rdrandinf$asy.pvalue
      ),
      rdrandinf_p1 = list(
        obs = ctv_rdrandinf_p1$obs.stat,
        p = ctv_rdrandinf_p1$p.value,
        asy = ctv_rdrandinf_p1$asy.pvalue
      )
    ),
    CIT_2024_CUP = list(
      locrand_rdrandinf = list(
        obs = cit_locrand_ri$obs.stat,
        p = cit_locrand_ri$p.value,
        asy = cit_locrand_ri$asy.pvalue
      ),
      locrand_rdwinselect = list(
        window = c(cit_locrand_rdw$w_left, cit_locrand_rdw$w_right),
        first_rows = cit_locrand_rdw$results[1:3, , drop = FALSE]
      ),
      fuzzy_tsls = list(
        obs = unname(cit_fuzzy_tsls$obs.stat),
        p = cit_fuzzy_tsls$p.value,
        asy = cit_fuzzy_tsls$asy.pvalue
      )
    )
  ))
}

compare_values <- function(actual, expected, path = character(), tolerance = 1e-12) {
  label <- paste(path, collapse = "$")
  if (!length(path)) label <- "<root>"

  if (is.list(expected)) {
    if (!is.list(actual)) stop("Type mismatch at ", label, call. = FALSE)
    if (!identical(names(actual), names(expected))) {
      stop("Name mismatch at ", label, call. = FALSE)
    }
    for (name in names(expected)) {
      compare_values(actual[[name]], expected[[name]], c(path, name), tolerance)
    }
    return(invisible(TRUE))
  }

  if (is.numeric(expected)) {
    if (!is.numeric(actual)) stop("Type mismatch at ", label, call. = FALSE)
    if (!isTRUE(all.equal(actual, expected, tolerance = tolerance, check.attributes = TRUE))) {
      stop("Numerical mismatch at ", label, call. = FALSE)
    }
    return(invisible(TRUE))
  }

  if (!identical(actual, expected)) stop("Value mismatch at ", label, call. = FALSE)
  invisible(TRUE)
}

expected <- list(
  CTV_2017_JPAM = list(
    rdwinselect = list(
      window = c(-0.9, 0.9),
      first_rows = structure(
        c(
          0.4, 0.26666666666667, 0.2,
          5, 8, 7,
          1, 0.86416624044068, 0.8829959121224,
          9, 18, 24,
          10, 16, 22,
          -0.3, -0.5, -0.7,
          0.3, 0.5, 0.7
        ),
        dim = c(3L, 7L),
        dimnames = list(NULL, c(
          "p-value", "Variable", "Bi.test", "Obs<c", "Obs>=c",
          "w_left", "w_right"
        ))
      )
    ),
    rdrandinf = list(
      obs = -2.2798226248062,
      p = 0,
      asy = 0.00451554860005
    ),
    rdrandinf_p1 = list(
      obs = -2.51513150711625,
      p = 0,
      asy = 0.16045004659952
    )
  ),
  CIT_2024_CUP = list(
    locrand_rdrandinf = list(
      obs = 9.16713411946533,
      p = 0,
      asy = 1.1935968e-07
    ),
    locrand_rdwinselect = list(
      window = c(-0.84848404, 0.84848404),
      first_rows = structure(
        c(
          0.26, 0.26, 0.42,
          3, 3, 7,
          0.2295229434967, 0.26493089646101, 0.29620636859909,
          9, 11, 13,
          16, 18, 20,
          -0.48433244, -0.56300163, -0.63778758,
          0.48433244, 0.56300163, 0.63778758
        ),
        dim = c(3L, 7L),
        dimnames = list(NULL, c(
          "p-value", "Variable", "Bi.test", "Obs<c", "Obs>=c",
          "w_left", "w_right"
        ))
      )
    ),
    fuzzy_tsls = list(
      obs = 0.29861111111111,
      p = NA,
      asy = 0.03835434080977
    )
  )
)

cat("Repository:       ", repo_root, "\n", sep = "")
cat("Replication root: ", replication_root, "\n", sep = "")
load_rdlocrand_source()
actual <- run_cases()

if (record || is.null(expected)) {
  cat("\nCurrent replication baseline:\n")
  dput(actual)
  quit(status = 0)
}

compare_values(actual, expected)
cat("\nR replication baseline checks passed.\n")
