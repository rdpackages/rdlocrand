#!/usr/bin/env Rscript

options(warn = 1)

args <- commandArgs(trailingOnly = TRUE)
mode <- "release"
if ("--dev" %in% args) mode <- "dev"
if ("--pre-push" %in% args) mode <- "pre-push"
if ("--release" %in% args) mode <- "release"
keep_artifacts <- "--keep-artifacts" %in% args

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
pkg_dir <- file.path(repo_root, "R", "rdlocrand")

add_common_paths <- function() {
  if (.Platform$OS.type == "windows") {
    Sys.unsetenv(c("LC_ALL", "LC_CTYPE"))
    r_minor <- sub("\\..*", "", R.version$minor)
    local_app_data <- chartr("\\", "/", Sys.getenv("LOCALAPPDATA"))
    user_lib <- file.path(local_app_data, "R", "win-library",
                          paste(R.version$major, r_minor, sep = "."))
    if (dir.exists(user_lib)) {
      Sys.setenv(R_LIBS_USER = user_lib)
      Sys.setenv(R_LIBS = user_lib)
      .libPaths(c(user_lib, .libPaths()))
    }
  }
  if (.Platform$OS.type == "windows" && !nzchar(Sys.which("pandoc"))) {
    pandoc_dir <- file.path(Sys.getenv("LOCALAPPDATA"), "Pandoc")
    pandoc_exe <- file.path(pandoc_dir, "pandoc.exe")
    if (file.exists(pandoc_exe)) {
      Sys.setenv(PATH = paste(pandoc_dir, Sys.getenv("PATH"), sep = .Platform$path.sep))
    }
  }
}

run <- function(command, args = character(), wd = repo_root) {
  cat("\n==>", command, paste(args, collapse = " "), "\n")
  oldwd <- getwd()
  on.exit(setwd(oldwd), add = TRUE)
  setwd(wd)
  status <- system2(command, args)
  if (!identical(status, 0L)) {
    stop(sprintf("Command failed with status %s: %s", status, command), call. = FALSE)
  }
  invisible(TRUE)
}

run_r <- function(expr, wd = repo_root) {
  run(file.path(R.home("bin"), "Rscript"), c("-e", expr), wd = wd)
}

require_packages <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) {
    stop(
      "Missing required package(s): ", paste(missing, collapse = ", "),
      "\nInstall them in your user library before running this check.",
      call. = FALSE
    )
  }
}

clean_artifacts <- function() {
  for (attempt in seq_len(5)) {
    paths <- c(
      file.path(repo_root, "check_as_cran"),
      file.path(repo_root, "rdlocrand.Rcheck"),
      Sys.glob(file.path(repo_root, "rdlocrand_*.tar.gz"))
    )
    paths <- paths[file.exists(paths)]
    if (!length(paths)) return(invisible(TRUE))
    unlink(paths, recursive = TRUE, force = TRUE)
    if (!any(file.exists(paths))) return(invisible(TRUE))
    Sys.sleep(1)
  }
  remaining <- paths[file.exists(paths)]
  if (length(remaining)) {
    message("Could not remove generated artifact(s): ", paste(remaining, collapse = ", "))
  }
  invisible(FALSE)
}

build_tarball <- function() {
  run(file.path(R.home("bin"), "R"), c("CMD", "build", "R/rdlocrand"))
  tarball <- Sys.glob(file.path(repo_root, "rdlocrand_*.tar.gz"))
  if (length(tarball) != 1L) stop("Expected exactly one rdlocrand source tarball.", call. = FALSE)
  tarball
}

add_common_paths()

cat("Repository:", repo_root, "\n")
cat("Package:   ", pkg_dir, "\n")
cat("Mode:      ", mode, "\n")
cat("Pandoc:    ", if (nzchar(Sys.which("pandoc"))) Sys.which("pandoc") else "<not found>", "\n")

if (mode == "pre-push") {
  tarball <- build_tarball()
  run(file.path(R.home("bin"), "R"), c("CMD", "check", "--no-manual", basename(tarball)))
  if (!keep_artifacts) clean_artifacts()
  cat("\nPre-push checks passed.\n")
  quit(status = 0)
}

require_packages("roxygen2")
run_r("roxygen2::roxygenise()", wd = pkg_dir)

test_dir <- file.path(pkg_dir, "tests", "testthat")
if (dir.exists(test_dir)) {
  require_packages("testthat")
  run_r("testthat::test_local()", wd = pkg_dir)
} else {
  cat("\nNo testthat tests found; skipping focused R tests.\n")
}

if (mode == "dev") {
  run(file.path(R.home("bin"), "R"), c("CMD", "check", "--no-manual", "R/rdlocrand"))
  if (!keep_artifacts) clean_artifacts()
  cat("\nDevelopment checks passed.\n")
  quit(status = 0)
}

tarball <- build_tarball()

dir.create(file.path(repo_root, "check_as_cran"), showWarnings = FALSE)
run(
  file.path(R.home("bin"), "R"),
  c("CMD", "check", "--no-manual", "--as-cran", "-o", "check_as_cran", basename(tarball))
)

if (!keep_artifacts) clean_artifacts()
cat("\nRelease checks passed.\n")
