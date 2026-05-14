###############################################################################
# rdsensitivity: sensitivity analysis for randomization inference in RD
# !version 2.0 14-May-2026
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

#' Sensitivity analysis for RD designs under local randomization
#'
#' \code{rdsensitivity} analyzes the sensitivity of randomization p-values
#' and confidence intervals to different window lengths.
#'
#'
#' @author
#' Matias D. Cattaneo, Princeton University. \email{matias.d.cattaneo@gmail.com}
#'
#' Rocio Titiunik, Princeton University. \email{rocio.titiunik@gmail.com}
#'
#' Gonzalo Vazquez-Bare, UC Santa Barbara. \email{gvazquezbare@gmail.com}
#'
#' @references
#'
#' Cattaneo, M.D., B. Frandsen and R. Titiunik. (2015). \href{https://rdpackages.github.io/references/Cattaneo-Frandsen-Titiunik_2015_JCI.pdf}{Randomization Inference in the Regression Discontinuity Design: An Application to Party Advantages in the U.S. Senate}. \emph{Journal of Causal Inference} 3(1): 1-24.
#'
#' Cattaneo, M.D., R. Titiunik and G. Vazquez-Bare. (2016). \href{https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf}{Inference in Regression Discontinuity Designs under Local Randomization}. \emph{Stata Journal} 16(2): 331-367.
#'
#' Cattaneo, M.D., R. Titiunik and G. Vazquez-Bare. (2017). \href{https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2017_JPAM.pdf}{Comparing Inference Approaches for RD Designs: A Reexamination of the Effect of Head Start on Child Mortality}. \emph{Journal of Policy Analysis and Management} 36(3): 643-681.
#'
#' @param Y a vector containing the values of the outcome variable.
#' @param R a vector containing the values of the running variable.
#' @param cutoff the RD cutoff (default is 0).
#' @param wlist the list of windows to the right of the cutoff. By default the program constructs 10 windows around the cutoff with 5 observations each.
#' @param wlist_left the list of windows to the left of the cutoff. If not specified, the windows are constructed symmetrically around the cutoff based on the values in wlist.
#' @param tlist the list of treatment-effect values under the null to be evaluated. By default the program uses ten evenly spaced points within the asymptotic confidence interval for a constant treatment effect in the smallest window to be used.
#' @param statistic the randomization test statistic to be used. Allowed options are \code{diffmeans} (difference in means statistic), \code{ksmirnov} (Kolmogorov-Smirnov statistic), and \code{ranksum} (Wilcoxon-Mann-Whitney standardized statistic). Default option is \code{diffmeans}. The statistic \code{ttest} is equivalent to \code{diffmeans} and included for backward compatibility.
#' @param p the order of the polynomial for the outcome adjustment model. Default is 0.
#' @param evalat specifies the point at which the adjusted variable is evaluated. Allowed options are \code{cutoff} and \code{means}. Default is \code{cutoff}.
#' @param kernel specifies the type of kernel to use as a weighting scheme. Allowed kernel types are \code{uniform} (uniform kernel), \code{triangular} (triangular kernel), and \code{epan} (Epanechnikov kernel). Default is \code{uniform}.
#' @param fuzzy indicates that the RD design is fuzzy. \code{fuzzy} should be specified as a vector containing the values of the endogenous treatment variable. This option uses an Anderson-Rubin/intention-to-treat statistic.
#' @param ci returns the confidence interval corresponding to the indicated window length. \code{ci} must be a two-element vector containing the left and right limits of the window. Default alpha is .05 (95\% level CI).
#' @param ci_alpha specifies the value of alpha for the confidence interval. Default alpha is .05 (95\% level CI).
#' @param reps the number of replications. Default is 1000.
#' @param seed the seed to be used for the randomization tests.
#' @param nodraw suppresses contour plot.
#' @param quietly suppresses the output table.
#'
#' @return
#' A list containing:
#' \item{tlist}{treatment-effect grid.}
#' \item{wlist}{right endpoints of the window grid.}
#' \item{wlist_left}{left endpoints of the window grid.}
#' \item{results}{matrix of p-values for each treatment-effect and window pair.}
#' \item{ci}{confidence interval; included only when \code{ci} is specified.}
#'
#' @examples
#' # Toy dataset
#' set.seed(123)
#' R <- runif(100,-1,1)
#' Y <- 1 + R -.5*R^2 + .3*R^3 + (R>=0) + rnorm(100)
#' # Sensitivity analysis
#' # Note: low number of replications to speed up process.
#' # The user should increase the number of replications.
#' tmp <- rdsensitivity(Y,R,wlist=seq(.75,2,by=.25),tlist=seq(0,5,by=1),
#'                      reps=500,nodraw=TRUE,quietly=TRUE)
#'
#'
#' @export


rdsensitivity <- function(Y,R,
                          cutoff = 0,
                          wlist,
                          wlist_left,
                          tlist,
                          statistic = 'diffmeans',
                          p = 0,
                          evalat = 'cutoff',
                          kernel = 'uniform',
                          fuzzy = NULL,
                          ci = NULL,
                          ci_alpha = 0.05,
                          reps = 1000,
                          seed = 666,
                          nodraw = FALSE,
                          quietly = FALSE){


  ###############################################################################
  # Parameters and error checking
  ###############################################################################

  if (cutoff<min(R,na.rm=TRUE) | cutoff>max(R,na.rm=TRUE)) stop('Cutoff must be within the range of the running variable')
  rdlocrand_validate_choice(
    statistic,
    c('diffmeans','ttest','ksmirnov','ranksum'),
    paste(paste(statistic, collapse = ', '),'not a valid statistic')
  )
  rdlocrand_validate_choice(evalat, c('cutoff','means'), 'evalat only admits means or cutoff')
  rdlocrand_validate_choice(
    kernel,
    c('uniform','triangular','epan'),
    paste(paste(kernel, collapse = ', '),'not a valid kernel')
  )
  if (missing(tlist) & p!=0) stop('need to specify tlist when p>0')
  if (!missing(wlist_left)){
    if (missing(wlist)) stop('Need to specify wlist when wlist_left is specified')
    if (length(wlist)!=length(wlist_left)) stop('Lengths of wlist and wlist_left need to coincide')
  }
  if(!is.null(ci) & length(ci)!=2) stop('Need to specify wleft and wright in CI option')

  restore_rng <- rdlocrand_seed_scope(seed)
  on.exit(restore_rng(), add = TRUE)

  data <- cbind(Y,R)
  data <- data[complete.cases(data),]
  Y <- data[,1]
  R <- data[,2]

  Rc <- R - cutoff


  ###############################################################################
  # Default window list
  ###############################################################################

  if (missing(wlist)){
    aux <- rdwinselect(Rc,wobs=5,quietly=TRUE)
    wlist <- aux$results[,7]
    wlist <- aux$results[,6]
  } else{
    wlist_orig <- wlist
    wlist <- wlist - cutoff
    if(missing(wlist_left)){
      wlist_left <- -wlist
      wlist_left_orig <- wlist_left
    } else{
      wlist_left_orig <- wlist_left
      wlist_left <- wlist_left - cutoff
    }
  }

  wnum <- length(wlist)

  ###############################################################################
  # Default tau list
  ###############################################################################

  if (missing(tlist)){
    D <- as.numeric(Rc >= 0)
    wfirst <- max(wlist[1],abs(wlist_left[1]))
    if (is.null(fuzzy)){
      Yaux <- Y[abs(Rc)<=wfirst]
      Daux <- D[abs(Rc)<=wfirst]
      aux <- lm(Yaux ~ Daux)
      ci.ub <- round(aux$coefficients['Daux']+1.96*sqrt(vcov(aux)['Daux','Daux']),2)
      ci.lb <- round(aux$coefficients['Daux']-1.96*sqrt(vcov(aux)['Daux','Daux']),2)
    } else {
      Yaux <- Y[abs(Rc)<=wfirst]
      Daux <- D[abs(Rc)<=wfirst]
      Taux <- fuzzy[abs(Rc)<=wfirst]
      aux <- AER::ivreg(Yaux ~ Taux,~Daux)
      ci.ub <- round(aux$coefficients[2]+1.96*sqrt(vcov(aux)[2,2]),2)
      ci.lb <- round(aux$coefficients[2]-1.96*sqrt(vcov(aux)[2,2]),2)
    }

    wstep <- round((ci.ub-ci.lb)/10,2)
    tlist <- seq(ci.lb,ci.ub,by=wstep)
  }


  ###############################################################################
  # Sensitivity analysis
  ###############################################################################

  results <- array(NA,dim=c(length(tlist),length(wlist)))
  if (quietly==FALSE) {cat('\nRunning sensitivity analysis...\n')}

  row <- 1
  for (t in tlist){
    for (w in 1:wnum){
      wright <- wlist[w]
      wleft <- wlist_left[w]
      if (evalat=='means'){
        ww <- (round(Rc,8) >= round(wleft,8)) & (round(Rc,8) <= round(wright,8))
        Rw <- R[ww]
        Dw <- D[ww]
        evall <- mean(Rw[Dw==0])
        evalr <- mean(Rw[Dw==1])
      } else{
        evall <- NULL
        evalr <- NULL
      }

      aux <- rdrandinf(Y,Rc,wl=wleft,wr=wright,p=p,reps=reps,nulltau=t,
                       statistic=statistic,kernel=kernel,evall=evall,evalr=evalr,
                       fuzzy=fuzzy,seed=seed,quietly=TRUE)
      results[row,w] <- aux$p.value
    }
    row <- row + 1
  }
  if (quietly==FALSE) cat('Sensitivity analysis complete.\n')


  ###############################################################################
  # Confidence interval
  ###############################################################################

  if (!is.null(ci)){
    ci.window.l <- ci[1] - cutoff
    ci.window.r <- ci[2] - cutoff

    if (is.element(ci.window.r,wlist)==TRUE & is.element(ci.window.l,wlist_left)==TRUE){
      col <- which(wlist==ci.window.r)
      aux <- results[,col]

      conf.int <- find_CI(aux,ci_alpha,tlist)
      rownames(conf.int) <- NULL

    } else{
      stop('window specified in ci not in wlist')
    }
  }


  ###############################################################################
  # Output
  ###############################################################################

  if (is.null(ci)){
    output <- list(tlist = tlist, wlist = wlist_orig, wlist_left = wlist_left_orig, results = results)
  } else {
    output <- list(tlist = tlist, wlist = wlist_orig, wlist_left = wlist_left_orig, results = results, ci = conf.int)
  }


  ###############################################################################
  # Plot
  ###############################################################################

  if (nodraw==FALSE){
    if (dim(results)[2]==1){
      warning('need a window grid to draw plot')
    } else if (dim(results)[1]==1){
      warning('need a tau grid to draw plot')
    } else {
      filled.contour(wlist,tlist,t(results),
                     xlab='window',ylab='treatment effect',
                     key.title=title(main = 'p-value',cex.main=.8),
                     levels=seq(0,1,by=.01),col=gray.colors(100,1,0))

    }
  }

  return(output)

}
