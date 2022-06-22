###############################################################################
# rdrbounds: Rosenbaum bounds for randomization inference in RD
# !version 1.0 21-Jun-2022
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

#' Rosenbaum bounds for RD designs under local randomization
#'
#' \code{rdrbounds} calculates lower and upper bounds for the
#'  randomization p-value under different degrees of departure from a
#'  local randomized experiment, as suggested by Rosenbaum (2002).
#'
#' @author
#' Matias Cattaneo, Princeton University. \email{cattaneo@princeton.edu}
#'
#' Rocio Titiunik, Princeton University. \email{titiunik@princeton.edu}
#'
#' Gonzalo Vazquez-Bare, UC Santa Barbara. \email{gvazquez@econ.ucsb.edu}
#'
#' @references
#'
#' Cattaneo, M.D., R. Titiunik and G. Vazquez-Bare. (2016). \href{https://rdpackages.github.io/references/Cattaneo-Titiunik-VazquezBare_2016_Stata.pdf}{Inference in Regression Discontinuity Designs under Local Randomization}. \emph{Stata Journal} 16(2): 331-367.
#'
#' Rosenbaum, P. (2002). Observational Studies. Springer.
#'
#'
#' @param Y a vector containing the values of the outcome variable.
#' @param R a vector containing the values of the running variable.
#' @param cutoff the RD cutoff (default is 0).
#' @param wlist the list of window lengths to be evaluated. By default the program constructs 10 windows around the cutoff, the first one including 10 treated and control observations and adding 5 observations to each group in subsequent windows.
#' @param gamma the list of values of gamma to be evaluated.
#' @param expgamma the list of values of exp(gamma) to be evaluated. Default is \code{c(1.5,2,2.5,3)}.
#' @param bound specifies which bounds the command calculates. Options are \code{upper} for upper bound, \code{lower} for lower bound and \code{both} for both upper and lower bounds. Default is \code{both}.
#' @param statistic the statistic to be used in the balance tests. Allowed options are \code{diffmeans} (difference in means statistic), \code{ksmirnov} (Kolmogorov-Smirnov statistic) and \code{ranksum} (Wilcoxon-Mann-Whitney standardized statistic). Default option is \code{diffmeans}. The statistic \code{ttest} is equivalent to \code{diffmeans} and included for backward compatibility.
#' @param p the order of the polynomial for outcome adjustment model. Default is 0.
#' @param evalat specifies the point at which the adjusted variable is evaluated. Allowed options are \code{cutoff} and \code{means}. Default is \code{cutoff}.
#' @param kernel specifies the type of kernel to use as weighting scheme. Allowed kernel types are \code{uniform} (uniform kernel), \code{triangular} (triangular kernel) and \code{epan} (Epanechnikov kernel). Default is \code{uniform}.
#' @param fuzzy indicates that the RD design is fuzzy. \code{fuzzy} can be specified as a vector containing the values of the endogenous treatment variable, or as a list where the first element is the vector of endogenous treatment values and the second element is a string containing the name of the statistic to be used. Allowed statistics are \code{ar} (Anderson-Rubin statistic) and \code{tsls} (2SLS statistic). Default statistic is \code{ar}. The \code{tsls} statistic relies on large-sample approximation.
#' @param nulltau the value of the treatment effect under the null hypothesis. Default is 0.
#' @param prob the probabilities of treatment for each unit when assignment mechanism is a Bernoulli trial. This option should be specified as a vector of length equal to the length of the outcome and running variables.
#' @param fmpval reports the p-value under fixed margins randomization, in addition to the p-value under Bernoulli trials.
#' @param reps number of replications. Default is 1000.
#' @param seed the seed to be used for the randomization tests.
#'
#' @return
#' \item{gamma}{list of gamma values.}
#' \item{expgamma}{list of exp(gamma) values.}
#' \item{wlist}{window grid.}
#' \item{p.values}{p-values for each window (under gamma = 0).}
#' \item{lower.bound}{list of lower bound p-values for each window and gamma pair.}
#' \item{upper.bound}{list of upper bound p-values for each window and gamma pair.}
#'
#' @examples
#' # Toy dataset
#' R <- runif(100,-1,1)
#' Y <- 1 + R -.5*R^2 + .3*R^3 + (R>=0) + rnorm(100)
#' # Rosenbaum bounds
#' # Note: low number of replications and windows to speed up process.
#' # The user should increase these values.
#' rdrbounds(Y,R,expgamma=c(1.5,2),wlist=c(.3),reps=100)
#'
#'
#' @export


rdrbounds = function(Y,R,
                     cutoff = 0,
                     wlist,
                     gamma,
                     expgamma,
                     bound = 'both',
                     statistic = 'ranksum',
                     p = 0,
                     evalat = 'cutoff',
                     kernel = 'uniform',
                     fuzzy = NULL,
                     nulltau = 0,
                     prob,
                     fmpval = FALSE,
                     reps = 1000,
                     seed = 666){


  ###############################################################################
  # Parameters and error checking
  ###############################################################################

  if (cutoff<=min(R,na.rm=TRUE) | cutoff>=max(R,na.rm=TRUE)) stop('Cutoff must be within the range of the running variable')
  if (bound!='both' & bound!='upper' & bound!='lower') stop('bound option incorrectly specified')

  data <- cbind(Y,R)
  data <- data[complete.cases(data),]
  Y <- data[,1]
  R <- data[,2]

  Rc <- R - cutoff
  D <- as.numeric(Rc >= 0)

  if (missing(gamma) & missing(expgamma)){
    gammalist <- c(1.5,2,2.5,3)
  }
  if (missing(gamma) & !missing(expgamma)){
    gammalist <- expgamma
  }
  if (!missing(gamma) & missing(expgamma)){
    gammalist <- exp(gamma)
  }
  if (!missing(gamma) & !missing(expgamma)){
    stop('gamma and expgamma cannot be specified simultaneously')
  }

  if (missing(wlist)){
    aux <- rdwinselect(Rc,wobs=5,nwindows=5,quietly=TRUE)
    wlist <- round(aux$results[,1],2)
  }

  if (seed>0){
    set.seed(seed)
  } else if (seed!=-1){
    stop('Seed has to be a positive integer or -1 for system seed')
  }

  evall <- cutoff
  evalr <- cutoff


  ###############################################################################
  # Randomization p-value
  ###############################################################################

  cat('\nCalculating randomization p-value...\n')
  P <- array(NA,dim=c(2,length(wlist)))

  count <- 1

  if (fmpval==FALSE){
    for (w in wlist){
      ww <- (round(Rc,8) >= round(-w,8)) & (round(Rc,8) <= round(w,8))
      Dw <- D[ww]
      Rw <- Rc[ww]
      if (missing(prob)){
        prob.be <- rep(mean(Dw),length(R))
      } else{
        prob.be <- prob
      }
      if (evalat=='means'){
        evall <- mean(Rw[Dw==0])
        evalr <- mean(Rw[Dw==1])
      }
      aux <- rdrandinf(Y,Rc,wl=-w,wr=w,bernoulli=prob.be,reps=reps,p=p,
                      nulltau=nulltau,statistic=statistic,
                      evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                      quietly=TRUE)
      P[1,count] <- aux$p.value

      cat(paste0('\nBernoulli p-value (w = ',w,') = ',round(P[1,count],3)))

      count <- count + 1
    }
  } else {
    for (w in wlist){
      ww <- (round(Rc,8) >= round(-w,8)) & (round(Rc,8) <= round(w,8))
      Dw <- D[ww]
      Rw <- Rc[ww]
      if (missing(prob)){
        prob.be <- rep(mean(Dw),length(R))
      } else{
        prob.be <- prob
      }
      if (evalat=='means'){
        evall <- mean(Rw[Dw==0])
        evalr <- mean(Rw[Dw==1])
      }
      aux.be <- rdrandinf(Y,Rc,wl=-w,wr=w,bernoulli=prob.be,reps=reps,p=p,
                         nulltau=nulltau,statistic=statistic,
                         evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                         quietly=TRUE)
      P[1,count] <- aux.be$p.value

      aux.fm <- rdrandinf(Y,Rc,wl=-w,wr=w,reps=reps,p=p,
                         nulltau=nulltau,statistic=statistic,
                         evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                         quietly=TRUE)
      P[2,count] <- aux.fm$p.value

      cat(paste0('\nBernoulli p-value (w = ',w,') = ',round(P[1,count],3)))
      cat(paste0('\nFixed margins p-value (w = ',w,') = ',round(P[2,count],3)))

      count <- count + 1

    }
  }

  cat('\n')

  ###############################################################################
  # Sensitivity analysis
  ###############################################################################


  cat('\nRunning sensitivity analysis...\n')

  count.g <- 1

  if (bound=='upper'){

    p.ub <- array(NA,dim=c(length(gammalist),length(wlist)))

    for (G in gammalist){

      plow <- 1/(1+G)
      phigh <- G/(1+G)

      count.w <- 1

      for (w in wlist){

        ww <- (round(Rc,8) >= round(-w,8)) & (round(Rc,8) <= round(w,8))
        Dw <- D[ww]
        Yw <- Y[ww]
        Rw <- Rc[ww]

        data.w <- cbind(Yw,Rw,Dw)
        jj <- order(data.w[,1],decreasing=TRUE)
        data.dec <- data.w[jj,]
        Yw.dec <- data.dec[,1]
        Rw.dec <- data.dec[,2]

        nw <- length(Rw)
        nw1 <- sum(Dw)
        nw0 <- nw - nw1
        pvals.ub <- NULL

        for (u in seq(1,nw)){

          uplus <- c(rep(1,u),rep(0,nw-u))
          p.aux <- phigh*uplus + plow*(1-uplus)
          aux <- rdrandinf(Yw.dec,Rw.dec,wl=-w,wr=w,bernoulli=p.aux,reps=reps,p=p,
                          nulltau=nulltau,statistic=statistic,
                          evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                          quietly=TRUE)
          pvals.ub <- c(pvals.ub,aux$p.value)

        }

        p.ub.w <- max(pvals.ub)
        p.ub[count.g,count.w] <- p.ub.w

        count.w <- count.w + 1

      }

      count.g <- count.g + 1

    }
  }

  if (bound=='both'){

    p.ub <- array(NA,dim=c(length(gammalist),length(wlist)))
    p.lb <- array(NA,dim=c(length(gammalist),length(wlist)))

    for (G in gammalist){

      plow <- 1/(1+G)
      phigh <- G/(1+G)

      count.w <- 1

      for (w in wlist){

        ww <- (round(Rc,8) >= round(-w,8)) & (round(Rc,8) <= round(w,8))
        Dw <- D[ww]
        Yw <- Y[ww]
        Rw <- Rc[ww]

        data.w <- cbind(Yw,Rw,Dw)
        ii <- order(data.w[,1])
        data.inc <- data.w[ii,]
        Yw.inc <- data.inc[,1]
        Rw.inc <- data.inc[,2]
        jj <- order(data.w[,1],decreasing=TRUE)
        data.dec <- data.w[jj,]
        Yw.dec <- data.dec[,1]
        Rw.dec <- data.dec[,2]

        nw <- length(Rw)
        nw1 <- sum(Dw)
        nw0 <- nw - nw1
        pvals.ub <- NULL
        pvals.lb <- NULL

        for (u in seq(1,nw)){

          uplus <- c(rep(1,u),rep(0,nw-u))
          p.aux <- phigh*uplus + plow*(1-uplus)
          aux <- rdrandinf(Yw.dec,Rw.dec,wl=-w,wr=w,bernoulli=p.aux,reps=reps,p=p,
                          nulltau=nulltau,statistic=statistic,
                          evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                          quietly=TRUE)
          pvals.ub <- c(pvals.ub,aux$p.value)
          uminus <- c(rep(0,nw-u),rep(1,u))
          p.aux <- phigh*uminus + plow*(1-uminus)
          aux <- rdrandinf(Yw.inc,Rw.inc,wl=-w,wr=w,bernoulli=p.aux,reps=reps,p=p,
                          nulltau=nulltau,statistic=statistic,
                          evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                          quietly=TRUE)
          pvals.lb <- c(pvals.lb,aux$p.value)

        }

        p.ub.w <- max(pvals.ub)
        p.lb.w <- min(pvals.lb)
        p.ub[count.g,count.w] <- p.ub.w
        p.lb[count.g,count.w] <- p.lb.w

        count.w <- count.w + 1

      }

      count.g <- count.g + 1

    }
  }

  if (bound=='lower'){

    p.lb <- array(NA,dim=c(length(gammalist),length(wlist)))

    for (G in gammalist){

      plow <- 1/(1+G)
      phigh <- G/(1+G)

      count.w <- 1

      for (w in wlist){

        ww <- (round(Rc,8) >= round(-w,8)) & (round(Rc,8) <= round(w,8))
        Dw <- D[ww]
        Yw <- Y[ww]
        Rw <- Rc[ww]

        data.w <- cbind(Yw,Rw,Dw)
        ii <- order(data.w[,1])
        data.inc <- data.w[ii,]
        Yw.inc <- data.inc[,1]
        Rw.inc <- data.inc[,2]

        nw <- length(Rw)
        nw1 <- sum(Dw)
        nw0 <- nw - nw1
        pvals.ub <- NULL
        pvals.lb <- NULL

        for (u in seq(1,nw)){

          uminus <- c(rep(0,nw-u),rep(1,u))
          p.aux <- phigh*uminus + plow*(1-uminus)
          aux <- rdrandinf(Yw.inc,Rw.inc,wl=-w,wr=w,bernoulli=p.aux,reps=reps,p=p,
                          nulltau=nulltau,statistic=statistic,
                          evall=evall,evalr=evalr,kernel=kernel,fuzzy=fuzzy,
                          quietly=TRUE)
          pvals.lb <- c(pvals.lb,aux$p.value)

        }

        p.lb.w <- min(pvals.lb)
        p.lb[count.g,count.w] <- p.lb.w

        count.w <- count.w + 1

      }

      count.g <- count.g + 1

    }
  }

  cat('\nSensitivity analysis complete.\n')


  ###############################################################################
  # Output
  ###############################################################################

  if (fmpval==FALSE){
    P <- P[-2,]
  }

  if (bound=='both'){
    output <- list(gamma = log(gammalist), expgamma = gammalist, wlist = wlist,
                  p.values = P, lower.bound = p.lb, upper.bound = p.ub)
  }

  if (bound=='upper'){
    output <- list(gamma = log(gammalist), expgamma = gammalist, wlist = wlist,
                  p.values = P, upper.bound = p.ub)
  }

  if (bound=='lower'){
    output <- list(gamma = log(gammalist), expgamma = gammalist, wlist = wlist,
                  p.values = P, lower.bound = p.lb)
  }

  return(output)

}
