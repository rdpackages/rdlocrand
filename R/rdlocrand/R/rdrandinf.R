###############################################################################
# rdrandinf: randomization inference in RD window
# !version 1.1 22-May-2025
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

#' Randomization Inference for RD Designs under Local Randomization
#'
#' \code{rdrandinf} implements randomization inference and related methods for RD designs,
#' using observations in a specified or data-driven selected window around the cutoff where
#' local randomization is assumed to hold.
#'
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
#'
#'
#' @param Y a vector containing the values of the outcome variable.
#' @param R a vector containing the values of the running variable.
#' @param cutoff the RD cutoff (default is 0).
#' @param wl the left limit of the window. The default takes the minimum of the running variable.
#' @param wr the right limit of the window. The default takes the maximum of the running variable.
#' @param statistic the statistic to be used in the balance tests. Allowed options are \code{diffmeans} (difference in means statistic), \code{ksmirnov} (Kolmogorov-Smirnov statistic) and \code{ranksum} (Wilcoxon-Mann-Whitney standardized statistic). Default option is \code{diffmeans}. The statistic \code{ttest} is equivalent to \code{diffmeans} and included for backward compatibility.
#' @param p the order of the polynomial for outcome transformation model (default is 0).
#' @param evall the point at the left of the cutoff at which to evaluate the transformed outcome is evaluated. Default is the cutoff value.
#' @param evalr specifies the point at the right of the cutoff at which the transformed outcome is evaluated. Default is the cutoff value.
#' @param kernel specifies the type of kernel to use as weighting scheme. Allowed kernel types are \code{uniform} (uniform kernel), \code{triangular} (triangular kernel) and \code{epan} (Epanechnikov kernel). Default is \code{uniform}.
#' @param fuzzy indicates that the RD design is fuzzy. \code{fuzzy} can be specified as a vector containing the values of the endogenous treatment variable, or as a list where the first element is the vector of endogenous treatment values and the second element is a string containing the name of the statistic to be used. Allowed statistics are \code{itt} (intention-to-treat statistic) and \code{tsls} (2SLS statistic). Default statistic is \code{ar}. The \code{tsls} statistic relies on large-sample approximation.
#' @param nulltau the value of the treatment effect under the null hypothesis (default is 0).
#' @param d the effect size for asymptotic power calculation. Default is 0.5 * standard deviation of outcome variable for the control group.
#' @param dscale the fraction of the standard deviation of the outcome variable for the control group used as alternative hypothesis for asymptotic power calculation. Default is 0.5.
#' @param ci calculates a confidence interval for the treatment effect by test inversion. \code{ci} can be specified as a scalar or a vector, where the first element indicates the value of alpha for the confidence interval (typically 0.05 or 0.01) and the remaining elements, if specified, indicate the grid of treatment effects to be evaluated. This option uses \code{rdsensitivity} to calculate the confidence interval. See corresponding help for details. Note: the default tlist can be narrow in some cases, which may truncate the confidence interval. We recommend the user to manually set a large enough tlist.
#' @param interfci the level for Rosenbaum's confidence interval under arbitrary interference between units.
#' @param bernoulli the probabilities of treatment for each unit when assignment mechanism is a Bernoulli trial. This option should be specified as a vector of length equal to the length of the outcome and running variables.
#' @param reps the number of replications (default is 1000).
#' @param seed the seed to be used for the randomization test.
#' @param quietly suppresses the output table.
#' @param covariates the covariates used by \code{rdwinselect} to choose the window when \code{wl} and \code{wr} are not specified. This should be a matrix of size n x k where n is the total sample size and k is the number of covariates.
#' @param obsmin the minimum number of observations above and below the cutoff in the smallest window employed by the companion command \code{rdwinselect}. Default is 10.
#' @param wmin the smallest window to be used (if \code{minobs} is not specified) by the companion command \code{rdwinselect}. Specifying both \code{wmin} and \code{obsmin} returns an error.
#' @param wobs the number of observations to be added at each side of the cutoff at each step.
#' @param wstep the increment in window length (if \code{obsstep} is not specified) by the companion command \code{rdwinselect}.  Specifying both \code{obsstep} and \code{wstep} returns an error.
#' @param wasymmetric allows for asymmetric windows around the cutoff when (\code{wobs} is specified).
#' @param wmasspoints specifies that the running variable is discrete and each masspoint should be used as a window.
#' @param nwindows the number of windows to be used by the companion command \code{rdwinselect}. Default is 10.
#' @param dropmissing drop rows with missing values in covariates when calculating windows.
#' @param rdwstat the statistic to be used by the companion command \code{rdwinselect} (see corresponding help for options). Default option is \code{ttest}.
#' @param approx forces the companion command \code{rdwinselect} to conduct the covariate balance tests using a large-sample approximation instead of finite-sample exact randomization inference methods.
#' @param rdwreps the number of replications to be used by the companion command \code{rdwinselect}. Default is 1000.
#' @param level the minimum accepted value of the p-value from the covariate balance tests to be used by the companion command \code{rdwinselect}. Default is .15.
#' @param plot draws a scatter plot of the minimum p-value from the covariate balance test against window length implemented by the companion command \code{rdwinselect}.
#' @param firststage reports the results from the first step when using tsls.
#' @param obsstep the minimum number of observations to be added on each side of the cutoff for the sequence of fixed-increment nested windows. Default is 2. This option is deprecated and only included for backward compatibility.
#'
#' @return
#' \item{sumstats}{summary statistics}
#' \item{obs.stat}{observed statistic(s)}
#' \item{p.value}{randomization p-value(s)}
#' \item{asy.pvalue}{asymptotic p-value(s)}
#' \item{window}{chosen window}
#' \item{ci}{confidence interval (only if \code{ci} option is specified)}
#' \item{interf.ci}{confidence interval under interferecen (only if \code{interfci} is specified)}
#'
#' @examples
#' # Toy dataset
#' X <- array(rnorm(200),dim=c(100,2))
#' R <- X[1,] + X[2,] + rnorm(100)
#' Y <- 1 + R -.5*R^2 + .3*R^3 + (R>=0) + rnorm(100)
#' # Randomization inference in window (-.75,.75)
#' tmp <- rdrandinf(Y,R,wl=-.75,wr=.75)
#' # Randomization inference in window (-.75,.75), all statistics
#' tmp <- rdrandinf(Y,R,wl=-.75,wr=.75,statistic='all')
#' # Randomization inference with window selection
#' # Note: low number of replications to speed up process.
#' # The user should increase the number of replications.
#' tmp <- rdrandinf(Y,R,statistic='all',covariates=X,wmin=.5,wstep=.125,rdwreps=500)
#'
#'
#'
#' @export


rdrandinf <- function(Y,R,
                      cutoff = 0,
                      wl = NULL,
                      wr = NULL,
                      statistic = 'diffmeans',
                      p = 0,
                      evall = NULL,
                      evalr = NULL,
                      kernel = 'uniform',
                      fuzzy = NULL,
                      nulltau = 0,
                      d = NULL,
                      dscale = NULL,
                      ci,
                      interfci = NULL,
                      bernoulli = NULL,
                      reps = 1000,
                      seed = 666,
                      quietly = FALSE,
                      covariates,
                      obsmin = NULL,
                      wmin = NULL,
                      wobs = NULL,
                      wstep = NULL,
                      wasymmetric = FALSE,
                      wmasspoints = FALSE,
                      nwindows = 10,
                      dropmissing = FALSE,
                      rdwstat = 'diffmeans',
                      approx = FALSE,
                      rdwreps = 1000,
                      level = .15,
                      plot = FALSE,
                      firststage = FALSE,
                      obsstep = NULL){


  ###############################################################################
  # Parameters and error checking
  ###############################################################################

  randmech <- 'fixed margins'

  Rc.long <- R - cutoff

  if (!is.null(fuzzy)){
    statistic <- ''
    if (is.numeric(fuzzy)==TRUE){
      fuzzy.stat <- 'ar'
      fuzzy.tr <- fuzzy
    } else {
      fuzzy.tr <- as.numeric(fuzzy[-length(fuzzy)])
      if (fuzzy[length(fuzzy)]=='ar' | fuzzy[length(fuzzy)]=='itt') fuzzy.stat <- 'ar'
      else if (fuzzy[length(fuzzy)]=='tsls') fuzzy.stat <- 'wald'
      else {stop('fuzzy statistic not valid')}
    }
  }
  else{fuzzy.stat  <- ''}

  if(is.null(fuzzy)){
    if(is.null(bernoulli)){
      data <- cbind(Y,R)
      data <- data[complete.cases(data),]
      Y <- data[,1]
      R <- data[,2]
    } else{
      data <- cbind(Y,R,bernoulli)
      data <- data[complete.cases(data),]
      Y <- data[,1]
      R <- data[,2]
      bernoulli <- data[,3]
    }
  }
  else {
    if(missing(bernoulli)){
      data <- cbind(Y,R,fuzzy.tr)
      data <- data[complete.cases(data),]
      Y <- data[,1]
      R <- data[,2]
      fuzzy.tr <- data[,3]
    } else{
      data <- cbind(Y,R,bernoulli,fuzzy.tr)
      data <- data[complete.cases(data),]
      Y <- data[,1]
      R <- data[,2]
      bernoulli <- data[,3]
      fuzzy.tr <- data[,4]
    }

  }

  if (cutoff<min(R,na.rm=TRUE) | cutoff>max(R,na.rm=TRUE)) stop('Cutoff must be within the range of the running variable')
  if (p<0) stop('p must be a positive integer')

  if (is.null(fuzzy)){
    if (statistic!='diffmeans' & statistic!='ttest' & statistic!='ksmirnov' & statistic!='ranksum' & statistic!='all') stop(paste(statistic,'not a valid statistic'))
  }

  if (kernel!='uniform' & kernel!='triangular' & kernel!='epan') stop(paste(kernel,'not a valid kernel'))
  if (kernel!='uniform' & !is.null(evall) & !is.null(evalr)){
    if (evall!=cutoff | evalr!=cutoff) stop('kernel only allowed when evall=evalr=cutoff')
  }
  if (kernel!='uniform' & statistic!='ttest' & statistic!='diffmeans') stop('kernel only allowed for diffmeans')
  if (!missing(ci)){if (ci[1]>1 | ci[1]<0) stop('ci must be in [0,1]')}
  if (!is.null(interfci)){
    if (interfci>1 | interfci<0) stop('interfci must be in [0,1]')
    if (statistic!='diffmeans' & statistic!='ttest' & statistic!='ksmirnov' & statistic!='ranksum') stop('interfci only allowed with ttest, ksmirnov or ranksum')
  }
  if (!is.null(bernoulli)){
    randmech <- 'Bernoulli'
    if (max(bernoulli,na.rm=TRUE)>1 | min(bernoulli,na.rm=TRUE)<0) stop('bernoulli probabilities must be in [0,1]')
    if (length(bernoulli)!=length(R)) stop('bernoulli should have the same length as the running variable')
  }
  if (!is.null(wl) & !is.null(wr)){
    wselect <- 'set by user'
    if (wl>=wr) stop('wl has to be smaller than wr')
    if (wl>cutoff | wr<cutoff) stop('window does not include cutoff')
  }
  if (is.null(wl) & !is.null(wr)) stop('wl not specified')
  if (!is.null(wl) & is.null(wr)) stop('wr not specified')
  if (!is.null(evall) & is.null(evalr)) stop('evalr not specified')
  if (is.null(evall) & !is.null(evalr)) stop('evall not specified')
  if (!is.null(d) & !is.null(dscale)) stop('cannot specify both d and dscale')

  Rc <- R - cutoff
  D <- as.numeric(Rc >= 0)

  n <- length(D)
  n1 <- sum(D)
  n0 <- n - n1

  if (seed>0){
    set.seed(seed)
  } else if (seed!=-1){
    stop('Seed has to be a positive integer or -1 for system seed')
  }

  ###############################################################################
  # Window selection
  ###############################################################################

  if (is.null(wl) & is.null(wr)){
    if (missing(covariates)){
      wl <- min(R,na.rm=TRUE)
      wr <- max(R,na.rm=TRUE)
      wselect <- 'run. var. range'
    } else {
      wselect <- 'rdwinselect'
      if (quietly==FALSE) cat('\nRunning rdwinselect...\n')
      rdwlength <- rdwinselect(Rc.long,covariates,obsmin=obsmin,obsstep=obsstep,wmin=wmin,wstep=wstep,wobs=wobs,
                               wasymmetric=wasymmetric,wmasspoints=wmasspoints,dropmissing=dropmissing,nwindows=nwindows,
                               statistic=rdwstat,approx=approx,reps=rdwreps,plot=plot,level=level,seed=seed,quietly=TRUE)
      wl <- cutoff + rdwlength$w_left
      wr <- cutoff + rdwlength$w_right
      if (quietly==FALSE) cat('\nrdwinselect complete.\n')
    }
  }
  if (quietly==FALSE) cat(paste0('\nSelected window = [',round(wl,3),';',round(wr,3),'] \n'))


  if (!is.null(evall)&!is.null(evalr)){if (evall<wl | evalr>wr){stop('evall and evalr need to be inside window')}}

  ww <-  (round(R,8) >= round(wl,8)) & (round(R,8) <= round(wr,8))

  Yw <- Y[ww]
  Rw <- Rc[ww]
  Dw <- D[ww]

  if (!is.null(fuzzy)){
    Tw <- fuzzy.tr[ww]
  }

  if (is.null(bernoulli)){
    data <- cbind(Yw,Rw,Dw)
    data <- data[complete.cases(data),]
    Yw <- data[,1]
    Rw <- data[,2]
    Dw <- data[,3]
  } else {
    Bew <- bernoulli[ww]
    data <- cbind(Yw,Rw,Dw,Bew)
    data <- data[complete.cases(data),]
    Yw <- data[,1]
    Rw <- data[,2]
    Dw <- data[,3]
    Bew <- data[,4]
  }


  n.w <- length(Dw)
  n1.w <- sum(Dw)
  n0.w <- n.w - n1.w


  ###############################################################################
  # Summary statistics
  ###############################################################################

  sumstats <- array(NA,dim=c(5,2))
  sumstats[1,] <- c(n0,n1)
  sumstats[2,] <- c(n0.w,n1.w)
  mean0 <- mean(Yw[Dw==0],na.rm=TRUE)
  mean1 <- mean(Yw[Dw==1],na.rm=TRUE)
  sd0 <- sd(Yw[Dw==0],na.rm=TRUE)
  sd1 <- sd(Yw[Dw==1],na.rm=TRUE)
  sumstats[3,] <- c(mean0,mean1)
  sumstats[4,] <- c(sd0,sd1)
  sumstats[5,] <- c(wl,wr)

  if (is.null(d) & is.null(dscale)){
    delta <- .5*sd0
  }
  if (!is.null(d) & is.null(dscale)){
    delta <- d
  }
  if (is.null(d) & !is.null(dscale)){
    delta <- dscale*sd0
  }

  ###############################################################################
  # Weights
  ###############################################################################

  kweights <- rep(1,n.w)

  if (kernel=='triangular'){
    bwt <- wr - cutoff
    bwc <- wl - cutoff
    kweights[Dw==1] <- (1-abs(Rw[Dw==1]/bwt))*(abs(Rw[Dw==1]/bwt)<1)
    kweights[Dw==0] <- (1-abs(Rw[Dw==0]/bwc))*(abs(Rw[Dw==0]/bwc)<1)
  }
  if (kernel=='epan'){
    bwt <- wr - cutoff
    bwc <- wl - cutoff
    kweights[Dw==1] <- .75*(1-(Rw[Dw==1]/bwt)^2)*(abs(Rw[Dw==1]/bwt)<1)
    kweights[Dw==0] <- .75*(1-(Rw[Dw==0]/bwc)^2)*(abs(Rw[Dw==0]/bwc)<1)
  }


  ###############################################################################
  # Outcome adjustment: model and null hypothesis
  ###############################################################################

  Y.adj <- Yw

  if (p>0){
    if (is.null(evall) & is.null(evalr)){
      evall <- cutoff
      evalr <- cutoff
    }
    R.adj <- Rw + cutoff - Dw*evalr - (1-Dw)*evall
    Rpoly <- poly(R.adj,order=p,raw=TRUE)
    lfit.t <- lm(Yw[Dw==1] ~ Rpoly[Dw==1,],weights=kweights[Dw==1])
    Y.adj[Dw==1] <- lfit.t$residuals + lfit.t$coefficients[1]
    lfit.c <- lm(Yw[Dw==0] ~ Rpoly[Dw==0,],weights=kweights[Dw==0])
    Y.adj[Dw==0] <- lfit.c$residuals + lfit.c$coefficients[1]
  }

  if (is.null(fuzzy)){
    Y.adj.null <- Y.adj - nulltau*Dw
  } else{
    Y.adj.null <- Y.adj - nulltau*Tw
  }


  ###############################################################################
  # Observed statistics and asymptotic p-values
  ###############################################################################


  if (is.null(fuzzy)){
    results <- rdrandinf.model(Y.adj.null,Dw,statistic=statistic,pvalue=TRUE,kweights=kweights,delta=delta)
  } else {
    results <- rdrandinf.model(Y.adj.null,Dw,statistic=fuzzy.stat,endogtr=Tw,pvalue=TRUE,kweights=kweights,delta=delta)
  }

  obs.stat <- as.numeric(results$statistic)

  if (p==0){
    if (fuzzy.stat=='wald'){
      firststagereg <- lm(Tw ~ Dw)
      aux <- AER::ivreg(Yw ~ Tw | Dw,weights=kweights)
      obs.stat <- aux$coefficients["Tw"]
      se <- sqrt(diag(sandwich::vcovHC(aux,type='HC1'))['Tw'])
      ci.lb <- obs.stat - 1.96*se
      ci.ub <- obs.stat + 1.96*se
      tstat <- aux$coefficients['Tw']/se
      asy.pval <- as.numeric(2*pnorm(-abs(tstat)))
      asy.power <- as.numeric(1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se))
    } else {
      asy.pval <- as.numeric(results$p.value)
      asy.power <- as.numeric(results$asy.power)
    }

  } else {
    if (statistic=='diffmeans'|statistic=='ttest'|statistic=='all'){
      lfit <- lm(Yw ~ Dw + Rpoly + Dw*Rpoly,weights=kweights)
      se <- sqrt(diag(sandwich::vcovHC(lfit,type='HC2'))['Dw'])
      tstat <- lfit$coefficients['Dw']/se
      asy.pval <- as.numeric(2*pnorm(-abs(tstat)))
      asy.power <- as.numeric(1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se))
    }
    if (statistic=='ksmirnov'|statistic=='ranksum'){
      asy.pval <- NA
      asy.power <- NA
    }
    if (statistic=='all'){
      asy.pval <- c(as.numeric(asy.pval),NA,NA)
      asy.power <- c(as.numeric(asy.power),NA,NA)
    }

    if (fuzzy.stat=='wald'){
      inter <- Rpoly*Dw
      firststagereg <- lm(Tw ~ Dw)
      aux <- AER::ivreg(Yw ~ Rpoly + inter + Tw | Rpoly + inter + Dw,weights=kweights)
      obs.stat <- aux$coefficients["Tw"]
      se <- sqrt(diag(sandwich::vcovHC(aux,type='HC1'))['Tw'])
      ci.lb <- obs.stat - 1.96*se
      ci.ub <- obs.stat + 1.96*se
      tstat <- aux$coefficients['Tw']/se
      asy.pval <- as.numeric(2*pnorm(-abs(tstat)))
      asy.power <- as.numeric(1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se))
    }
  }


  ###############################################################################
  # Randomization-based inference
  ###############################################################################


  if (statistic == 'all'){
    stats.distr <- array(NA,dim=c(reps,3))
  } else{
    stats.distr <- array(NA,dim=c(reps,1))
  }

  if (quietly==FALSE){cat('\nRunning randomization-based test...\n')}

  if (fuzzy.stat!='wald'){
    if (is.null(bernoulli)){

      max.reps <- choose(n.w,n1.w)
      reps <- min(reps,max.reps)
      if (max.reps<reps){
        warning(paste0('Chosen no. of reps > total no. of permutations.\n reps set to ',reps,'.'))
      }

      for (i in 1:reps) {
        D.sample <- sample(Dw,replace=FALSE)
        if (is.null(fuzzy)){
          obs.stat.sample <- as.numeric(rdrandinf.model(Y.adj.null,D.sample,statistic,kweights=kweights,delta=delta)$statistic)
        } else {
          obs.stat.sample <- as.numeric(rdrandinf.model(Y.adj.null,D.sample,statistic=fuzzy.stat,endogtr=Tw,kweights=kweights,delta=delta)$statistic)
        }
        stats.distr[i,] <- obs.stat.sample
      }

    } else {

      for (i in 1:reps) {
        D.sample <- as.numeric(runif(n.w)<=Bew)
        if (mean(D.sample)==1 | mean(D.sample)==0){
          stats.distr[i,] <- NA # ignore cases where bernoulli assignment mechanism gives no treated or no controls
        } else {
          obs.stat.sample <- as.numeric(rdrandinf.model(Y.adj.null,D.sample,statistic,kweights=kweights,delta=delta)$statistic)
          stats.distr[i,] <- obs.stat.sample
        }
      }

    }

    if(quietly==FALSE) cat('Randomization-based test complete. \n')

    if (statistic == 'all'){
      p.value1 <- mean(abs(stats.distr[,1]) >= abs(obs.stat[1]),na.rm=TRUE)
      p.value2 <- mean(abs(stats.distr[,2]) >= abs(obs.stat[2]),na.rm=TRUE)
      p.value3 <- mean(abs(stats.distr[,3]) >= abs(obs.stat[3]),na.rm=TRUE)
      p.value <- c(p.value1,p.value2,p.value3)
    } else{
      p.value <- mean(abs(stats.distr) >= abs(obs.stat),na.rm=TRUE)
    }
  } else {
    p.value <- NA
  }

  ###############################################################################
  # Confidence interval
  ###############################################################################

  if (!missing(ci)){
    ci.alpha <- ci[1]
    if (fuzzy.stat!='wald'){

      wr_c <- wr - cutoff
      wl_c <- wl - cutoff

      if (length(ci)>1){
        tlist <- ci[-1]
        aux <- rdsensitivity(Y,Rc,p=p,wlist=wr_c,wlist_left=wl_c,tlist=tlist,fuzzy=fuzzy,ci=c(wl_c,wr_c),ci_alpha=ci.alpha,
                             reps=reps,quietly=quietly,seed=seed,nodraw=TRUE)

      } else {
        aux <- rdsensitivity(Y,Rc,p=p,wlist=wr_c,wlist_left=wl_c,fuzzy=fuzzy,ci=c(wl_c,wr_c),ci_alpha=ci.alpha,
                             reps=reps,quietly=quietly,seed=seed,nodraw=TRUE)

      }
      conf.int <- aux$ci
    }
    else {
      conf.int <- c(ci.lb,ci.ub)
    }
    if (is.na(conf.int[1]) | is.na(conf.int[2])){
      warning('Consider a larger tlist in ci() option.')
    }
  }


  ###############################################################################
  # Confidence interval under interference
  ###############################################################################

  if (!is.null(interfci)){
    p.low <- interfci/2
    p.high <- 1-interfci/2
    qq <- quantile(stats.distr,probs=c(p.low,p.high))
    interf.ci <- c(obs.stat-as.numeric(qq[2]),obs.stat-as.numeric(qq[1]))
  }


  ###############################################################################
  # Output and display results
  ###############################################################################


  if (missing(ci) & is.null(interfci)){
    output <- list(sumstats = sumstats,
                   obs.stat = obs.stat,
                   p.value = p.value,
                   asy.pvalue = asy.pval,
                   window = c(wl,wr))
  }
  if (!missing(ci) & is.null(interfci)){
    output <- list(sumstats = sumstats,
                   obs.stat = obs.stat,
                   p.value = p.value,
                   asy.pvalue = asy.pval,
                   window = c(wl,wr),
                   ci = conf.int)
  }
  if (missing(ci) & !is.null(interfci)){
    output <- list(sumstats = sumstats,
                   obs.stat = obs.stat,
                   p.value = p.value,
                   asy.pvalue = asy.pval,
                   window = c(wl,wr),
                   interf.ci = interf.ci)
  }
  if (!missing(ci) & !is.null(interfci)){
    output <- list(sumstats = sumstats,
                   obs.stat = obs.stat,
                   p.value = p.value,
                   asy.pvalue = asy.pval,
                   window = c(wl,wr),
                   ci = conf.int,
                   interf.ci = interf.ci)
  }


  if (quietly==FALSE){
    if (statistic=='diffmeans'|statistic=='ttest'){statdisp = 'Diff. in means'}
    if (statistic=='ksmirnov'){statdisp = 'Kolmogorov-Smirnov'}
    if (statistic=='ranksum'){statdisp = 'Rank sum z-stat'}
    if (fuzzy.stat=='ar'){
      statdisp <- 'ITT'
    }
    if (fuzzy.stat=='wald'){statdisp = 'TSLS'}

    cat('\n\n')
    cat(format('Number of obs     =',    width = 18))
    cat(format(sprintf('%6.0f',n),       width = 14, justify='right')); cat("\n")
    cat(format('Order of poly     =',    width = 18))
    cat(format(sprintf('%6.0f',p),       width = 14, justify='right')); cat("\n")
    cat(format('Kernel type       =',    width = 18))
    cat(format(kernel,                   width = 14, justify='right')); cat("\n")
    cat(format('Reps              =',    width = 18))
    cat(format(sprintf('%6.0f',reps),    width = 14, justify='right')); cat("\n")
    cat(format('Window            =',    width = 18))
    cat(format(wselect,                  width = 14, justify='right')); cat("\n")
    cat(format('H0:          tau  =',    width = 18))
    cat(format(sprintf('%4.3f',nulltau), width = 14, justify='right')); cat("\n")
    cat(format('Randomization     =',    width = 18))
    cat(format(randmech,                 width = 14, justify='right'))
    cat('\n\n')

    cat(format("Cutoff c = ",           width = 10))
    cat(format(sprintf('%4.3f',cutoff), width = 8, justify='right'))
    cat(format("Left of c",             width = 12,justify='right'))
    cat(format("Right of c",            width = 12,justify='right')); cat("\n")

    cat(format("Number of obs",         width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0),     width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1),     width = 12,justify='right')); cat("\n")

    cat(format("Eff. number of obs",    width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0.w),   width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1.w),   width = 12,justify='right')); cat("\n")

    cat(format("Mean of outcome",       width = 19,justify='right'))
    cat(format(sprintf('%4.3f',mean0),  width = 12,justify='right'))
    cat(format(sprintf('%4.3f',mean1),  width = 12,justify='right')); cat("\n")

    cat(format("S.d. of outcome",       width = 19,justify='right'))
    cat(format(sprintf('%4.3f',sd0),    width = 12,justify='right'))
    cat(format(sprintf('%4.3f',sd1),    width = 12,justify='right')); cat("\n")

    cat(format("Window",                width = 19,justify='right'))
    cat(format(sprintf('%4.3f',wl),     width = 12,justify='right'))
    cat(format(sprintf('%4.3f',wr),     width = 12,justify='right')); cat("\n")
    cat('\n')

    cat(paste0(rep('=',80),collapse='')); cat('\n')

    if (firststage==TRUE & fuzzy.stat=='wald'){
      cat("First stage regression"); cat('\n')
      print(summary(firststagereg))
      cat(paste0(rep('=',80),collapse='')); cat('\n')
    }

    cat(format('',              width = 31))
    cat(format('Finite sample', width = 20,justify='centre'))
    cat(format('Large sample',  width = 29,justify='centre'));cat('\n')

    cat(format('', width = 31))
    cat(paste0(rep('-',18),collapse=''))
    cat(format('', width = 2))
    cat(paste0(rep('-',29),collapse='')); cat('\n')

    cat(format('Statistic',            width = 19,justify='right'))
    cat(format('T',                    width = 11,justify='right'))
    cat(format('P>|T|',                width = 21,justify='centre'))
    cat(format('P>|T|',                width = 9,justify='left'))
    cat(format('Power vs d = ',        width = 13,justify='right'))
    cat(format(sprintf('%4.3f',delta), width = 7,justify='right')); cat('\n')

    cat(paste0(rep('=',80),collapse='')); cat('\n')


    if (statistic!='all'){

      cat(format(statdisp,                   width = 19, justify='right'))
      cat(format(sprintf('%4.3f',obs.stat),  width = 11, justify='right'))
      cat(format(sprintf('%1.3f',p.value),   width = 21, justify='centre'))
      cat(format(sprintf('%1.3f',asy.pval),  width = 9,  justify='left'))
      cat(format(sprintf('%1.3f',asy.power), width = 20, justify='right')); cat('\n')

    }

    if (statistic=='all'){

      cat(format('Diff. in means',              width = 19, justify='right'))
      cat(format(sprintf('%4.3f',obs.stat[1]),  width = 11, justify='right'))
      cat(format(sprintf('%1.3f',p.value[1]),   width = 21, justify='centre'))
      cat(format(sprintf('%1.3f',asy.pval[1]),  width = 9,  justify='left'))
      cat(format(sprintf('%1.3f',asy.power[1]), width = 20, justify='right')); cat('\n')

      cat(format('Kolmogorov-Smirnov',          width = 19, justify='right'))
      cat(format(sprintf('%4.3f',obs.stat[2]),  width = 11, justify='right'))
      cat(format(sprintf('%1.3f',p.value[2]),   width = 21, justify='centre'))
      cat(format(sprintf('%1.3f',asy.pval[2]),  width = 9,  justify='left'))
      cat(format(sprintf('%1.3f',asy.power[2]), width = 20, justify='right')); cat('\n')

      cat(format('Rank sum z-stat',             width = 19, justify='right'))
      cat(format(sprintf('%4.3f',obs.stat[3]),  width = 11, justify='right'))
      cat(format(sprintf('%1.3f',p.value[3]),   width = 21, justify='centre'))
      cat(format(sprintf('%1.3f',asy.pval[3]), width = 9,  justify='left'))
      cat(format(sprintf('%1.3f',asy.power[3]), width = 20, justify='right')); cat('\n')

    }

    cat(paste0(rep('=',80),collapse='')); cat('\n')


    if (!missing(ci)){
      cat('\n')
      if (fuzzy.stat!='wald'){
        if(nrow(conf.int)==1){
          cat(paste0((1-ci.alpha)*100,'% confidence interval: [',round(conf.int[1],3),',',round(conf.int[2],3),']\n'))
        } else{
          cat(paste0((1-ci.alpha)*100,'% confidence interval:'))
          cat('\n')
          print(round(conf.int,3))
          cat('\n')
          cat('Note: CI is disconnected - each row is a subset of the CI')
        }
      } else {
        cat(paste0((1-ci.alpha)*100,'% confidence interval: [',round(ci.lb,3),',',round(ci.ub,3),']\n'))
        cat('CI based on asymptotic approximation')
      }


    }

    if (!is.null(interfci)){
      cat('\n')
      cat(paste0((1-interfci)*100,'% confidence interval under interference: [',round(interf.ci[1],3),';',round(interf.ci[2],3),']')); cat('\n')
    }
  }
  return(output)
}
