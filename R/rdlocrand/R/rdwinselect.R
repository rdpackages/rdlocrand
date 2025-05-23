###############################################################################
# rdwinselect: window selection for randomization inference in RD
# !version 1.1 22-May-2025
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

#' Window selection for RD designs under local randomization
#'
#' \code{rdwinselect} implements the window-selection procedure
#'  based on balance tests for RD designs under local randomization.
#'  Specifically, it constructs a sequence of nested windows around the RD cutoff
#'  and reports binomial tests for the running variable runvar and covariate balance
#'  tests for covariates covariates (if specified). The recommended window is the
#'  largest window around the cutoff such that the minimum p-value of the balance test
#'  is larger than a prespecified level for all nested (smaller) windows. By default,
#'  the p-values are calculated using randomization inference methods.
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
#' @param R a vector containing the values of the running variable.
#' @param X the matrix of covariates to be used in the balancing tests. The matrix is optional but the recommended window is only provided when at least one covariate is specified. This should be a matrix of size n x k where n is the total sample size and $k$ is the number of covariates.
#' @param cutoff the RD cutoff (default is 0).
#' @param obsmin the minimum number of observations above and below the cutoff in the smallest window. Default is 10.
#' @param wmin the smallest window to be used.
#' @param wobs the number of observations to be added at each side of the cutoff at each step. Default is 5.
#' @param wasymmetric allows for asymmetric windows around the cutoff when (\code{wobs} is specified).
#' @param wmasspoints specifies that the running variable is discrete and each masspoint should be used as a window.
#' @param wstep the increment in window length.
#' @param nwindows the number of windows to be used. Default is 10.
#' @param dropmissing drop rows with missing values in covariates when calculating windows.
#' @param statistic the statistic to be used in the balance tests. Allowed options are \code{diffmeans} (difference in means statistic), \code{ksmirnov} (Kolmogorov-Smirnov statistic), \code{ranksum} (Wilcoxon-Mann-Whitney standardized statistic) and \code{hotelling} (Hotelling's T-squared statistic). Default option is \code{diffmeans}. The statistic \code{ttest} is equivalent to \code{diffmeans} and included for backward compatibility.
#' @param p the order of the polynomial for outcome adjustment model (for covariates). Default is 0.
#' @param evalat specifies the point at which the adjusted variable is evaluated. Allowed options are \code{cutoff} and \code{means}. Default is \code{cutoff}.
#' @param kernel specifies the type of kernel to use as weighting scheme. Allowed kernel types are \code{uniform} (uniform kernel), \code{triangular} (triangular kernel) and \code{epan} (Epanechnikov kernel). Default is \code{uniform}.
#' @param approx forces the command to conduct the covariate balance tests using a large-sample approximation instead of finite-sample exact randomization inference methods.
#' @param level the minimum accepted value of the p-value from the covariate balance tests. Default is .15.
#' @param reps number of replications. Default is 1000.
#' @param seed the seed to be used for the randomization tests.
#' @param plot draws a scatter plot of the minimum p-value from the covariate balance test against window length.
#' @param quietly suppress output
#' @param obsstep the minimum number of observations to be added on each side of the cutoff for the sequence of fixed-increment nested windows. This option is deprecated and only included for backward compatibility.
#'
#' @return
#' \item{window}{recommended window (NA is covariates are not specified)}
#' \item{wlist}{list of window lengths}
#' \item{results}{table including window lengths, minimum p-value in each window, corresponding number of the variable with minimum p-value (i.e. column of covariate matrix), Binomial test p-value and sample sizes to the left and right of the cutoff in each window.}
#' \item{summary}{summary statistics.}
#'
#' @examples
#' # Toy dataset
#' X <- array(rnorm(200),dim=c(100,2))
#' R <- X[1,] + X[2,] + rnorm(100)
#' # Window selection adding 5 observations at each step
#' # Note: low number of replications to speed up process.
#' tmp <- rdwinselect(R,X,obsmin=10,wobs=5,reps=500)
#' # Window selection setting initial window and step
#' # The user should increase the number of replications.
#' tmp <- rdwinselect(R,X,wmin=.5,wstep=.125,reps=500)
#' # Window selection with approximate (large sample) inference and p-value plot
#' tmp <- rdwinselect(R,X,wmin=.5,wstep=.125,approx=TRUE,nwin=80,quietly=TRUE,plot=TRUE)
#'
#'
#' @export


rdwinselect <- function(R, X,
                       cutoff = 0,
                       obsmin = NULL,
                       wmin = NULL,
                       wobs = NULL,
                       wstep = NULL,
                       wasymmetric = FALSE,
                       wmasspoints = FALSE,
                       dropmissing = FALSE,
                       nwindows = 10,
                       statistic = 'diffmeans',
                       p = 0,
                       evalat = 'cutoff',
                       kernel = 'uniform',
                       approx = FALSE,
                       level = .15,
                       reps = 1000,
                       seed = 666,
                       plot = FALSE,
                       quietly = FALSE,
                       obsstep = NULL) {


  ###############################################################################
  # Parameters and error checking
  ###############################################################################

  if (cutoff<=min(R,na.rm=TRUE) | cutoff>=max(R,na.rm=TRUE)) stop('Cutoff must be within the range of the running variable')
  if (p<0) stop('p must be a positive integer')
  if (p>0 & approx==TRUE & statistic!='ttest' & statistic!='diffmeans') stop('approximate and p>1 can only be combined with diffmeans')
  if (statistic!='diffmeans' & statistic!='ttest' & statistic!='ksmirnov' & statistic!='ranksum' & statistic!='hotelling') stop(paste(statistic,'not a valid statistic'))
  if (evalat!='cutoff' & evalat!='means') stop('evalat only admits means or cutoff')
  if (kernel!='uniform' & kernel!='triangular' & kernel!='epan') stop(paste(kernel,'not a valid kernel'))
  if (kernel!='uniform' & evalat!='cutoff') stop('kernel can only be combined with evalat(cutoff)')
  if (kernel!='uniform' & statistic!='ttest' & statistic!='diffmeans') stop('kernel only allowed for diffmeans')
  if (!is.null(obsmin) & !is.null(wmin)) stop('cannot set both obsmin and wmin')
  if (!is.null(wobs) & !is.null(wstep)) stop('cannot set both wobs and wstep')
  if (wmasspoints==TRUE) {
    if (!is.null(obsmin)) stop('obsmin not allowed with wmasspoints')
    if (!is.null(wmin)) stop('wmin not allowed with wmasspoints')
    if (!is.null(wobs)) stop('wobs not allowed with wmasspoints')
    if (!is.null(wstep)) stop('wstep not allowed with wmasspoints')
  }

  Rc <- R - cutoff
  D <- Rc >= 0

  if (!missing(X)){
    if (dropmissing==FALSE){
      data <- data.frame(Rc,D,X)
      data <- data[complete.cases(data[,1:2]),]
      data <- data[order(data$Rc),]
      Rc <- data[,1]
      D <- data[,2]
      X <- data[,c(-1,-2)]
    } else {
      data <- data.frame(Rc,D,X)
      data <- data[complete.cases(data),]
      data <- data[order(data$Rc),]
      Rc <- data[,1]
      D <- data[,2]
      X <- data[,c(-1,-2)]
    }


  } else{
    data <- data.frame(Rc,D)
    data <- data[complete.cases(data),]
    data <- data[order(data$Rc),]
    Rc <- data[,1]
    D <- data[,2]
  }

  if (!missing(X)){X <- as.matrix(X)}
  if (seed>0){
    set.seed(seed)
  } else if (seed!=-1){
    stop('Seed has to be a positive integer or -1 for system seed')
  }

  if (approx==FALSE){testing_method='rdrandinf'}else{testing_method='approximate'}

  n <- length(Rc)
  n1 <- sum(D)
  n0 <- n - n1
  dups <- merge(Rc,table(Rc),by=1)
  dups <- dups[,2]

  if (max(dups)>1){
    cat('Mass points detected in running variable')
    cat('\n')
    cat('You may use wmasspoints option for constructing windows at each mass point')
    cat('\n')
    mp_left <- unique(Rc[D==0])
    mp_right <- unique(Rc[D==1])
    if (wmasspoints==TRUE){
      nmax <- min(max(length(mp_left),length(mp_right)),nwindows)
      wlist <- matrix(NA,nrow=2,ncol=nmax)
    }
  }


  ###############################################################################
  # Define initial window
  ###############################################################################

  ## Define initial window

  if (is.null(wmin)){
    posl <-
    posr <- n0 + 1

    if (is.null(obsmin)){
      obsmin <- 10
    }
    if (wmasspoints==TRUE){
      obsmin <- 1
      wasymmetric <- TRUE
    }
    if (!is.null(obsstep)){
      wmin <- findwobs_sym(obsmin,1,posl,posr,Rc,dups)
    }
    if (wasymmetric==TRUE){
      tmp <- findwobs(obsmin,1,posl,posr,Rc,dups)
      wmin_left <- tmp$wlength_left
      posmin_left <- tmp$poslist_left
      wmin_right <- tmp$wlength_right
      posmin_right <- tmp$poslist_right
    } else{
      wmin_right <- findwobs_sym(obsmin,1,posl,posr,Rc,dups)
      wmin_left <- -wmin_right
    }

  } else{
    wcount <- length(wmin)
    if (wcount==1){
      wmin_right <- wmin
      wmin_left <- -wmin
      posmin_right <- 45 + sum(Rc<=wmin & Rc>=0)
      posmin_left <- n0 - sum(Rc<0 & Rc>=-wmin) + 1
    } else if(wcount==2){
      wmin_left <- wmin[1]
      wmin_right <- wmin[2]
      posmin_right <- n0 + sum(Rc<=wmin_right & Rc>=0)
      posmin_left <- n0 - sum(Rc<0 & Rc>=wmin_left) + 1
    } else{
      stop('wmin option incorrectly specified')
    }
  }


  ###############################################################################
  # Define window list
  ###############################################################################

  if (!is.null(obsstep)){
    warning('obsstep included for bacwkard compatibility only. \n The use of wstep and wobs is recommended.')
    wstep <- findstep(Rc,D,obsmin,obsstep,10)
    wlist_right <- seq(from=wmin,by=wstep,length.out=nwindows)
    wlist_left <- NULL
  } else if (!is.null(wstep)){
    wmax_left <- max(wmin_left - wstep*(nwindows-1),min(Rc))
    wmax_right <- min(wmin_right + wstep*(nwindows-1),max(Rc))
    wlist_left <- sort(seq(from=wmax_left,to=wmin_left,by=wstep),decreasing=TRUE)
    wlist_right <- seq(from=wmin_right,to=wmax_right,by=wstep)
  } else {
    if (is.null(wobs)){
      wobs <- 5
    }
    if (wmasspoints==TRUE){
      wobs <- 1
    }
    posl <- max(n0 - sum(Rc<0 & Rc>=wmin_left),1)
    posr <- min(n0 + 1 + sum(Rc>=0 & Rc<=wmin_right),n)
    if (wasymmetric==TRUE){
      tmp <- findwobs(wobs,nwindows-1,posl,posr,Rc,dups)
      wlist_left <- c(wmin_left,tmp$wlist_left)
      poslist_left <- c(posmin_left,tmp$poslist_left)
      wlist_right <- c(wmin_right,tmp$wlist_right)
      poslist_right <- c(posmin_right,tmp$poslist_right)
    } else{
      wlist <- findwobs_sym(wobs,nwindows-1,posl,posr,Rc,dups)
      wlist_right <- c(wmin_right,wlist)
      wlist_left <- c(wmin_left,wlist)
    }
  }

  nmax <- min(nwindows,length(wlist_right))
  if (nmax<nwindows){
    cat('\n')
    warning('Not enough observations to calculate all windows. \n Consider changing wmin(), wobs() or wstep().')
  }


  ###############################################################################
  # Summary statistics
  ###############################################################################

  table_sumstats <- array(NA,dim=c(5,2))
  table_sumstats[1,] <- c(n0,n1)

  qq0 <- round(quantile(abs(Rc[D==0]),probs = c(.01,.05,.1,.2),type=1,na.rm=TRUE),5)
  qq1 <- round(quantile(Rc[D==1],probs = c(.01,.05,.1,.2),type=1,na.rm=TRUE),5)

  n0.q1 <- sum(Rc>=-qq0[1]& Rc<0,na.rm=TRUE)
  n0.q2 <- sum(Rc>=-qq0[2]& Rc<0,na.rm=TRUE)
  n0.q3 <- sum(Rc>=-qq0[3]& Rc<0,na.rm=TRUE)
  n0.q4 <- sum(Rc>=-qq0[4]& Rc<0,na.rm=TRUE)
  n1.q1 <- sum(Rc<=qq1[1]& Rc>=0,na.rm=TRUE)
  n1.q2 <- sum(Rc<=qq1[2]& Rc>=0,na.rm=TRUE)
  n1.q3 <- sum(Rc<=qq1[3]& Rc>=0,na.rm=TRUE)
  n1.q4 <- sum(Rc<=qq1[4]& Rc>=0,na.rm=TRUE)

  table_sumstats[2,] <- c(n0.q1,n1.q1)
  table_sumstats[3,] <- c(n0.q2,n1.q2)
  table_sumstats[4,] <- c(n0.q3,n1.q3)
  table_sumstats[5,] <- c(n0.q4,n1.q4)

  ###############################################################################
  ## Display upper-right panel
  ###############################################################################

  if (quietly==FALSE){
    cat('\n')
    cat('\nWindow selection for RD under local randomization \n')
    cat('\n')
    cat(format('Number of obs     =',    width = 18))
    cat(format(sprintf('%6.0f',n),    width = 14,justify='right')); cat("\n")
    cat(format('Order of poly     =',    width = 18))
    cat(format(sprintf('%2.0f',p),    width = 14,justify='right')); cat("\n")
    cat(format('Kernel type       =',    width = 18))
    cat(format(kernel,                width = 14,justify='right')); cat("\n")
    cat(format('Reps              =',    width = 18))
    cat(format(sprintf('%6.0f',reps), width = 14,justify='right')); cat("\n")
    cat(format('Testing method    =',    width = 18))
    cat(format(testing_method,        width = 14,justify='right')); cat("\n")
    cat(format('Balance test      =',    width = 18))
    cat(format(statistic,             width = 14,justify='right'))
    cat('\n\n')
  }


  ###############################################################################
  ## Display upper left panel
  ###############################################################################

  if (quietly==FALSE){
    cat(format("Cutoff c = ",           width = 10))
    cat(format(sprintf('%4.3f',cutoff), width = 8,justify='right'))
    cat(format("Left of c",             width = 12,justify='right'))
    cat(format("Right of c",            width = 12,justify='right')); cat("\n")

    cat(format("Number of obs",        width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0),    width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1),    width = 12,justify='right')); cat("\n")

    cat(format("1st percentile",       width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0.q1), width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1.q1), width = 12,justify='right')); cat("\n")

    cat(format("5th percentile",       width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0.q2), width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1.q2), width = 12,justify='right')); cat("\n")

    cat(format("10th percentile",      width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0.q3), width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1.q3), width = 12,justify='right')); cat("\n")

    cat(format("20th percentile",      width = 19,justify='right'))
    cat(format(sprintf('%6.0f',n0.q4), width = 12,justify='right'))
    cat(format(sprintf('%6.0f',n1.q4), width = 12,justify='right'))
    cat("\n\n")
  }


  ###############################################################################
  # Balance tests
  ###############################################################################

  table_rdw <- array(NA,dim=c(nmax,7))

  ## Being main panel display

  if (quietly==FALSE){
    cat(paste0(rep('=',80),collapse='')); cat('\n')
    cat(format('Window',            width = 19,justify='centre'))
    cat(format('p-value ',          width = 11,justify='right'))
    cat(format('Var. name',         width = 16,justify='right'))
    cat(format('Bin.test',          width = 12,justify='right'))
    cat(format('Obs<c',             width = 11,justify='right'))
    cat(format('Obs>=c',            width = 11,justify='right')); cat('\n')
    cat(paste0(rep('=',80),collapse='')); cat('\n')
  }

  for (j in 1:nmax){

    if (wasymmetric==TRUE & is.null(wstep) & is.null(obsstep)){
      wlower <- wlist_left[j]
      wupper <- wlist_right[j]

      position_l <- poslist_left[j]
      position_r <- poslist_right[j]

      ww <- (Rc>=Rc[position_l] & Rc<=Rc[position_r])

    } else{
      wupper <- wlist_right[j]
      wlower <- -wupper
      wlist_left[j] <- wlower

      ww <- (Rc>=wlower & Rc<=wupper)

    }

    Dw <- D[ww]
    Rw <- Rc[ww]

    ## Drop NA values

    if (!missing(X)){
      Xw <- X[ww,]
      data <- data.frame(Rw,Dw,Xw)
      data <- data[complete.cases(data),]
      Rw <- data[,1]
      Dw <- data[,2]
      Xw <- data[,c(-1,-2)]
    } else {
      data <- cbind(Rw,Dw)
      data <- data[complete.cases(data),]
      Rw <- data[,1]
      Dw <- data[,2]
    }

    ## Sample sizes

    n0.w <- sum(Dw==0)
    n1.w <- sum(Dw==1)
    n.w <- n0.w+n1.w
    table_rdw[j,4] <- n0.w
    table_rdw[j,5] <- n1.w

    if (n0.w==0 | n1.w==0){
      table_rdw[j,1] <- NA
      table_rdw[j,2] <- NA
      varname <- ''
    } else{

      ## Binomial test

      bitest <- binom.test(sum(Dw),length(Dw),p=0.5)
      table_rdw[j,3] <- bitest$p.value

      if (!missing(X)){

        ## Weights

        kweights <- rep(1,n.w)

        if (kernel=='triangular'){
          kweights <- (1-abs(Rw/wupper))*(abs(Rw/wupper)<=1)
          kweights[kweights==0] <- .Machine$double.eps
        }
        if (kernel=='epan'){
          kweights <- .75*(1-(Rw/wupper)^2)*(abs(Rw/wupper)<=1)
          kweights[kweights==0] <- .Machine$double.eps
        }

        ## Model adjustment

        if (p>0){

          X.adj <- matrix(NA,nrow=nrow(Xw),ncol=ncol(Xw))

          if (evalat=='cutoff'){
            evall <- cutoff
            evalr <- cutoff
          } else if (evalat=='means'){
            evall <- mean(Rw[Dw==0]) + cutoff
            evalr <- mean(Rw[Dw==1]) + cutoff
          }
          R.adj <- Rw + cutoff - Dw*evalr - (1-Dw)*evall
          Rpoly <- poly(R.adj,order=p,raw=TRUE)

          for (k in 1:ncol(Xw)){
            lfit.t <- lm(Xw[Dw==1,k] ~ Rpoly[Dw==1,],weights=kweights[Dw==1])
            X.adj[Dw==1,k] <- lfit.t$residuals + lfit.t$coefficients[1]
            lfit.c <- lm(Xw[Dw==0,k] ~ Rpoly[Dw==0,],weights=kweights[Dw==0])
            X.adj[Dw==0,k] <- lfit.c$residuals + lfit.c$coefficients[1]
          }
          Xw <- X.adj
        }

        ## Statistics and p-values

        if (statistic=='hotelling'){
          aux <- hotelT2(Xw,Dw)
          obs.stat <- as.numeric(aux$statistic)
          if (approx==FALSE){
            stat.distr <- array(NA,dim=c(reps,1))
            for (i in 1:reps){
              D.sample <- sample(Dw,replace=FALSE)
              aux.sample <- hotelT2(Xw,D.sample)
              obs.stat.sample <- as.numeric(aux.sample$statistic)
              stat.distr[i] <- obs.stat.sample
            }
            p.value <- mean(abs(stat.distr) >= abs(obs.stat))
          } else {
            p.value <- as.numeric(aux$p.value)
          }
          table_rdw[j,1] <- p.value
          varname <- NA
        } else {
          aux <- rdrandinf.model(Xw,Dw,statistic=statistic,kweights=kweights,pvalue=TRUE)
          obs.stat <- as.numeric(aux$statistic)
          if (approx==FALSE){
            stat.distr <- array(NA,dim=c(reps,ncol(X)))
            for (i in 1:reps){
              D.sample <- sample(Dw,replace=FALSE)
              aux.sample <- rdrandinf.model(Xw,D.sample,statistic=statistic,kweights=kweights)
              obs.stat.sample <- as.numeric(aux.sample$statistic)
              stat.distr[i,] <- obs.stat.sample
            }
            p.value <- rowMeans(t(abs(stat.distr)) >= abs(obs.stat))
          } else {
            if (p==0){
              p.value <- as.numeric(aux$p.value)
            } else {
              p.value <- numeric(ncol(X))
              for (k in 1:ncol(X)){
                lfit <- lm(Xw[,k] ~ Dw + Rpoly + Dw*Rpoly,weights=kweights)
                tstat <- lfit$coefficients['Dw']/sqrt(sandwich::vcovHC(lfit,type='HC2')['Dw','Dw'])
                p.value[k] <- 2*pnorm(-abs(tstat))
              }
            }
          }

          table_rdw[j,1] <- min(p.value)
          tmp <- which.min(p.value)
          table_rdw[j,2] <- tmp

          if (!is.null(colnames(X)[tmp])){
            if (colnames(X)[tmp]!='') {
              varname <- substring(colnames(X)[tmp],1,15)
            }
            else{
              varname <- tmp
            }
          } else {
            varname <- tmp
          }
        }

      } else {
        table_rdw[j,1] <- NA
        table_rdw[j,2] <- NA
        varname <- NA
      }
    }
    table_rdw[j,6] <- wlower
    table_rdw[j,7] <- wupper

    if (quietly==FALSE){
      cat(format(sprintf('%6.4f',wlower+cutoff),    width = 9,justify='right'))
      cat(format(sprintf('%6.4f',wupper+cutoff),    width = 9,justify='right'))
      cat(format(sprintf('%1.3f',table_rdw[j,1]), width = 11,justify='right'))
      cat(format(varname,                           width = 16,justify='right'))
      cat(format(sprintf('%1.3f',table_rdw[j,3]), width = 12,justify='right'))
      cat(format(sprintf('%6.0f',table_rdw[j,4]),   width = 11,justify='right'))
      cat(format(sprintf('%6.0f',table_rdw[j,5]),   width = 11,justify='right'))
      cat('\n')
    }
  }

  if (quietly==FALSE){
    cat(paste0(rep('=',80),collapse=''))
  }

  ###############################################################################
  # Find recommended window
  ###############################################################################

  if (!missing(X)){
    Pvals <- table_rdw[,1]

    if (!is.na(Pvals[1]) & Pvals[1]<level){
      cat('\n')
      cat('Smallest window does not pass covariate test.')
      cat('\n')
      cat('Decrease smallest window or reduce level.')
      tmp <- -1
      rec_length <- NA
      rec_window <- NA
    } else if (all(Pvals>=level)){
      tmp <- length(Pvals)
      rec_window <- c(cutoff+table_rdw[tmp,6],cutoff+table_rdw[tmp,7])
    } else{
      tmp <- min(which(Pvals<level))
      tmp <- tmp - 1
      rec_window <- c(cutoff+table_rdw[tmp,6],cutoff+table_rdw[tmp,7])
    }

    if (quietly==FALSE & tmp!=-1){
      cat('\n')
      cat(paste0('Recommended window is [',round(rec_window[1],4),';',round(rec_window[2],4),'] with ',table_rdw[tmp,4]+table_rdw[tmp,5],' observations (',
                 table_rdw[tmp,4],' below, ',table_rdw[tmp,5],' above).'))
      cat('\n\n')
    }

  } else {
    if(quietly==FALSE){
      cat('\n')
      cat('Note: no covariates specified.')
      cat('\n')
      cat(('Need to specify covariates to find recommended length.'))
    }
    rec_window <- NA
  }


  ###############################################################################
  # Plot p-values
  ###############################################################################

  if (plot==TRUE){
    if (!missing(X)){
      plot(wlist_right,Pvals)
    } else {
      stop('Cannot draw plot without covariates')
    }
  }


  ###############################################################################
  # Output
  ###############################################################################

  colnames(table_sumstats) <- c('Left of c','Right of c')
  rownames(table_sumstats) <- c('Number of obs','1th percentile','5th percentile','10th percentile','20th percentile')

  colnames(table_rdw) <- c('p-value','Variable','Bi.test','Obs<c','Obs>=c','w_left','w_right')

  output <- list(w_left  = rec_window[1],
                 w_right = rec_window[2],
                 wlist_left = wlist_left,
                 wlist_right = wlist_right,
                 results = table_rdw,
                 summary = table_sumstats)

  return(output)

}
