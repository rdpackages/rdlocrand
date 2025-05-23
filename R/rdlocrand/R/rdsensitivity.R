###############################################################################
# rdsensitivity: sensitivity analysis for randomization inference in RD
# !version 1.1 22-May-2025
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
###############################################################################

#' Sensitivity analysis for RD designs under local randomization
#'
#' \code{rdsensitivity} analyze the sensitivity of randomization p-values
#' and confidence intervals to different window lengths.
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
#' @param wlist the list of windows to the right of the cutoff. By default the program constructs 10 windows around the cutoffwith 5 observations each.
#' @param wlist_left the list of windows  to the left of the cutoff. If not specified, the windows are constructed symmetrically around the cutoff based on the values in wlist.
#' @param tlist the list of values of the treatment effect under the null to be evaluated. By default the program employs ten evenly spaced points within the asymptotic confidence interval for a constant treatment effect in the smallest window to be used.
#' @param statistic the statistic to be used in the balance tests. Allowed options are \code{diffmeans} (difference in means statistic), \code{ksmirnov} (Kolmogorov-Smirnov statistic) and \code{ranksum} (Wilcoxon-Mann-Whitney standardized statistic). Default option is \code{diffmeans}. The statistic \code{ttest} is equivalent to \code{diffmeans} and included for backward compatibility.
#' @param p the order of the polynomial for outcome adjustment model. Default is 0.
#' @param evalat specifies the point at which the adjusted variable is evaluated. Allowed options are \code{cutoff} and \code{means}. Default is \code{cutoff}.
#' @param kernel specifies the type of kernel to use as weighting scheme. Allowed kernel types are \code{uniform} (uniform kernel), \code{triangular} (triangular kernel) and \code{epan} (Epanechnikov kernel). Default is \code{uniform}.
#' @param fuzzy indicates that the RD design is fuzzy. \code{fuzzy} can be specified as a vector containing the values of the endogenous treatment variable, or as a list where the first element is the vector of endogenous treatment values and the second element is a string containing the name of the statistic to be used. Allowed statistics are \code{ar} (Anderson-Rubin statistic) and \code{tsls} (2SLS statistic). Default statistic is \code{ar}. The \code{tsls} statistic relies on large-sample approximation.
#' @param ci returns the confidence interval corresponding to the indicated window length. \code{ci} has to be a two-dimensional vector indicating the left and right limits of the window. Default alpha is .05 (95\% level CI).
#' @param ci_alpha Specifies value of alpha for the confidence interval. Default alpha is .05 (95\% level CI).
#' @param reps number of replications. Default is 1000.
#' @param seed the seed to be used for the randomization tests.
#' @param nodraw suppresses contour plot.
#' @param quietly suppresses the output table.
#'
#' @return
#' \item{tlist}{treatment effects grid}
#' \item{wlist}{window grid}
#' \item{results}{table with corresponding p-values for each window and treatment effect pair.}
#' \item{ci}{confidence interval (if \code{ci} is specified).}
#'
#' @examples
#' # Toy dataset
#' R <- runif(100,-1,1)
#' Y <- 1 + R -.5*R^2 + .3*R^3 + (R>=0) + rnorm(100)
#' # Sensitivity analysis
#' # Note: low number of replications to speed up process.
#' # The user should increase the number of replications.
#' tmp <- rdsensitivity(Y,R,wlist=seq(.75,2,by=.25),tlist=seq(0,5,by=1),reps=500)
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
  if (statistic!='diffmeans' & statistic!='ttest' & statistic!='ksmirnov' & statistic!='ranksum') stop(paste(statistic,'not a valid statistic'))
  if (evalat!='cutoff' & evalat!='means') stop('evalat only admits means or cutoff')
  if (missing(tlist) & p!=0) stop('need to specify tlist when p>0')
  if (!missing(wlist_left)){
    if (missing(wlist)) stop('Need to specify wlist when wlist_left is specified')
    if (length(wlist)!=length(wlist_left)) stop('Lengths of wlist and wlist_left need to coincide')
  }
  if(!is.null(ci) & length(ci)!=2) stop('Need to specify wleft and wright in CI option')

  if (seed>0){
    set.seed(seed)
  } else if (seed!=-1){
    stop('Seed has to be a positive integer or -1 for system seed')
  }

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
