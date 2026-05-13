#################################################################
# Auxiliary functions for rdlocrand
# !version 1.1 22-May-2025
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
#################################################################

rdlocrand_rng_env <- new.env(parent = emptyenv())
rdlocrand_rng_env$depth <- 0L
rdlocrand_rng_env$had_seed <- FALSE
rdlocrand_rng_env$seed <- NULL

rdlocrand_seed_scope <- function(seed){
  if (length(seed)!=1L || is.na(seed) || !is.numeric(seed) || !(seed>0 || seed==-1)){
    stop('Seed has to be a positive integer or -1 for system seed')
  }

  env <- rdlocrand_rng_env
  outermost <- env$depth==0L

  if (outermost){
    env$had_seed <- exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    if (env$had_seed){
      env$seed <- get(".Random.seed", envir = .GlobalEnv, inherits = FALSE)
    } else {
      env$seed <- NULL
    }
  }

  env$depth <- env$depth + 1L

  if (seed>0){
    set.seed(seed)
  }

  restored <- FALSE
  function(){
    if (restored){
      return(invisible(NULL))
    }
    restored <<- TRUE
    env$depth <- max(0L, env$depth - 1L)
    if (outermost){
      if (env$had_seed){
        assign(".Random.seed", env$seed, envir = .GlobalEnv)
      } else if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)){
        rm(".Random.seed", envir = .GlobalEnv)
      }
      env$had_seed <- FALSE
      env$seed <- NULL
    }
    invisible(NULL)
  }
}

rdlocrand_validate_choice <- function(value, choices, message){
  if (length(value)!=1L || is.na(value) || !(value %in% choices)){
    stop(message, call. = FALSE)
  }
  invisible(value)
}


#################################################################
# rdrandinf observed statistics and asymptotic p-values
#################################################################

ksmirnov.statistic <- function(x,y){
  n.x <- length(x)
  n.y <- length(y)
  z <- c(x,y)
  w <- c(rep(1/n.x,n.x),rep(-1/n.y,n.y))
  max(abs(cumsum(w[order(z)])))
}

rdrandinf.model <- function(Y,D,statistic,pvalue=FALSE,kweights,endogtr,delta=NULL){

  n <- length(D)
  n1 <- sum(D)
  n0 <- n - n1

  Y <- as.matrix(Y)

  if (statistic=='ttest'|statistic=='diffmeans'){
    if (all(kweights==1)){
      Y1 <- t(Y[D==1,])
      Y0 <- t(Y[D==0,])
      M1 <- rowMeans(Y1)
      M0 <- rowMeans(Y0)
      stat <- M1-M0
      if (pvalue==TRUE){
        V1 <- rowMeans((Y1-rowMeans(Y1))^2)/(n1-1)
        V0 <- rowMeans((Y0-rowMeans(Y0))^2)/(n0-1)
        se <- sqrt(V1+V0)
        t.stat <- (M1-M0)/se
        asy.pval <- 2*pnorm(-abs(t.stat))
        if (!is.null(delta)){
          asy.power <- 1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se)
        } else{asy.power = NA}
      }
    } else {
      stat <- numeric(ncol(Y))
      asy.pval <- numeric(ncol(Y))
      for (k in 1:ncol(Y)){
        lm.aux <- lm(Y[,k] ~ D,weights=kweights)
        stat[k] <- lm.aux$coefficient['D']
        if (pvalue==TRUE){
          se <- sqrt(sandwich::vcovHC(lm.aux,type='HC2')['D','D'])
          t.stat <- stat[k]/se
          asy.pval[k] <- 2*pnorm(-abs(t.stat))
          if (!is.null(delta)){
            asy.power <- 1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se)
          } else {asy.power = NA}
        }
      }
    }
  }

  if (statistic=='ksmirnov'){
    stat <- numeric(ncol(Y))
    if (pvalue==TRUE){
      asy.pval <- numeric(ncol(Y))
      asy.power <- NA
    }
    for (k in 1:ncol(Y)){
      if (pvalue==TRUE){
        aux.ks <- ks.test(Y[D==0,k],Y[D==1,k])
        stat[k] <- aux.ks$statistic
        asy.pval[k] <- aux.ks$p.value
      } else {
        stat[k] <- ksmirnov.statistic(Y[D==0,k],Y[D==1,k])
      }
    }
  }

  if (statistic=='ranksum'){
    stat <- numeric(ncol(Y))
    for (k in 1:ncol(Y)){
      ri <- rank(Y[,k])
      Tstat <- sum(ri[D==0])
      s2 <- var(ri)
      ETstat <- n0*(n+1)/2
      VTstat <- n0*n1*s2/n
      se <- sqrt(VTstat)
      stat[k] <- (Tstat-ETstat)/se
    }
    if (pvalue==TRUE){
      asy.pval <- 2*pnorm(-abs(stat))
      sigma <- sd(Y)
      if (!is.null(delta)){
        asy.power <- pnorm(sqrt(3*n0*n1/((n0+n1+1)*pi))*delta/sigma-1.96)
      } else {asy.power = NA}
    }
  }

  if (statistic=='all'){
    stat1 <- mean(Y[D==1])-mean(Y[D==0])
    if (pvalue==TRUE){
      aux.ks <- ks.test(Y[D==0],Y[D==1])
      stat2 <- aux.ks$statistic
    } else {
      stat2 <- ksmirnov.statistic(Y[D==0],Y[D==1])
    }
    Tstat <- sum(rank(Y)[D==0])
    ri <- seq(1,n)
    s2 <- var(ri)
    ETstat <- n0*(n+1)/2
    VTstat <- n0*n1*s2/n
    se3 <- sqrt(VTstat)
    stat3 <- (Tstat-ETstat)/se3
    stat <- c(stat1,stat2,stat3)
    if (pvalue==TRUE){
      se1 <- sqrt(var(Y[D==1])/n1+var(Y[D==0])/n0)
      t.stat <- stat1/se1
      asy.pval1 <- 2*pnorm(-abs(t.stat))
      asy.pval2 <- aux.ks$p.value
      asy.pval3 <- 2*pnorm(-abs(stat3))
      asy.pval <- c(asy.pval1,asy.pval2,asy.pval3)
      if (!is.null(delta)){
        asy.power1 <- 1-pnorm(1.96-delta/se1)+pnorm(-1.96-delta/se1)
        asy.power2 <- NA
        sigma <- sd(Y)
        asy.power3 <- pnorm(sqrt(3*n0*n1/((n0+n1+1)*pi))*delta/sigma-1.96)
        asy.power <- c(asy.power1,asy.power2,asy.power3)
      } else {asy.power <- c(NA,NA,NA)}
    }
  }

  if (statistic=='ar'){
    stat = mean(Y[D==1])-mean(Y[D==0])
    if (pvalue==TRUE){
      se <- sqrt(var(Y[D==1])/n1+var(Y[D==0])/n0)
      t.stat <- stat/se
      asy.pval <- 2*pnorm(-abs(t.stat))
      if (!is.null(delta)){
        asy.power <- 1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se)
      } else {asy.power <- NA}
    }
  }

  if (statistic=='wald'){
    fs <- lm(endogtr ~ D)
    rf <- lm(Y ~ D)
    stat <- rf$coefficients['D']/fs$coefficients['D']
    if (pvalue==TRUE){
      ehat <- Y - mean(Y) - stat*(endogtr - mean(endogtr))
      ehat2 <- ehat^2
      se <- sqrt((mean(ehat2)*var(D))/(n*cov(D,endogtr)^2))
      tstat <- stat/se
      asy.pval <- 2*pnorm(-abs(tstat))
      if (!is.null(delta)){
        asy.power <- 1-pnorm(1.96-delta/se)+pnorm(-1.96-delta/se)
      } else {asy.power <- NA}
    }
  }


  if (pvalue==TRUE){
    output <- list(statistic = as.numeric(stat),p.value = as.numeric(asy.pval),asy.power = as.numeric(asy.power))
  } else{
      output <- list(statistic = as.numeric(stat))
  }

  return(output)
}

#################################################################
# Fast Bernoulli p-value for rank-sum randomization tests
#################################################################

rdrandinf.bernoulli.ranksum.pvalue <- function(Y,R,prob,reps,nulltau=0,seed=666){

  D <- as.numeric(R >= 0)
  Y.adj.null <- Y - nulltau*D
  ranks <- rank(Y.adj.null)
  rank.var <- var(ranks)
  n <- length(D)

  ranksum.stat <- function(D.sample){
    n1 <- sum(D.sample)
    n0 <- n - n1
    Tstat <- sum(ranks[D.sample==0])
    ETstat <- n0*(n+1)/2
    VTstat <- n0*n1*rank.var/n
    (Tstat-ETstat)/sqrt(VTstat)
  }

  obs.stat <- ranksum.stat(D)
  stats.distr <- array(NA,dim=c(reps,1))

  restore_rng <- rdlocrand_seed_scope(seed)
  on.exit(restore_rng(), add = TRUE)

  for (i in 1:reps) {
    D.sample <- as.numeric(runif(n)<=prob)
    if (mean(D.sample)==1 | mean(D.sample)==0){
      stats.distr[i,] <- NA
    } else {
      stats.distr[i,] <- ranksum.stat(D.sample)
    }
  }

  mean(abs(stats.distr) >= abs(obs.stat),na.rm=TRUE)
}


#################################################################
# Hotelling's T2 statistic
#################################################################

hotelT2 <- function(X,D) {

  n <- length(D)
  n1 <- sum(D)
  n0 <- n - n1
  p <- ncol(X)

  X1 <- X[D==1,]
  X0 <- X[D==0,]
  X1bar <- as.matrix(colMeans(X[D==1,]))
  X0bar <- as.matrix(colMeans(X[D==0,]))
  S1 <- cov(X1)
  S0 <- cov(X0)
  Spool <- (S1*(n1-1)+S0*(n0-1))/(n-2)
  SpoolInv <- solve(Spool)

  T2 <- (n0*n1/n)*t(X1bar-X0bar)%*%SpoolInv%*%(X1bar-X0bar)
  Fstat <- ((n-p-1)/((n-2)*p))*T2
  pval <- 1-pf(Fstat,p,n-1-p)

  output <- list(statistic = as.numeric(T2),Fstat = as.numeric(Fstat),p.value = as.numeric(pval))
  names(output) <- c('statistic','Fstat','p.value')

  return(output)
}

#################################################################
# Find window increments
#################################################################

findwobs <- function(wobs,nwin,posl,posr,R,dups){

  N <- length(R)
  Nc <- sum(R<0)
  Nt <- sum(R>=0)
  mpoints_l <- length(unique(R[1:Nc]))
  mpoints_r <- length(unique(R[(Nc+1):N]))
  mpoints_max <- max(mpoints_l,mpoints_r)
  nwin_mp <- min(nwin,mpoints_max)
  poslold <- posl
  posrold <- posr

  win <- 1
  wlist_left <- vector("list", nwin_mp)
  poslist_left <- vector("list", nwin_mp)
  wlist_right <- vector("list", nwin_mp)
  poslist_right <- vector("list", nwin_mp)

  #while(win<=nwin & wobs<min(posl,Nt-(posr-Nc-1))){
  while(win<=nwin_mp & wobs<max(posl,Nt-(posr-Nc-1))){

    poslold <- posl
    posrold <- posr

    while(dups[posl]<wobs & sum(R[posl]<=R[posl:poslold])<wobs & posl>1){
      posl <- max(posl - dups[posl],1)
    }

    while(dups[posr]<wobs & sum(R[posrold:posr]<=R[posr])<wobs & posr<N){
      posr <- min(posr + dups[posr],N)
    }

    wlength_left <- R[posl]
    wlength_right <- R[posr]

    wlist_left[[win]] <- wlength_left
    poslist_left[[win]] <- posl
    wlist_right[[win]] <- wlength_right
    poslist_right[[win]] <- posr

    posl <- max(posl - dups[posl],1)
    posr <- min(posr + dups[posr],N)

    win <- win + 1

  }

  used_windows <- win - 1
  if (used_windows > 0){
    keep_windows <- seq_len(used_windows)
    wlist_left <- unlist(wlist_left[keep_windows])
    poslist_left <- unlist(poslist_left[keep_windows])
    wlist_right <- unlist(wlist_right[keep_windows])
    poslist_right <- unlist(poslist_right[keep_windows])
  } else {
    wlist_left <- NULL
    poslist_left <- NULL
    wlist_right <- NULL
    poslist_right <- NULL
  }

  output <- list(posl = posl, posr = posr, wlength_left = wlength_left, wlength_right = wlength_right,
                 wlist_left = wlist_left, wlist_right = wlist_right,
                 poslist_left = poslist_left, poslist_right = poslist_right)

  return(output)

}


#################################################################
# Find symmetric window increments
#################################################################

findwobs_sym <- function(wobs,nwin,posl,posr,R,dups){

  N <- length(R)
  Nc <- sum(R<0)
  Nt <- sum(R>=0)
  poslold <- posl
  posrold <- posr
  wlist <- vector("list", nwin)
  win <- 1

  while(win<=nwin & wobs<min(posl,Nt-(posr-Nc-1))){

    poslold <- posl
    posrold <- posr

    while(dups[posl]<wobs & sum(R[posl]<=R[posl:poslold])<wobs){
      posl <- max(posl - dups[posl],1)
    }

    while(dups[posr]<wobs & sum(R[posrold:posr]<=R[posr])<wobs){
      posr <- min(posr + dups[posr],N)
    }

    if(abs(R[posl])<R[posr]){
      posl <- Nc + 1 - sum(-R[posr]<=R[1:Nc])
    }

    if(abs(R[posl])>R[posr]){
      posr <- sum(R[(Nc+1):N]<=abs(R[posl])) + Nc
    }

    wlength <- max(-R[posl],R[posr])
    wlist[[win]] <- wlength

    posl <- max(posl - dups[posl],1)
    posr <- min(posr + dups[posr],N)
    win <- win + 1

  }

  used_windows <- win - 1
  if (used_windows > 0){
    wlist <- unlist(wlist[seq_len(used_windows)])
  } else {
    wlist <- NULL
  }

  return(wlist)

}

#################################################################
# Find CI
#################################################################

find_CI <- function(pvals,alpha,tlist){
  if(all(pvals>=alpha)){
    CI <- matrix(c(tlist[1],tlist[length(tlist)]),nrow=1,ncol=2)
  } else if(all(pvals<alpha)){
    CI <- matrix(NA,nrow=1,ncol=2)
  } else{
    whichvec <- which(pvals>=alpha)
    breaks <- c(0,which(diff(whichvec)!=1),length(whichvec))
    indexmat <- cbind(whichvec[breaks[-length(breaks)]+1],whichvec[breaks[-1]])
    CI <- t(apply(indexmat,1,function(x) tlist[x]))
  }
  return(CI)
}

#################################################################
# Find window length - DEPRECATED: for backward compatibility
#################################################################

wlength <- function(R,D,num){

  X1 <- R[D==1]
  X1 <- sort(abs(X1))
  X0 <- R[D==0]
  X0 <- sort(abs(X0))
  m <- min(length(X1),length(X0))
  if (num>m){
    num <- m
  }
  xt <- X1[num]
  xc <- X0[num]
  minw <- max(xc,xt)
  return(minw)
}

#################################################################
# Find default step - DEPRECATED: for backward compatibility
#################################################################

findstep <- function(R,D,obsmin,obsstep,times) {
  step_indices <- 1:times
  S <- numeric(length(step_indices))
  for (j in seq_along(step_indices)){
    i <- step_indices[j]
    U <- wlength(R,D,obsmin+obsstep*i)
    L <- wlength(R,D,obsmin+obsstep*(i-1))
    Snext <- U - L
    S[j] <- Snext
  }
  step <- max(S)
  return(step)
}

