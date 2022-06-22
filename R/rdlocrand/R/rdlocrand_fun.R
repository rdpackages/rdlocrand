#################################################################
# Auxiliary functions for rdlocrand
# !version 1.0 21-Jun-2022
# Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
#################################################################


#################################################################
# rdrandinf observed statistics and asymptotic p-values
#################################################################

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
    stat <- NULL
    asy.pval <- NULL
    for (k in 1:ncol(Y)){
      aux.ks <- ks.test(Y[D==0,k],Y[D==1,k])
      stat <- c(stat,aux.ks$statistic)
      if (pvalue==TRUE){
        asy.pval <- c(asy.pval,aux.ks$p.value)
        asy.power <- NA
      }
    }
  }

  if (statistic=='ranksum'){
    stat <- NULL
    asy.pval <- NULL
    for (k in 1:ncol(Y)){
      Ustat <- wilcox.test(Y[D==0,k],Y[D==1,k])$statistic
      Tstat <- Ustat + n0*(n0+1)/2
      ri <- rank(Y[,k])
      s2 <- var(ri)
      ETstat <- n0*(n+1)/2
      VTstat <- n0*n1*s2/n
      se <- sqrt(VTstat)
      stat <- c(stat,(Tstat-ETstat)/se)
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
    aux.ks <- ks.test(Y[D==0],Y[D==1])
    stat2 <- aux.ks$statistic
    Ustat <- wilcox.test(Y[D==0],Y[D==1])$statistic
    Tstat <- Ustat + n0*(n0+1)/2
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
  wlist_left <- NULL
  poslist_left <- NULL
  wlist_right <- NULL
  poslist_right <- NULL

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

    wlist_left <- c(wlist_left,wlength_left)
    poslist_left <- c(poslist_left,posl)
    wlist_right <- c(wlist_right,wlength_right)
    poslist_right <- c(poslist_right,posr)

    posl <- max(posl - dups[posl],1)
    posr <- min(posr + dups[posr],N)

    win <- win + 1

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
  wlist <- NULL
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
    wlist <- c(wlist,wlength)

    posl <- max(posl - dups[posl],1)
    posr <- min(posr + dups[posr],N)
    win <- win + 1

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
    index_l <- min(whichvec)
    index_r <- max(whichvec)
    indexmat <- matrix(c(index_l,index_r),nrow=1,ncol=2)

    whichvec_cut <- whichvec
    dif <- diff(whichvec_cut)
    while(all(dif==1)==FALSE){
      cut <- min(which(dif!=1))
      auxvec <- whichvec_cut[1:cut]
      indexmat <- rbind(indexmat,c(min(auxvec),max(auxvec)))
      whichvec_cut <- whichvec_cut[(cut+1):length(whichvec_cut)]

      dif <- diff(whichvec_cut)
    }
    if(nrow(indexmat)>1){
      indexmat <- indexmat[2:nrow(indexmat),]
      indexmat <- rbind(indexmat,c(min(whichvec_cut),max(whichvec_cut)))
    }
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
  S <- NULL
  for (i in 1:times){
    U <- wlength(R,D,obsmin+obsstep*i)
    L <- wlength(R,D,obsmin+obsstep*(i-1))
    Snext <- U - L
    S <- c(S,Snext)
  }
  step <- max(S)
  return(step)
}

