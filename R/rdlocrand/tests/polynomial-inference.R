library(rdlocrand)
quiet <- function(expr) {
  value <- NULL
  invisible(capture.output(value <- force(expr)))
  value
}
equal <- function(actual, expected, tolerance = 1e-8) {
  stopifnot(isTRUE(all.equal(unname(actual), unname(expected), tolerance = tolerance, check.attributes = FALSE)))
}
r <- 2+c(seq(-.9, -.015, length.out=48), seq(.01, 1.1, length.out=63))
d <- as.numeric(r>=2); i <- seq_along(r)
t <- .7*d+.4*sin(i*.41)+.2*(r-2)+.1*d*(r-2)
y <- 1.2*t+.5*(r-2)-.25*d*(r-2)+(sin(i*1.7)+.4*cos(i*.33))*(1+.5*(r-2))
base <- list(Y=y, R=r, cutoff=2, wl=1.1, wr=3.1, quietly=TRUE, nulltau=.6)
run <- function(...) quiet(do.call(rdrandinf, c(base, list(...))))

# Independent OLS sandwich and IV moment-matrix sandwich; HC1 IV also matches AER.
for (p in 1:2) for (kernel in c('uniform','triangular','epan')) for (vce in c('HC1','HC2','HC3')) {
  u <- (r-2)/ifelse(d==1,1.1,-.9)
  w <- switch(kernel, uniform=rep(1,length(r)), triangular=pmax(1-abs(u),0), epan=.75*pmax(1-u^2,0))
  keep <- w>1e-14; w <- w[keep]; dw <- d[keep]; tw <- t[keep]
  powers <- outer(r[keep]-2, seq_len(p), '^')
  z <- cbind(1, dw, powers, dw*powers)
  for (kind in c('sharp','ar','tsls')) {
    yy <- if (kind=='sharp') y[keep]-.6*dw else if (kind=='ar') y[keep]-.6*tw else y[keep]
    fuzzy <- if(kind=='sharp') NULL else if(kind=='ar') t else c(t,'tsls')
    if (kind!='tsls') {
      fit <- lm(yy ~ z-1, weights=w)
      estimate <- coef(fit)[2]
      se <- sqrt(sandwich::vcovHC(fit,type=vce)[2,2])
    } else {
      x <- z; x[,2] <- tw
      zw <- z*sqrt(w); xw <- x*sqrt(w)
      projected <- zw %*% solve(crossprod(zw),crossprod(zw,xw))
      bread <- solve(crossprod(projected))
      beta <- bread %*% crossprod(projected,yy*sqrt(w))
      residual <- as.numeric(yy-x%*%beta)*sqrt(w)
      h <- rowSums((projected%*%bread)*projected)
      scale <- switch(vce,HC1=length(yy)/(length(yy)-ncol(x)),HC2=1/(1-h),HC3=1/(1-h)^2)
      cov <- bread %*% crossprod(projected,projected*as.numeric(residual^2*scale)) %*% bread
      estimate <- beta[2]; se <- sqrt(cov[2,2])
      if (vce=='HC1') {
        fit <- AER::ivreg(yy ~ tw+powers+dw:powers | dw+powers+dw:powers, weights=w)
        equal(c(estimate,se),c(coef(fit)['tw'],sqrt(sandwich::vcovHC(fit,type='HC1')['tw','tw'])))
      }
    }
    out <- run(p=p,kernel=kernel,vce=vce,fuzzy=fuzzy)
    equal(c(out$obs.stat,out$se),c(estimate,se))
    equal(out$asy.pvalue,2*pnorm(-abs((estimate-if(kind=='tsls') .6 else 0)/se)))
    stopifnot(is.na(out$p.value),out$inference=='large-sample')
  }
}

# HC3 default, custom evaluation points, normal CIs, and inversion of AR tests.
out <- run(p=2, evall=1.85, evalr=2.15)
powers <- outer(r-d*2.15-(1-d)*1.85,1:2,'^')
fit <- lm(I(y-.6*d) ~ d*powers)
equal(c(out$obs.stat,out$se),c(coef(fit)['d'],sqrt(sandwich::vcovHC(fit,type='HC3')['d','d'])))
stopifnot(out$vce=='HC3')
for (kind in c('sharp','tsls')) {
  out <- run(p=1,ci=.1,fuzzy=if(kind=='sharp') NULL else c(t,'tsls'))
  estimate <- out$obs.stat+if(kind=='sharp') .6 else 0
  equal(as.numeric(out$ci),estimate+c(-1,1)*qnorm(.95)*out$se)
}
grid <- seq(-2,4,.25)
find_ci <- getFromNamespace('find_CI','rdlocrand')
for (fuzzy in list(NULL,t)) {
  out <- run(p=2,kernel='triangular',fuzzy=fuzzy,ci=c(.1,grid))
  sens <- quiet(rdsensitivity(y,r,cutoff=2,wlist=3.1,wlist_left=1.1,p=2,fuzzy=fuzzy,
                             tlist=grid,kernel='triangular',nodraw=TRUE,quietly=TRUE))
  equal(out$ci,find_ci(sens$results[,1],.1,grid))
}
equal(find_ci(c(.2,0,.4,.4,0),.1,0:4),matrix(c(0,0,2,3),2,byrow=TRUE))
sens <- quiet(rdsensitivity(y,r,cutoff=2,wlist=3.1,wlist_left=1.1,p=1,tlist=.6,
                           evalat='means',nodraw=TRUE,quietly=TRUE))
out <- run(p=1,evall=mean(r[d==0]),evalr=mean(r[d==1]))
equal(sens$results[1,1],out$asy.pvalue)

# The documented list input and legacy flat-vector input select the same fuzzy test.
for (p in 0:2) for (method in c('ar', 'itt', 'tsls')) {
  a <- run(p=p, fuzzy=list(t,method), reps=29)
  b <- run(p=p, fuzzy=c(t,method), reps=29)
  for (key in c('obs.stat','p.value','asy.pvalue','se')) equal(a[[key]],b[[key]])
}

# Removing incomplete fuzzy observations is identical to supplying complete data.
yn <- y; rn <- r; tn <- t; yn[4]<-NA; rn[12]<-NA; tn[17]<-NA
keep <- complete.cases(yn,rn,tn)
for (kind in c('ar','tsls')) {
  a <- quiet(rdrandinf(yn,rn,cutoff=2,wl=1.1,wr=3.1,p=1,quietly=TRUE,
                      fuzzy=if(kind=='ar') tn else c(tn,'tsls')))
  b <- quiet(rdrandinf(yn[keep],rn[keep],cutoff=2,wl=1.1,wr=3.1,p=1,quietly=TRUE,
                      fuzzy=if(kind=='ar') tn[keep] else c(tn[keep],'tsls')))
  equal(c(a$obs.stat,a$se),c(b$obs.stat,b$se))
}

fallback <- function(fun,args,keys) {
  messages <- character()
  a <- withCallingHandlers(quiet(do.call(fun,c(args,list(p=2)))),warning=function(w){
    messages <<- c(messages,conditionMessage(w)); invokeRestart('muffleWarning')
  })
  stopifnot(sum(grepl('Polynomial adjustment',messages))==1,a$p.requested==2,a$p==0)
  b <- quiet(do.call(fun,c(args,list(p=0))))
  for (key in keys) equal(a[[key]],b[[key]])
}
for (stat in c('ksmirnov','ranksum','all'))
  fallback(rdrandinf,c(base,list(statistic=stat,reps=29)),c('obs.stat','p.value','asy.pvalue'))
fallback(rdrandinf,c(base,list(interfci=.1,reps=29)),c('interf.ci','p.value'))
for (stat in c('ksmirnov','ranksum','hotelling'))
  fallback(rdwinselect,list(R=r,X=cbind(y,t),cutoff=2,wmin=.8,wstep=.1,nwindows=2,
                           statistic=stat,reps=29,quietly=TRUE),'results')
fallback(rdsensitivity,list(Y=y,R=r,cutoff=2,wlist=3.1,wlist_left=1.1,tlist=c(0,1),
                            statistic='ksmirnov',reps=29,nodraw=TRUE,quietly=TRUE),'results')
fallback(rdrbounds,list(Y=y,R=r-2,wlist=.8,expgamma=1.2,reps=19),c('p.values','lower.bound','upper.bound'))

for (opts in list(list(p=-1),list(p=.5),list(p=1,vce='HC0')))
  stopifnot(inherits(try(quiet(do.call(rdrandinf,c(base,opts))),silent=TRUE),'try-error'))
for (rr in list(c(rep(-1,10),rep(1,10)),c(-.2,-.1,seq(.1,1,length.out=10))))
  stopifnot(inherits(try(quiet(rdrandinf(sin(seq_along(rr)),rr,p=1,wl=-1,wr=1,quietly=TRUE)),silent=TRUE),'try-error'))
stopifnot(inherits(try(run(p=1,fuzzy=c(r-2,'tsls')),silent=TRUE),'try-error'))
messages <- character()
x <- matrix(cos(seq_along(r)*.71),ncol=1)
out <- withCallingHandlers(quiet(rdrandinf(y,r,cutoff=2,covariates=x,p=1,rdwstat='ksmirnov',
                       wmin=.5,wstep=.1,nwindows=2,level=0,rdwreps=29,quietly=TRUE)),
                       warning=function(w){messages <<- c(messages,conditionMessage(w));invokeRestart('muffleWarning')})
stopifnot(out$p==1,sum(grepl('Polynomial adjustment',messages))==1)
rr <- seq(-1,1,length.out=80); yy <- sin(seq_along(rr))
for(vce in c('HC1','HC2','HC3')) {
  balance <- quiet(rdwinselect(rr,matrix(yy,ncol=1),p=1,vce=vce,kernel='triangular',
                             wmin=1,wstep=.1,nwindows=1,quietly=TRUE))
  direct <- quiet(rdrandinf(yy,rr,p=1,vce=vce,kernel='triangular',wl=-1,wr=1,quietly=TRUE))
  equal(balance$results[1,1],direct$asy.pvalue)
}
# p=0 TSLS honors the tested null and requested confidence level, retaining HC1.
for (kernel in c('uniform','triangular','epan')) {
  u <- (r-2)/ifelse(d==1,1.2,-1)
  weights <- switch(kernel,uniform=rep(1,length(r)),triangular=1-abs(u),epan=.75*(1-u^2))
  fit <- AER::ivreg(y ~ t | d,weights=weights)
  beta <- unname(coef(fit)['t']);se <- sqrt(sandwich::vcovHC(fit,type='HC1')['t','t'])
  for (tau in c(0,.6,beta)) for (alpha in c(.2,.1,.05)) {
    out <- quiet(rdrandinf(y,r,cutoff=2,wl=1,wr=3.2,p=0,fuzzy=c(t,'tsls'),
                         kernel=kernel,nulltau=tau,ci=alpha,quietly=TRUE))
    equal(out$obs.stat,beta)
    equal(out$asy.pvalue,2*pnorm(-abs((beta-tau)/se)))
    equal(as.numeric(out$ci),beta+c(-1,1)*qnorm(1-alpha/2)*se)
  }
}
cat('Polynomial HC and fallback checks passed.\n')
