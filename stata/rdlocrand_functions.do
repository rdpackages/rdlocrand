********************************************************************************
* RDLOCRAND package: auxiliary functions
* !version 0.7.1 2020-08-22
* Authors: Matias Cattaneo, Rocio Titiunik, Gonzalo Vazquez-Bare
********************************************************************************

version 13

********************************************************************************
** rdlocrand_findwobs(): find window increments (rdwinselect)
********************************************************************************

capture mata: mata drop rdlocrand_findwobs()
mata

void rdlocrand_findwobs(real scalar wobs, 
			  real scalar nwin, 
			  real scalar posl, 
			  real scalar posr,
			  runvar,dupsvar)
{

	st_view(R=.,.,runvar,"`touse'")
	st_view(dups=.,.,dupsvar,"`touse'")

	N = length(R)
	Nc = sum(R:<0)
	Nt = sum(R:>=0)
	poslold = posl
	posrold = posr

	win = 1
	wlist = .

	while((win <= nwin) & (wobs < min((posl,Nt-(posr-Nc-1))))) {

		poslold = posl
		posrold = posr

		while((dups[posl]<wobs) & (sum(R[posl] :<= R[posl..poslold])<wobs)) {
			posl = max((posl - dups[posl],1))
		}

		while((dups[posr]<wobs) & (sum(R[posrold..posr] :<= R[posr]))<wobs) {
			posr = min((posr + dups[posr],N))
		}

		if (abs(R[posl]) < R[posr]) {
			posl = Nc + 1 - sum(-R[posr] :<= R[1..Nc])
		}
		
		if (abs(R[posl]) > R[posr]) {
			posr = sum(R[(Nc+1)..N] :<= abs(R[posl])) + Nc
		}
		
		wlength = max((-R[posl],R[posr]))
		
		wlist = (wlist,wlength)

		posl = max((posl - dups[posl],1))
		posr = min((posr + dups[posr],N))
		win = win + 1
	}
	
	st_numscalar("posl",posl)
	st_numscalar("posr",posr)
	st_numscalar("wlength",wlength)
	st_matrix("wlist",wlist)
}

mata mosave rdlocrand_findwobs(), replace

end


********************************************************************************
** rdlocrand_reclength(): find recommended length (rdwinselect)
********************************************************************************

capture mata: mata drop rdlocrand_reclength()
mata:
void rdlocrand_reclength(Mp,real scalar level)
{
	if (Mp[1]<level) {
		ind = .
	} else if (sum(Mp:<level)==0){
		ind = rows(Mp)
	} else {
		aux = select(Mp,Mp:<level)
		maxindex(Mp:==aux[1],1,ind=.,x=.)
		ind = ind[1] - 1
	}
	st_numscalar("index",ind)
}

mata mosave rdlocrand_reclength(), replace

end

********************************************************************************
** rdlocrand_confint: calculate confidence interval (rdsensitivity)
********************************************************************************

capture mata: mata drop rdlocrand_confint()
mata:
void rdlocrand_confint(real scalar colci, real scalar level, real matrix T)
{
	R=st_matrix("Res")
	if (!allof(R[,colci]:>=level,0)){
		maxindex(R[,colci]:>=level,1,i=.,j=.)
		st_numscalar("cilb",T[i[1,1],1])
		st_numscalar("ciub",T[i[rows(i),1],1])
	}
	else {
		st_numscalar("cilb",.)
		st_numscalar("ciub",.)
	}

}

mata mosave rdlocrand_confint(), replace

end

********************************************************************************
** rdlocrand_wlength(): find window length - DEPRECATED: for backward compatibility only
********************************************************************************

capture mata: mata drop rdlocrand_wlength()
mata:
function rdlocrand_wlength(runv, treat, cont, real scalar num)
{
	real scalar xt
	real scalar xc
	
	st_view(R=.,.,runv)
	x0 = min(max(R)\abs(min(R)))
	st_view(X=.,.,runv,treat)
	X = sort(X,1)
	st_view(Y=.,.,runv,cont)
	Z = sort(abs(Y),1)
	m = min(length(X)\length(Z))
	if (m<num) num=m
	xt = X[num]
	xc = Z[num]
	minw = max(xt\xc)
	
	return(xt,xc,minw)
}

mata mosave rdlocrand_wlength(), replace

end

********************************************************************************
** rdlocrand_findstep(): find step - DEPRECATED: for backward compatibility only
********************************************************************************

capture mata: mata drop rdlocrand_findstep()
mata:
void rdlocrand_findstep(real scalar minobs,real scalar addobs,real scalar times, runv, treat, cont)
{
	S = rdlocrand_wlength(runv,treat,cont,minobs+addobs)-rdlocrand_wlength(runv,treat,cont,minobs)
	for(i=1; i<=times-1; i++) {
		U = rdlocrand_wlength(runv,treat,cont,minobs+addobs*i)
		L = rdlocrand_wlength(runv,treat,cont,minobs+addobs*(i-1))
		Snext = U-L
		S = (S\Snext)
	}
	step = max(S)
	st_numscalar("step",step)
}

mata mosave rdlocrand_findstep(), replace

end
