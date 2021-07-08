********************************************************************************
* RDLOCRAND package: auxiliary functions
* !version 1.0 2021-07-07
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
	mpoints_l = length(uniqrows(R[1..Nc]))
	mpoints_r = length(uniqrows(R[(Nc+1)..N]))
	mpoints_max = max((mpoints_l,mpoints_r))
	nwin_mp = min((nwin,mpoints_max))
	poslold = posl
	posrold = posr

	win = 1
	wlist_left = .
	poslist_left = .
	wlist_right = .
	poslist_right = .
	
	//while((win <= nwin) & (wobs < min((posl,Nt-(posr-Nc-1))))) {
	while((win <= nwin_mp) & (wobs < max((posl,Nt-(posr-Nc-1)))) ) {

		poslold = posl
		posrold = posr

		while((dups[posl]<wobs) & (sum(R[posl] :<= R[posl..poslold])<wobs) & (posl>1)) {
			posl = max((posl - dups[posl],1))
		}

		while((dups[posr]<wobs) & (sum(R[posrold..posr] :<= R[posr])<wobs) & (posr<N)) {
			posr = min((posr + dups[posr],N))
		}
			
		wlength_left = R[posl]
		wlength_right = R[posr]
		
		if(wlist_left==.){
		    wlist_left = wlength_left
			poslist_left = posl
			wlist_right = wlength_right
			poslist_right = posr
		}
		else {
		    wlist_left = (wlist_left,wlength_left)
			poslist_left = (poslist_left,posl)
			wlist_right = (wlist_right,wlength_right)
			poslist_right = (poslist_right,posr)
		}				

		posl = max((posl - dups[posl],1))
		posr = min((posr + dups[posr],N))
		
		win = win + 1
		
	}
	
	st_numscalar("posl",posl)
	st_numscalar("posr",posr)
	st_numscalar("wlength_left",wlength_left)
	st_numscalar("wlength_right",wlength_right)
	st_matrix("wlist_left",wlist_left)
	st_matrix("wlist_right",wlist_right)
	st_matrix("poslist_left",poslist_left)
	st_matrix("poslist_right",poslist_right)
}

mata mosave rdlocrand_findwobs(), replace

end


********************************************************************************
** rdlocrand_findwobs_sym(): find window symmetric increments (rdwinselect)
********************************************************************************

capture mata: mata drop rdlocrand_findwobs_sym()
mata

void rdlocrand_findwobs_sym(real scalar wobs, 
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
		
		if(wlist==.){
		    wlist = wlength
		}
		else {
		    wlist = (wlist,wlength)
		}		
		posl = max((posl - dups[posl],1))
		posr = min((posr + dups[posr],N))
		win = win + 1
	}
	
	st_numscalar("posl",posl)
	st_numscalar("posr",posr)
	st_numscalar("wlength",wlength)
	st_matrix("wlist",wlist)
}

mata mosave rdlocrand_findwobs_sym(), replace

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
void rdlocrand_confint(real matrix pvals, real scalar alpha, real matrix tlist)
{
	
	if(all(pvals:>=alpha)){
	   CI = (tlist[1],tlist[cols(tlist)]) 
	}
	else if (all(pvals:<alpha)){
	    CI = (.,.)
	}
	else {
	    whichvec = selectindex(pvals:>=alpha)
		index_l = min(whichvec)
		index_r = max(whichvec)
		indexmat = (index_l,index_r)
		
		whichvec_cut = whichvec
		dif = whichvec_cut[2..cols(whichvec_cut)] - whichvec_cut[1..(cols(whichvec_cut)-1)]
		while(any(dif:!=1)){
		    cut = min(selectindex(dif:!=1))
			auxvec = whichvec_cut[1..cut]
			indexmat = (indexmat \ (min(auxvec),max(auxvec)))
			whichvec_cut = whichvec_cut[(cut+1)..cols(whichvec_cut)]
		    dif = whichvec_cut[2..cols(whichvec_cut)] - whichvec_cut[1..(cols(whichvec_cut)-1)]
		}
		CI = (tlist[1,indexmat[1,1]],tlist[1,indexmat[1,2]])
		if (rows(indexmat)>1){
		    indexmat = indexmat[2..rows(indexmat),1...]
			indexmat = (indexmat \ (min(whichvec_cut),max(whichvec_cut)))
			CI = (tlist[1,indexmat[1,1]],tlist[1,indexmat[1,2]])
			for (j=2 ; j<=rows(indexmat) ; ++j){
				CI = (CI \ (tlist[1,indexmat[j,1]],tlist[1,indexmat[j,2]]))
			}
		}
	}
	
	st_matrix("CI",CI)
}

mata mosave rdlocrand_confint(), replace

end

********************************************************************************
** rdlocrand_confint_check: check confidence interval (rdsensitivity)
********************************************************************************

capture mata: mata drop rdlocrand_confint_check()
mata:
void rdlocrand_confint_check(real matrix wlist, real scalar cileft, real scalar ciright)
{
    	
	if (all(floatround(wlist[2,1...]):!=floatround(ciright))){
	    CI_position = -1
	}
	else if (all(floatround(wlist[1,1...]):!=floatround(cileft))){
	    CI_position = -2
	}
	else {
	    CI_position = min(selectindex(floatround(wlist[2,1...]):==floatround(ciright)))
	}
	st_numscalar("CI_position",CI_position)
}

mata mosave rdlocrand_confint_check(), replace

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
