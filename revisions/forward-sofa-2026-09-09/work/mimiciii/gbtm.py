"""Finite mixture of quadratic censored-normal trajectories (0 <= SOFA <= 24).

This is a new Python implementation, not the unavailable original SAS fit.
The likelihood uses only observed values; missing trajectories are not filled.
"""
import numpy as np
from scipy.special import log_ndtr,logsumexp,ndtr
from scipy.optimize import minimize
from sklearn.cluster import KMeans

class CensoredGBTM:
    def __init__(self,k=3,degree=2,starts=5,max_iter=150,seed=20260908):
        self.k=k;self.degree=degree;self.starts=starts;self.max_iter=max_iter;self.seed=seed
    def _design(self,t):return np.vander(np.asarray(t,dtype=float)/60,self.degree+1,increasing=True)
    @staticmethod
    def _terms(y,mu,sigma):
        mask=np.isfinite(y);a=np.nan_to_num(y);z=(a-mu)/sigma
        logp=-.5*z*z-np.log(sigma)-.5*np.log(2*np.pi)
        gm=z/sigma;gs=z*z-1
        lower=mask&(a<=0);upper=mask&(a>=24)
        zl=(0-mu)/sigma;zu=(24-mu)/sigma
        ll=log_ndtr(zl);lu=log_ndtr(-zu)
        lam_l=np.exp(-.5*zl*zl-.5*np.log(2*np.pi)-ll)
        lam_u=np.exp(-.5*zu*zu-.5*np.log(2*np.pi)-lu)
        logp=np.where(lower,ll,np.where(upper,lu,logp))
        gm=np.where(lower,-lam_l/sigma,np.where(upper,lam_u/sigma,gm))
        gs=np.where(lower,-lam_l*zl,np.where(upper,lam_u*zu,gs))
        return np.where(mask,logp,0),np.where(mask,gm,0),np.where(mask,gs,0)
    def log_density(self,y,t):
        x=self._design(t)
        return np.column_stack([self._terms(y,x@self.beta[j],self.sigma[j])[0].sum(axis=1) for j in range(self.k)])
    def predict_proba(self,y,t):
        p=self.log_density(y,t)+np.log(self.pi)
        return np.exp(p-logsumexp(p,axis=1,keepdims=True))
    def fit(self,y,t):
        y=np.asarray(y,dtype=float);x=self._design(t);mask=np.isfinite(y)
        if (mask.sum(axis=1)<2).any():raise ValueError('Each fitted patient needs at least two observed time points.')
        n,j=y.shape
        mean=np.nanmean(y,axis=1);slope=np.array([np.polyfit(np.asarray(t)[m]/60,v[m],1)[0] for v,m in zip(y,mask)])
        init=np.column_stack([mean,np.clip(slope,-20,20)])
        scale=np.maximum(init.std(axis=0),1e-6);best=None;self.start_audit=[]
        for start in range(self.starts):
            labels=KMeans(n_clusters=self.k,n_init=1,random_state=self.seed+start*71).fit_predict(init/scale)
            resp=np.eye(self.k)[labels]*.98+.02/self.k
            beta=np.zeros((self.k,self.degree+1));sigma=np.full(self.k,2.)
            prev=-np.inf;success=True;history=[]
            for iteration in range(self.max_iter):
                pi=np.maximum(resp.mean(axis=0),1e-8);pi/=pi.sum()
                for g in range(self.k):
                    weights=resp[:,g,None];sums=(weights*mask).sum(axis=0)
                    if iteration==0:
                        yy=(weights*np.nan_to_num(y)).sum(axis=0)/np.maximum(sums,1e-12)
                        beta[g]=np.linalg.lstsq(x*np.sqrt(sums[:,None]),yy*np.sqrt(sums),rcond=None)[0]
                    def objective(theta):
                        lp,gm,gs=self._terms(y,x@theta[:-1],np.exp(theta[-1]))
                        value=-(lp*weights).sum()
                        grad=np.r_[-x.T@(gm*weights).sum(axis=0),-(gs*weights).sum()]
                        return value,grad
                    opt=minimize(objective,np.r_[beta[g],np.log(sigma[g])],method='L-BFGS-B',jac=True,
                      bounds=[(-80,80)]*(self.degree+1)+[(np.log(.35),np.log(20))],options={'maxiter':150,'ftol':1e-10,'gtol':1e-5})
                    beta[g]=opt.x[:-1];sigma[g]=np.exp(opt.x[-1]);success &= bool(opt.success)
                dens=np.column_stack([self._terms(y,x@beta[g],sigma[g])[0].sum(axis=1) for g in range(self.k)])+np.log(pi)
                norm=logsumexp(dens,axis=1);ll=norm.sum();resp=np.exp(dens-norm[:,None]);history.append(float(ll))
                if np.isfinite(prev) and abs(ll-prev)<1e-6*(1+abs(prev)):break
                prev=ll
            info={'start':start,'log_likelihood':float(ll),'iterations':iteration+1,'optimizer_success_all':success,'em_converged':iteration+1<self.max_iter,'monotone_up_to_tolerance':bool(np.all(np.diff(history)>=-1e-4))}
            self.start_audit.append(info)
            if best is None or ll>best[0]:best=(ll,beta.copy(),sigma.copy(),pi.copy(),info)
        self.loglik,self.beta,self.sigma,self.pi,self.best_start=best
        # Deterministic label order by fitted end-of-window expected score.
        means=self.mean_curve(t);order=np.argsort(means[:,-1]);self.beta=self.beta[order];self.sigma=self.sigma[order];self.pi=self.pi[order]
        self.n=n;self.n_parameters=self.k*(self.degree+2)+self.k-1;self.bic=-2*self.loglik+self.n_parameters*np.log(n)
        return self
    def mean_curve(self,t):
        mu=self.beta@self._design(t).T;s=self.sigma[:,None];a=-mu/s;b=(24-mu)/s
        phi=lambda z:np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        return mu*(ndtr(b)-ndtr(a))+s*(phi(a)-phi(b))+24*(1-ndtr(b))
    def to_dict(self):
        return {'k':self.k,'degree':self.degree,'beta':self.beta.tolist(),'sigma':self.sigma.tolist(),'pi':self.pi.tolist(),'loglik':self.loglik,'bic':self.bic,'n':self.n,'n_parameters':self.n_parameters,'best_start':self.best_start,'starts':self.start_audit}
