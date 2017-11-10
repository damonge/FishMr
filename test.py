import numpy as np
import matplotlib.pyplot as plt
import fishmr as fst
import sostuff as stf
import healpy as hp
import emcee as mc
import time
import os

#Unused yet
nside=256

#Run parameters
include_sdr=False
alens=1.0

#Output prefix
prefout="A%.1lf"
if include_sdr :
    prefout+="_w"
else :
    prefout+="_wo"
prefout+="_sdr"

#Output files
fname_chain=prefout+"_chain"

#MCMC parameters
nwalkers=100
nsteps_burn=100
nsteps_per_chain=1000


#Get frequencies and noise levels for the current strawperson experiments
bands,noises=stf.sens_calc_SAT(1,2,0,1,0.05,n_years=5.)

#Generate true experiment and sky
cmbm=fst.CMBModel()
xp_true=fst.ExperimentSimple(freqs=bands,noi_flat=noises,alpha_knee=-2.6,ell_knee=60.,diameter=5000.,fsky=0.05,name='SO_SAT')
sky_true=fst.SkyModel(contain_E=True,cmb_model=cmbm,A_lens=alens,contain_dust_sync_corr=include_sdr)

#Generate experiment and sky models
sky_model=fst.SkyModel(contain_E=True,cmb_model=cmbm,A_lens=alens,contain_dust_sync_corr=include_sdr)
xp_model=fst.ExperimentSimple(freqs=bands,noi_flat=noises,alpha_knee=-2.6,ell_knee=60.,diameter=5000.,fsky=0.05,name='SO_SAT_model')

#Estimate 1-sigma Fisher uncertainties
fsh=fst.get_fisher_cl(xp_true,sky_true);

#Parameter ordering
pars=fst.construct_parameters(sky_model)
pars_vary=pars[pars['vary']]

#Index parameters
npar=len(pars_vary)
#ind_par=dict(zip(pars_vary['name'],np.arange(npar,dtype=int)))
par0=pars_vary['val']
dpar=pars_vary['dval']
sigmas=np.array([fsh.get_sigma(p['name']) for p in pars_vary])
    
def get_sky(par) :
#    if include_sdr :
#        r_sd=par[ind_par['r_dust_sync']]
#    else :
#        r_sd=0.
    dkeys=dict(zip(pars_vary['name'],par))
    
    sky=fst.SkyModel(contain_E=True,contain_sync=True,contain_dust=True,
                     contain_dust_sync_corr=include_sdr,contain_CO1=False,contain_CO2=False,cmb_model=cmbm,**dkeys)
#    sky=fst.SkyModel(contain_E=True,contain_sync=True,contain_dust=True,
#                     contain_dust_sync_corr=include_sdr,contain_CO1=False,contain_CO2=False,cmb_model=cmbm,
#                     r_prim=par[ind_par['r_prim']],A_lens=par[ind_par['A_lens']],
#                     A_sync_BB=par[ind_par['A_sync_BB']],A_sync_EE=par[ind_par['A_sync_EE']],
#                     A_dust_BB=par[ind_par['A_dust_BB']],A_dust_EE=par[ind_par['A_dust_EE']],
#                     alpha_sync=par[ind_par['alpha_sync']],beta_sync=par[ind_par['beta_sync']],
#                     alpha_dust=par[ind_par['alpha_dust']],beta_dust=par[ind_par['beta_dust']],
#                     temp_dust=par[ind_par['temp_dust']],r_dust_sync=r_sd)
    return sky

def like(sky,xpr,cl_data,inv_covar_data) :
    dx=cl_data-xpr.gst.unwrap_matrix(xp_true.observeCl(larr,sky))
    
    return -0.5*np.sum(dx*np.sum(inv_covar_data[:,:,:]*dx[:,None,:],axis=2))

def like_emcee(p,args_dict) :
    sky=get_sky(p)
    return like(sky,args_dict['xpr'],args_dict['cl_data'],args_dict['inv_covar_data'])


#Compute covariance matrix
larr=np.arange(30,300)
lwei=1./((2*larr+1.0)*xp_model.fsky)
cl_d=xp_true.gst.unwrap_matrix(xp_true.observeCl(larr,sky_true))
cl_cov=xp_true.gst.gaussian_covariance(xp_true.gst.wrap_vector(cl_d),lwei)
icl_cov=np.linalg.inv(cl_cov)
argdict={'xpr':xp_model,'cl_data':cl_d,'inv_covar_data':icl_cov}

print like_emcee(par0,argdict)
print like_emcee(par0+dpar,argdict)

f1m,f1v=fst.get_fisher_from_like(like_emcee,par0,dpar,argdict)
f0=fst.get_fisher_cl(xp_model,sky_model,xpr_true=xp_true,sky_true=sky_true)
f0m=f0.fishmat
f0v=f0.fisherr

print np.diag(f0m)/np.diag(f1m)-1
print f0v
print f1v
exit(1)

cov0=np.linalg.inv(f0m)
covv=np.linalg.inv(f1m)
print np.diag(f0m)/np.diag(f1m)-1
print np.sqrt(np.diag(cov0))
print np.sqrt(np.diag(covv))
        
start=time.time()
for i in np.arange(100) :
    like_emcee(par0,argdict)
stop=time.time()
print stop-start

if not os.path.isfile(fname_chain+".npy") :
    sampler=mc.EnsembleSampler(nwalkers,npar,like_emcee,args=[argdict])
    p0=par0[None,:]+sigmas[None,:]*0.05*np.random.randn(nwalkers,npar)
    #Burning phase
    print "Burning"
    pos,prob,stat=sampler.run_mcmc(p0,nsteps_burn)
    sampler.reset()
    #Running
    print "Running"
    sampler.run_mcmc(pos,nsteps_per_chain)
    print("Mean acceptance fraction: {0:.3f}"
          .format(np.mean(sampler.acceptance_fraction)))
    np.save(fname_chain,sampler.chain)

chain=np.load(fname_chain+".npy")
samples=(chain)[:,nsteps_per_chain/2:,:].reshape((-1,npar))
means=np.mean(samples,axis=0)
sigmas=np.std(samples,axis=0)
for p in pars :
    if p['vary'] :
        name=p['name']
        ip=ind_par[name]
        print name+" = %.3lE (%.3lE) +- %.3lE (%.3lE)"%(p['val'],means[ip]-p['val'],sigmas[ip],np.sqrt(np.diag(cov0))[ip])
exit(1)
for l,m,s in zip(labels,np.mean(samples,axis=0),np.std(samples,axis=0)) :
    print l+'= %lE +-'%m+' %lE'%s
