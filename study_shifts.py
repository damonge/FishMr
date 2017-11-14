import numpy as np
import matplotlib.pyplot as plt
import fishmr as fst
import sostuff as stf
import healpy as hp
import emcee as mc
import time
import os
import sys
from scipy.optimize import minimize

#Unused yet
nside=256

if  len(sys.argv)!=9 :
    print "Usage: study_shifts.py a_lens with_sdr with_shifts sigma_shifts with_gains sigma_gains seed do_mcmc"
    exit(1)
alens=float(sys.argv[1])
if int(sys.argv[2])>0 : include_sdr=True
else : include_sdr=False
if int(sys.argv[3])>0 : perturb_shifts=True
else : perturb_shifts=False
shift_sigma=float(sys.argv[4])
if int(sys.argv[5])>0 : perturb_gains=True
else : perturb_gains=False
gains_sigma=float(sys.argv[6])
seed=int(sys.argv[7])
if int(sys.argv[8])>0 : do_mcmc=True
else : do_mcmc=False

#Output prefix
prefout="output/A%.1lf"%alens
if include_sdr :
    prefout+="_w"
else :
    prefout+="_wo"
prefout+="_sdr"
if perturb_shifts :
    prefout+="_bshift_l%.3lf"%(np.log10(shift_sigma))
if perturb_gains :
    prefout+="_gshift_l%.3lf"%(np.log10(gains_sigma))
if perturb_shifts or  perturb_gains :
    prefout+="_s%d"%seed
print prefout
np.random.seed(seed)
    
#Output files
fname_fisher=prefout+"_fisher"
fname_chain=prefout+"_chain"

#MCMC parameters
nwalkers=100
nsteps_burn=100
nsteps_per_chain=1000

#Get frequencies and noise levels for the current strawperson experiments
bands,noises=stf.sens_calc_SAT(1,2,0,1,0.05,n_years=5.)
rshifts=np.ones_like(bands)
if perturb_shifts :
    rshifts+=np.random.randn(len(bands))*shift_sigma
rgains=np.ones_like(bands)
if perturb_gains :
    rgains+=np.random.randn(len(bands))*gains_sigma
    
#Generate true experiment and sky
cmbm=fst.CMBModel()
xp_true=fst.ExperimentSimple(freqs=bands,noi_flat=noises,alpha_knee=-2.6,ell_knee=60.,diameter=5000.,fsky=0.05,name='SO_SAT')
sky_true=fst.SkyModel(contain_E=True,cmb_model=cmbm,A_lens=alens,contain_dust_sync_corr=include_sdr)

#Generate experiment and sky models
sky_model=fst.SkyModel(contain_E=True,cmb_model=cmbm,A_lens=alens,contain_dust_sync_corr=include_sdr)
xp_model=fst.ExperimentSimple(freqs=bands*rshifts,noi_flat=noises,alpha_knee=-2.6,ell_knee=60.,
                              diameter=5000.,fsky=0.05,name='SO_SAT_model',gains=rgains)

#Parameter ordering
pars=fst.construct_parameters(sky_model)
pars_vary=pars[pars['vary']]

#Indexing parameters
npar=len(pars_vary)
par0=pars_vary['val']

def like(sky,xpr,cl_data,inv_covar_data) :
    dx=cl_data-xpr.gst.unwrap_matrix(xpr.observeCl(larr,sky))
    
    return -0.5*np.sum(dx*np.sum(inv_covar_data[:,:,:]*dx[:,None,:],axis=2))

def like_emcee(p,args_dict) :
    dkeys=dict(zip(pars_vary['name'],p))
    sky=fst.SkyModel(contain_E=True,contain_sync=True,contain_dust=True,
                     contain_dust_sync_corr=include_sdr,contain_CO1=False,
                     contain_CO2=False,cmb_model=cmbm,**dkeys)
    
    if sky.is_consistent() :
        lik=like(sky,args_dict['xpr'],args_dict['cl_data'],args_dict['inv_covar_data'])
        if np.isnan(lik) :
            return -1E200
        else :
            return lik
    else :
        return -1E200

def chi2_emcee(p,args_dict) :
    return -like_emcee(p,args_dict)
    
#Set up data vector and covariance
larr=np.arange(30,300)
cl_d=xp_true.gst.unwrap_matrix(xp_true.observeCl(larr,sky_true))
lwei=1./((2*larr+1.0)*xp_model.fsky)
cl_cov=xp_true.gst.gaussian_covariance(xp_true.gst.wrap_vector(cl_d),lwei)
icl_cov=np.linalg.inv(cl_cov)
argdict={'xpr':xp_model,'cl_data':cl_d,'inv_covar_data':icl_cov}

#Estimate 1-sigma Fisher uncertainties
fsh=fst.get_fisher_cl(xp_true,sky_true);
argdtrue={'xpr':xp_true,'cl_data':cl_d,'inv_covar_data':icl_cov}
sigmas=np.sqrt(np.diag(np.linalg.inv(fst.get_fisher_from_like(like_emcee,pars_vary['val'],pars_vary['dval'],argdtrue)[0])))

print "Minimizing"
res=minimize(chi2_emcee,pars_vary['val'],args=(argdict),method='Powell')
p_min=res.x
#fm,fv=fst.get_fisher_from_like_precise(like_emcee,p_min,argdict)
fm,fv=fst.get_fisher_from_like(like_emcee,p_min,pars_vary['dval'],argdict)
np.savez(fname_fisher,p_true=pars_vary['val'],p_bf=p_min,fmat=fm,fvec=fv)
sigmas_fs=np.sqrt(np.diag(np.linalg.inv(fm)))
print "Fisher results"
for ip,p in enumerate(pars_vary) :
    name=p['name']
    print name+" = %.3lE (%.3lE) +- %.3lE (%.3lE)"%(p_min[ip],p['val'],sigmas_fs[ip],sigmas[ip])



if do_mcmc :
    print "MCMCing"
    start=time.time()
    for i in np.arange(100) :
        like_emcee(pars_vary['val'],argdict)
    stop=time.time()
    print stop-start

    if not os.path.isfile(fname_chain+".npy") :
        sampler=mc.EnsembleSampler(nwalkers,npar,like_emcee,args=[argdict])
        p0=pars_vary['val'][None,:]+sigmas[None,:]*0.05*np.random.randn(nwalkers,npar)
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
    means_mc=np.mean(samples,axis=0)
    sigmas_mc=np.std(samples,axis=0)
    print "MCMC results"
    for ip,p in enumerate(pars_vary) :
        name=p['name']
        print name+" = %.3lE (%.3lE) +- %.3lE (%.3lE)"%(means_mc[ip],p['val'],sigmas_mc[ip],sigmas[ip])
