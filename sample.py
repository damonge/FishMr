import numpy as np
import fishmr as fst
import sostuff as stf

#Get frequencies and noise levels for the current strawperson experiments
bands,noises=stf.sens_calc_SAT(1,2,0,1,0.05,n_years=5.)

#Generate an Experiment object based on these bands and noises.
#Pass also information about non-white noise, resolution, f_sky etc.
#Note that you can also pass an array of relative gains (set to 1 by default) - see docstring
xp=fst.ExperimentSimple(freqs=bands,noi_flat=noises,alpha_knee=-2.6,ell_knee=60.,diameter=0.5,
                        fsky=0.05,name='SO_SAT')

#Generate a sky model. See docstring for details on 
sky=fst.SkyModel(r_prim=0.000,contain_dust_sync_corr=True,r_dust_sync=0.10)

#Compute Fisher matrix.
#Note that you can also use this function to compute Fisher bias associated to
#differences between the true sky or experiment models and the assumed ones.
#See docstrings for more details.
fisher=fst.get_fisher_cl(xp,sky)

#Use Fisher structure to compute error on r
print "r = %.3lE +- %.3lE (stat) +- %.3lE (syst)"%(fisher.get_value('r_prim'),
                                                   fisher.get_sigma('r_prim'),
                                                   fisher.get_bias('r_prim'))
