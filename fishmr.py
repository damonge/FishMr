import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import sys
import os

AMIN2RAD=np.pi/180/60.

def cl_plaw(l,lref,A,a) :
    """
    Power law power spectrum, such that
       D_ell = A*(l/lref)^a
    """
    cl_pl=A*(l/(lref+0.0))**a*2*np.pi/(l*(l+1.))
    return cl_pl

def freq_evolve(spec_type,nu_0,beta,temp,fco,nu) :
    """
    Frequency evolution in units of K_CMB
    spec_type: "BB", "PL", "mBB", "CO1", "CO2"
    """
    x=0.017611907*nu
    ex=np.exp(x)
    fcmb=ex*(x/(ex-1))**2
    if spec_type=="BB" : #CMB
        return 1.
    elif spec_type=="PL" : #Synch
        return (nu/nu_0)**beta/fcmb
    elif spec_type=="mBB" : #Dust
        x_to=0.0479924466*nu/temp
        x_from=0.0479924466*nu_0/temp
        return (nu/nu_0)**(1+beta)*(np.exp(x_from)-1)/(np.exp(x_to)-1)/fcmb
    elif spec_type=="CO1" : #CO_1
        fout=np.zeros_like(nu)
        fout[np.where((nu<130.) & (nu>85.))[0]]=1.
        return fout
    elif spec_type=="CO2" : #CO_2
        fout=np.zeros_like(nu)
        fout[np.where((nu<260.) & (nu>190.))[0]]=fco
        return fout

#Reads B-mode power spectrum from CAMB format and returns as C_l in an array starting at l=0
def read_cl_bb(fname,lmax=3000) :
    """
    Read BB power spectrum in CAMB format into l,C_l
    """
    data=np.loadtxt(fname,unpack=True)
    larr=data[0]
    dlbb=data[3]
    clbb=np.zeros(len(larr)+2)
    clbb[2:]=dlbb*2*np.pi/(larr*(larr+1))
    
    return np.arange(lmax),clbb[:lmax]

#Reads E-mode power spectrum from CAMB format and returns as C_l in an array starting at l=0
def read_cl_ee(fname,lmax=3000) :
    """
    Read EE power spectrum in CAMB format into l,C_l
    """
    data=np.loadtxt(fname,unpack=True)
    larr=data[0]
    dlbb=data[2]
    clbb=np.zeros(len(larr)+2)
    clbb[2:]=dlbb*2*np.pi/(larr*(larr+1))
    
    return np.arange(lmax),clbb[:lmax]


class SkyModel(object) :
    """
    Sky model class
    This class describes all the different components contributing to the sky emission.
    Currently supported:
      - CMB
      - Synchrotron
      - Thermal dust
      - Correlation between synchrotron and dust
      - CO1 and CO2
    """
    def __init__(self,
                 contain_cmb=True,contain_E=False,r_prim=0.,A_lens=1.,
                 contain_sync=True,A_sync=3.80,alpha_sync=-0.60,beta_sync=-3.0,nu0_sync=23.,xi_sync=50.,
                 contain_dust=True,A_dust=4.25,alpha_dust=-0.42,beta_dust=1.59,temp_dust=20.,nu0_dust=353.,xi_dust=50.,
                 contain_dust_sync_corr=False,r_dust_sync=0.0,
                 contain_CO1=False,A_CO1=0.5,alpha_CO1=-0.42,
                 contain_CO2=False,A_CO2=0.5,alpha_CO2=-0.42) :
        """
        Initializes SkyModel structure
        contain_cmb: include cmb?
        contain_E: include both E and B?
        contain_sync: include synchrotron?
        A_sync, alpha_sync, beta_sync, nu0_sync, xi_sync:
                D^sync_ell(nu1,nu2) = A_sync * (ell/80.)**alpha_sync *
                                      f_sync(nu1,nu0_sync,beta_sync) * f_sync(nu2,nu0_sync,beta_sync) *
                                      exp[-0.5*(log(nu1/nu2)/xi_sync)**2]
       contain_dust: include dust?
        A_dust, alpha_dust, beta_dust, temp_dust, nu0_dust, xi_dust:
                D^dust_ell(nu1,nu2) = A_dust * (ell/80.)**alpha_dust *
                                      f_dust(nu1,nu0_dust,beta_dust,temp_dust) * f_dust(nu2,nu0_dust,beta_dust,temp_dust) *
                                      exp[-0.5*(log(nu1/nu2)/xi_dust)**2]
        contain_dust_sync_corr: include sychrotron-dust correlation?
        r_dust_sync: synchrotron-dust correlation coefficient
        contain_CO1: include CO 1->0 at 115 GHz
        A_CO1, alpha_CO1: see similar parameters for dust
        contain_CO2: include CO 2->1 at 230 GHz
        A_CO2, alpha_CO2: see similar parameters for dust
        """

        self.contain_cmb=contain_cmb
        self.contain_E=contain_E
        self.contain_sync=contain_sync
        self.contain_dust=contain_dust
        self.contain_CO1=contain_CO1
        self.contain_CO2=contain_CO2

        ncomp=0
        if self.contain_cmb :
            ncomp+=1
            self.r_prim=r_prim
            self.A_lens=A_lens
            lp,cl_bb_prim=read_cl_bb("data/planck1_r1p00_tensCls.dat")
            lp,cl_ee_prim=read_cl_ee("data/planck1_r1p00_tensCls.dat")
            ll,cl_bb_lens=read_cl_bb("data/planck1_r0p00_lensedtotCls.dat")
            ll,cl_ee_lens=read_cl_ee("data/planck1_r0p00_lensedtotCls.dat")
            self.cl_bb_primf=interp1d(lp,cl_bb_prim)
            self.cl_bb_lensf=interp1d(ll,cl_bb_lens)
            self.cl_ee_primf=interp1d(lp,cl_ee_prim)
            self.cl_ee_lensf=interp1d(ll,cl_ee_lens)
            
        if self.contain_sync :
            ncomp+=1
            self.A_sync=A_sync
            self.alpha_sync=alpha_sync
            self.beta_sync=beta_sync
            self.nu0_sync=nu0_sync
            self.xi_sync=xi_sync
            
        if self.contain_dust :
            ncomp+=1
            self.A_dust=A_dust
            self.alpha_dust=alpha_dust
            self.beta_dust=beta_dust
            self.temp_dust=temp_dust
            self.nu0_dust=nu0_dust
            self.xi_dust=xi_dust

        if self.contain_sync and self.contain_dust :
            if contain_dust_sync_corr :
                self.contain_dust_sync_corr=contain_dust_sync_corr
                self.r_dust_sync=r_dust_sync
            else :
                self.contain_dust_sync_corr=False
            
        if self.contain_CO1 :
            ncomp+=1
            self.A_CO1=A_CO1
            self.alpha_CO1=alpha_CO1
            
        if self.contain_CO2 :
            ncomp+=1
            self.A_CO2=A_CO2
            self.alpha_CO2=alpha_CO2

        self.ncomp=ncomp

    def update_param(self,name,value) :
        """
        Update one of the parameters of this class.
        'name' must correspond to one of the parameters passed when initializing the structure
        """
        found=False

        if self.contain_cmb :
            if name=='r_prim' :
                self.r_prim=value
                found=True
            if name=='A_lens' :
                self.A_lens=value
                found=True
        if self.contain_sync :
            if name=='A_sync' :
                self.A_sync=value
                found=True
            if name=='alpha_sync' :
                self.alpha_sync=value
                found=True
            if name=='beta_sync' :
                self.beta_sync=value
                found=True
            if name=='nu0_sync' :
                self.nu0_sync=value
                found=True
            if name=='xi_sync' :
                self.xi_sync=value
                found=True

        if self.contain_dust :
            if name=='A_dust' :
                self.A_dust=value
                found=True
            if name=='alpha_dust' :
                self.alpha_dust=value
                found=True
            if name=='beta_dust' :
                self.beta_dust=value
                found=True
            if name=='temp_dust' :
                self.temp_dust=value
                found=True
            if name=='nu0_dust' :
                self.nu0_dust=value
                found=True
            if name=='xi_dust' :
                self.xi_dust=value
                found=True
                
        if self.contain_dust_sync_corr :
            if name=='r_dust_sync' :
                self.r_dust_sync=value
                found=True
            
        if self.contain_CO1 :
            if name=='A_CO1' :
                self.A_CO1=value
                found=True
            if name=='alpha_CO1' :
                self.alpha_CO1=value
                found=True
            
        if self.contain_CO2 :
            if name=='A_CO2' :
                self.A_CO2=value
                found=True
            if name=='alpha_CO2' :
                self.alpha_CO2=value
                found=True

        if not found :
            ValueError("Parameter "+name+" not found")
        
    def cl_cmb(self,l) :
        clp=(self.cl_bb_primf)(l)
        cll=(self.cl_bb_lensf)(l)
        return self.r_prim*clp+self.A_lens*cll

    def cl_cmb_ee(self,l) :
        clp=(self.cl_ee_primf)(l)
        cll=(self.cl_ee_lensf)(l)
        return self.r_prim*clp+cll
        
    def get_Cl(self,freqs,ells) :
        """
        Returns all cross-frequency power spectra according to the
        sky model encapsulated in this object.
        Power spectra are computed at all pairs of frequencies 
        derived from the array 'freqs' and on the multipoles 'ells'.
        """
        if np.asarray(freqs).ndim==0 :
            nus=np.array([freqs])
        else :
            nus=freqs.copy()
                
        if np.asarray(ells).ndim==0 :
            ls=np.array([ells])
        else :
            ls=ells.copy()
                
        nnu=len(nus)
        nell=len(ls)
        cl_comp=np.zeros([nell,self.ncomp])
        f_matrix=np.zeros([nnu,self.ncomp])
        decorr_nu=np.ones([nnu,nnu,self.ncomp])
        icomp=0
        if self.contain_cmb :
            cl_comp[:,icomp]=self.cl_cmb(ls)
            f_matrix[:,icomp]=freq_evolve("BB",None,None,None,None,nus)
            icomp_cmb=icomp
            icomp+=1
        
        if self.contain_sync :
            cl_comp[:,icomp]=cl_plaw(ls,80.,self.A_sync,self.alpha_sync)
            f_matrix[:,icomp]=freq_evolve("PL",self.nu0_sync,self.beta_sync,None,None,nus)
            decorr_nu[:,:,icomp]=np.exp(-0.5*(np.log(nus[:,None]/nus[None,:])/self.xi_sync)**2)
            icomp_sync=icomp
            icomp+=1

        if self.contain_dust :
            cl_comp[:,icomp]=cl_plaw(ls,80.,self.A_dust,self.alpha_dust)
            f_matrix[:,icomp]=freq_evolve("mBB",self.nu0_dust,self.beta_dust,self.temp_dust,None,nus)
            decorr_nu[:,:,icomp]=np.exp(-0.5*(np.log(nus[:,None]/nus[None,:])/self.xi_dust)**2)
            icomp_dust=icomp
            icomp+=1

        if self.contain_CO1 :
            cl_comp[:,icomp]=cl_plaw(ls,80.,self.A_CO1,self.alpha_CO1)
            f_matrix[:,icomp]=freq_evolve("CO1",None,None,None,None,nus)
            icomp+=1

        if self.contain_CO2 :
            cl_comp[:,icomp]=cl_plaw(ls,80.,self.A_CO2,self.alpha_CO2)
            f_matrix[:,icomp]=freq_evolve("CO2",None,None,None,0.5,nus)
            icomp+=1

        rcorr=np.identity(self.ncomp)
        if self.contain_dust_sync_corr :
            rcorr[icomp_sync,icomp_dust]=rcorr[icomp_dust,icomp_sync]=self.r_dust_sync

        a_nu=decorr_nu[:,:,:]*f_matrix[:,None,:]*f_matrix[None,:]
        csqr_matrix_comp=np.sqrt(cl_comp[:,None,None,:]*a_nu[None,:,:,:])
        c_matrix_comp=np.sum(np.dot(csqr_matrix_comp,rcorr)*csqr_matrix_comp,axis=3)

        if self.contain_E :
            csqr_matrix_comp_ee=csqr_matrix_comp.copy()
            csqr_matrix_comp_ee[:,:,:,icomp_cmb]=np.sqrt((self.cl_cmb_ee(ls))[:,None,None]*a_nu[None,:,:,icomp_cmb])
            c_matrix_comp=np.sum(np.dot(csqr_matrix_comp_ee,rcorr)*csqr_matrix_comp_ee,axis=3)
            c_out=np.zeros([nell,2*nnu,2*nnu])
            c_out[:,:nnu,:nnu]=c_matrix_comp
            c_out[:,nnu:,nnu:]=c_matrix_comp_ee
        else :
            c_out=c_matrix_comp.copy()

        return c_out

    
class ExperimentBase(object) :
    """
    Base Experiment class
    """

    def __init__(self) :
        self.typestr="base"
        self.name="dum"
        self.fsky=1.0
        self.freqs=np.array([150.])
        self.noi_flat=np.array([0.])
        self.diam=5.
        self.gains=np.array([1.])

    def getFsky(self) :
        return self.fsky

    def getFrequencies(self) :
        return self.freqs

    def getDiameter(self) :
        return self.diam

    def getNoiseFlat(self) :
        return self.noi_flat

    def getNoises(self,ells) :
        ells=np.asarray(ells)
        scalar_input=False
        if ells.ndim == 0 :
            ells=ells[None]
            scalar_input=True
        
        noises=np.zeros([ells,len(self.freqs),len(self.freqs)])

        if scalar_input :
            noises=noises[0,:,:]

        return noises

    def observeCl(self,ells,sky) :
        """
        Produce cross-band powers as observed by this experiment.
        This implies generating the true C_ells defined by a SkyModel
        object (sky) and perturbing them with this experiment's gains and noise.
        """
        noi=self.getNoises(ells)
        ctrue=sky.get_Cl(self.freqs,ells)
        gainmat=self.gains[:,None]*self.gains[None,:]
        return ctrue*gainmat[None,:,:]+noi

class ExperimentSimple(ExperimentBase) :
    """
    Simple experiment with a bunch of frequencies, noise levels,
    ell_knees and alpha_knees and a constant aperture
    """

    def __init__(self,fsky=None,name=None,freqs=None,gains=None,
                 noi_flat=None,alpha_knee=None,ell_knee=None,diameter=None) :
        """
        fsky: sky fraction
        name: experiment's name
        freqs: frequencies in GHz
        gains: relative gains at each frequency
        noi_flat: flat noise levels at each frequency (in uK_CMB arcmin)
        alpha_knee: knee exponent for non-flat noise
        ell_knee: knee multipole for non-flat noise
        diameter: telescope diameter in meters
        """
        self.typestr="Simple"
        if name is None : self.name="default"
        else : self.name=name
        if fsky is None : self.fsky=1.
        else : self.fsky=fsky
        if diameter is None : self.diam=5.
        else : self.diam=diameter
        if freqs is None : self.freqs=np.array([150.])
        else :
            freqs=np.asarray(freqs)
            if freqs.ndim==0 :
                freqs=freqs[None]
            self.freqs=freqs.copy()
        if gains is None : self.gains=np.ones(len(self.freqs))*1.
        else :
            if np.asarray(gains).ndim==0 :
                self.gains=gains*np.ones(len(freqs))
            else :
                if len(gains)!=len(self.freqs) :
                    raise ValueError("gains and freqs should have the same number of elements")
                self.gains=gains.copy()
        if noi_flat is None : self.noi_flat=np.ones(len(self.freqs))*1E20
        else :
            if np.asarray(noi_flat).ndim==0 :
                self.noi_flat=noi_flat*np.ones(len(freqs))
            else :
                if len(noi_flat)!=len(self.freqs) :
                    raise ValueError("noi_flat and freqs should have the same number of elements")
                self.noi_flat=noi_flat.copy()
        if alpha_knee is None : self.alpha_knee=-1.9*np.ones(len(self.freqs))
        else :
            if np.asarray(alpha_knee).ndim==0 :
                self.alpha_knee=alpha_knee*np.ones(len(freqs))
            else :
                if len(alpha_knee)!=len(self.freqs) :
                    raise ValueError("alpha_knee and freqs should have the same number of elements")
                self.alpha_knee=alpha_knee.copy()
        if ell_knee is None : self.ell_knee=np.ones(len(self.freqs))*0.01
        else :
            if np.asarray(ell_knee).ndim==0 :
                self.ell_knee=ell_knee*np.ones(len(freqs))
            else :
                if len(ell_knee)!=len(self.freqs) :
                    raise ValueError("ell_knee and freqs should have the same number of elements")
                self.ell_knee=ell_knee.copy()
        self.beam_sigmas=AMIN2RAD*1315.0/(2.355*self.freqs*self.diam)
                
    def getNoises(self,ells) :
        """
        Returns multi-frequency noise power spectrum (as an N_freq x N_freq matrix)
        sampled at the multipoles 'ells'
        """
        nlevels=self.noi_flat*AMIN2RAD
        ells=np.asarray(ells)
        scalar_input=False
        if ells.ndim == 0 :
            ells=ells[None]
            scalar_input=True

        nell=(np.ones(len(ells))[:,None]+(ells[:,None]/self.ell_knee[None,:])**self.alpha_knee[None,:])
        nell*=np.exp((ells*(ells+1.))[:,None]*((self.beam_sigmas[None,:])**2))
        nell*=(nlevels**2)[None,:]

        noises=np.zeros([len(ells),len(self.freqs),len(self.freqs)])
        for i,f in enumerate(self.freqs) :
            noises[:,i,i]=nell[:,i]

        if scalar_input :
            noises=noises[0,:,:]

        return noises

    
class ExperimentDouble(ExperimentBase) :
    """
    Experiment made up of two separate telescopes.
    Both telescopes should have the same frequency channels,
    but not necessarily the same noise properties or diameters.
    """

    def __init__(self,name=None,freqs=None,gains=None,fsky=None,exp1=None,exp2=None) :
        """
        See definition for ExperimentSimple
        This experiment will combine two ExperimentSimple objects in an
        inverse-variance sense. Both experiments are assumed to observe
        on the same frequencies, but may have different resolutions
        or beam properties.
        """
        if len(exp1.freqs)!=len(exp2.freqs) :
            raise ValueError("Incompatible experiments")
        if len(exp1.freqs)!=len(freqs) :
            raise ValueError("Incompatible experiments")
        self.exp1=ExperimentSimple(name=name,freqs=freqs,fsky=fsky,
                                   gains=gains,noi_flat=exp1.noi_flat,
                                   alpha_knee=exp1.alpha_knee,ell_knee=exp1.ell_knee,diameter=exp1.diam)
        self.exp2=ExperimentSimple(name=name,freqs=freqs,fsky=fsky,
                                   gains=gains,noi_flat=exp2.noi_flat,
                                   alpha_knee=exp2.alpha_knee,ell_knee=exp2.ell_knee,diameter=exp2.diam)
        self.typestr="Double"
        self.name=name
        self.fsky=fsky
        self.freqs=freqs.copy()
        self.gains=self.exp1.gains.copy()
        self.noi_flat=1./np.sqrt(1./self.exp1.noi_flat**2+1./self.exp2.noi_flat**2)

    def getNoises(self,ells) :
        noi1=self.exp1.getNoises(ells)
        noi2=self.exp2.getNoises(ells)
        noises=1./(1./noi1+1./noi2)

        return noises

    
class FisherParams(object) :
    """
    Fisher data class
    This class is a collection of variables relevant to make Fisher forecasts 
    (both for expected uncertainties and biases).
    Right now only get_fisher_cl should be used to instantiate this class.
    """
    def __init__(self,fisher_matrix,params_dict,l_arr,cl_fid,cl_deriv,fisher_error=None) :
        """
        fisher_matrix : Npar x Npar Fisher matrix
        params_dict : dictionary of parameters. Npar of them should be variable.
        l_arr : multipoles used to compute this Fisher data
        cl_fid : fiducial power spectra used when computing this Fisher data
        cl_deriv : power spectrum derivatives used to compute this Fisher data
        fisher_error : Npar Fisher error vector
        """
        self.ells=l_arr
        self.cl_fid=cl_fid.copy()
        self.dcl=cl_deriv.copy()
        self.fishmat=fisher_matrix.copy()
        self.covar=np.linalg.inv(fisher_matrix)
        self.p_ind={}
        self.p_val={}
        if fisher_error is None :
            self.fisherr=np.zeros(len(self.fishmat))
        else :
            if len(fisher_error)!=len(self.fishmat) :
                raise ValueError('Incompatible fisher error and matrix')
            self.fisherr=fisher_error.copy()
        self.bias=np.dot(self.covar,self.fisherr)
        
        indices_vary=np.where(params_dict['vary']==True)[0]
        nvary=len(indices_vary)
        if nvary!=len(self.fishmat) or nvary!=len(self.fisherr) :
            raise ValueError('Incompatible params and fisher matrix')
        
        for i1 in np.arange(nvary) :
            ip=indices_vary[i1]
            self.p_ind[params_dict['name'][ip]]=i1
            self.p_val[params_dict['name'][ip]] =params_dict['val'][ip]

    def get_param_names(self) :
        """
        Returns the names of all params included in this Fisher data
        """
        return self.p_ind.keys()
    def get_value(self,param) :
        """
        Get fiducial value for a parameter (or list of) 'param'
        """
        return_scalar=False
        if np.asarray(param).ndim==0 :
            pnames=np.array([param])
            return_scalar=True
        else :
            pnames=param

        values=np.zeros(len(pnames))
        for ip,p in enumerate(pnames) :
            values[ip]=self.p_val[p]

        if return_scalar :
            return values[0]
        else :
            return values
    
    def get_sigma(self,param) :
        """
        Get standard deviation for a parameter (or list of) 'param'
        """
        return_scalar=False
        if np.asarray(param).ndim==0 :
            pnames=np.array([param])
            return_scalar=True
        else :
            pnames=param

        errors=np.zeros(len(pnames))
        for ip,p in enumerate(pnames) :
            errors[ip]=np.sqrt(self.covar[self.p_ind[p],self.p_ind[p]])

        if return_scalar :
            return errors[0]
        else :
            return errors
    
    def get_bias(self,param) :
        """
        Get bias for a parameter (or list of) 'param'
        """
        return_scalar=False
        if np.asarray(param).ndim==0 :
            pnames=np.array([param])
            return_scalar=True
        else :
            pnames=param

        errors=np.zeros(len(pnames))
        for ip,p in enumerate(pnames) :
            errors[ip]=self.bias[self.p_ind[p]]

        if return_scalar :
            return errors[0]
        else :
            return errors
    
    
#Gets delensing factor as a function of noise level
def delens_factor(noise) :
    """
    Gets delensing factor as a function of reduced noise level.
    Basically an eyeball fit to Fig. 4 of 1509.06770
    """
    if noise<30000 :
        x=noise/5.
        return x**1.1/(1+x**1.1)
    else :
        return 0.4+0.005*noise

def get_ndet(net,t_yr,efficiency,f_sky,sigma) :
    """
    Gets number of detectors needed to achieve a given noise level (sigma - polarized, uK_CMB-arcmin)
    given NETs, observation time, efficiency and sky fraction
    """
    t_sec=t_yr*365*24*3600.
    return (net/sigma/AMIN2RAD)**2*8*np.pi*f_sky/efficiency/t_sec

def net_2_sigr(net,t_yr,efficiency,f_sky,n_det) :
    """
    Gets polarized noise level in uK_CMB-arcmin for given NETs, observation time, efficiency, sky
    fraction and number of detectors
    """
    t_sec=t_yr*365*24*3600.
    sigma_vec=net*np.sqrt(2.) #Transform to Q/U
    sigma_vec*=np.sqrt(4*np.pi*f_sky/(efficiency*t_sec*n_det))/AMIN2RAD #Transform to uK_amin
    return sigma_vec

def get_noiselevel(exper,add_CO_1=False,add_CO_2=False) :
    """
    Get noise level after component separation (for constant spectral indices)
    exper should be an Experiment object.
    """
    ncomp=3
    if add_CO_1 : ncomp+=1
    if add_CO_2 : ncomp+=1
    freqs=exper.getFrequencies()
    noise_levels=exper.getNoiseFlat()
    n_nu=len(freqs)
    f=np.zeros([ncomp,n_nu])
    f[0,:]=freq_evolve("BB" ,None,None,None,None,freqs)
    f[1,:]=freq_evolve("PL" ,23. ,-3. ,None,None,freqs)
    f[2,:]=freq_evolve("mBB",353.,1.54,19. ,None,freqs)
    if add_CO_1 :
        f[3,:]=freq_evolve("CO1",None   ,None ,None ,None,freqs)
    if add_CO_2 :
        f[4,:]=freq_evolve("CO2",None   ,None ,None ,0.5,freqs)
    noisemat=np.diag(noise_levels**2)
    noise_comps=np.linalg.inv(np.dot(f,np.dot(np.linalg.inv(noisemat),np.transpose(f))))

    return np.sqrt(noise_comps[0,0])
            
def get_fisher_cl(xpr,sky,lmin=30,lmax=300,xpr_true=None,sky_true=None) :
    """
    Returns a FisherParams object computed from the Experiment xpr
    assuming the sky model given by the SkyModel sky.
    Only multipoles between lmin and lmax will be used.
    Pass a true experiment (xpr_true) or a true sky (sky_true), different
    from the xpr or sky, if you want to estimate the associated bias on
    any parameter.
    """
    lmin_use=min(max(lmin,np.pi/np.sqrt(xpr.getFsky())),lmax-1)

    if xpr_true is None :
        xprt=xpr
    else :
        xprt=xpr_true
    if sky_true is None :
        skyt=sky
    else :
        skyt=sky_true
    
    pars_arr=[]
    if sky.contain_cmb :
        if sky.r_prim>0.001 :
            drprim=0.2*sky.r_prim
            onesided=False
        else :
            drprim=5E-4
            onesided=True
        pars_arr.append({'val':sky.r_prim ,'dval':drprim,'name':'r_prim'  ,
                         'label':'$r$'                  ,'vary':True})
        pars_arr.append({'val':sky.A_lens,'dval':0.01 ,'name':'A_lens'  ,
                         'label':'$A_{\\rm lens}$'      ,'vary':True})
    if sky.contain_sync :
        pars_arr.append({'val':sky.A_sync ,'dval':0.1  ,'name':'A_sync'  ,
                         'label':'$A_{\\rm sync}$'      ,'vary':True})
        pars_arr.append({'val':sky.alpha_sync,'dval':0.01 ,'name':'alpha_sync'  ,
                         'label':'$\\alpha_{\\rm sync}$','vary':True})
        pars_arr.append({'val':sky.beta_sync,'dval':0.005,'name':'beta_sync'  ,
                         'label':'$\\beta_{\\rm sync}$' ,'vary':True})
        pars_arr.append({'val':sky.nu0_sync,'dval':-1   ,'name':'nu0_sync',
                         'label':'$\\nu_{\\rm sync}$'   ,'vary':False})
        pars_arr.append({'val':sky.xi_sync  ,'dval':0.2  ,'name':'xi_sync'   ,
                         'label':'$\\xi_{\\rm sync}$'   ,'vary':False})
    if sky.contain_dust :
        pars_arr.append({'val':sky.A_dust,'dval':0.1  ,'name':'A_dust'  ,
                         'label':'$A_{\\rm dust}$'      ,'vary':True})
        pars_arr.append({'val':sky.alpha_dust,'dval':0.01 ,'name':'alpha_dust'  ,
                         'label':'$\\alpha_{\\rm dust}$','vary':True})
        pars_arr.append({'val':sky.beta_dust,'dval':0.005,'name':'beta_dust'  ,
                         'label':'$\\beta_{\\rm dust}$' ,'vary':True})
        pars_arr.append({'val':sky.temp_dust,'dval':0.5  ,'name':'temp_dust'  ,
                         'label':'$T_{\\rm dust}$'      ,'vary':True})
        pars_arr.append({'val':sky.nu0_dust,'dval':-1   ,'name':'nu0_dust',
                         'label':'$\\nu_{\\rm dust}$'   ,'vary':False})
        pars_arr.append({'val':sky.xi_dust ,'dval':0.2  ,'name':'xi_dust'   ,
                         'label':'$\\xi_{\\rm dust}$'   ,'vary':False})
    if sky.contain_dust_sync_corr :
        pars_arr.append({'val':sky.r_dust_sync,'dval':0.1  ,'name':'r_dust_sync'  ,
                         'label':'$r_{\\rm d-s}$'       ,'vary':True})
    if sky.contain_CO1 :
        pars_arr.append({'val':sky.A_CO1,'dval':0.05 ,'name':'A_CO1'   ,
                         'label':'$A^1_{\\rm CO}$'      ,'vary':True})
        pars_arr.append({'val':sky.alpha_CO1,'dval':0.1  ,'name':'alpha_CO1'   ,
                         'label':'$\\alpha^1_{\\rm CO}$','vary':True})
    if sky.contain_CO2 :
        pars_arr.append({'val':sky.A_CO2,'dval':0.05 ,'name':'A_CO2'   ,
                         'label':'$A^2_{\\rm CO}$'      ,'vary':True})
        pars_arr.append({'val':sky.alpha_CO2,'dval':0.1  ,'name':'alpha_CO2'   ,
                         'label':'$\\alpha^2_{\\rm CO}$','vary':True})
    pars_arr.append({'val':lmax ,'dval':-1   ,'name':'lmax'   ,
                     'label':'$\\ell_{\\rm max}$'   ,'vary':False})
    pars_arr.append({'val':lmin_use ,'dval':-1   ,'name':'lmin'   ,
                     'label':'$\\ell_{\\rm min}$'   ,'vary':False})
    pars_all=pd.DataFrame(pars_arr).to_records()

    fsky=xpr.getFsky()
    nu_arr=xpr.getFrequencies()
    nnu=len(nu_arr)
    l_arr=np.arange(pars_all[pars_all['name']=='lmin']['val'][0],
                    pars_all[pars_all['name']=='lmax']['val'][0])
    nl=len(l_arr)
    indices_vary=np.where(pars_all['vary']==True)[0]
    nvary=len(indices_vary)
    cl_fid=xpr.observeCl(l_arr,sky)
    cl_obs=xprt.observeCl(l_arr,skyt)
    icl_fid=np.linalg.inv(cl_fid)

    dcl_arr=np.zeros([nvary,len(cl_fid),len(cl_fid[0]),len(cl_fid[0])])
    for i in np.arange(nvary) :
        ip=indices_vary[i]
        name=pars_all['name'][ip]
        val=pars_all['val'][ip]
        dval=pars_all['dval'][ip]
        val_p=val+dval
        val_m=val-dval
        sky.update_param(name,val_p)
        clp=xpr.observeCl(l_arr,sky)
        sky.update_param(name,val_m)
        clm=xpr.observeCl(l_arr,sky)
        sky.update_param(name,val)
        dcl_arr[i,:,:,:]=(clp-clm)/(2*dval)

    fisher_ell=np.zeros([nl,nvary,nvary])
    fisherr_ell=np.zeros([nl,nvary])
    ecl=cl_obs-cl_fid
    for il in np.arange(nl) :
        for i1 in np.arange(nvary) :
            f_v=np.trace(np.dot(dcl_arr[i1,il,:,:],
                              np.dot(icl_fid[il,:,:],
                                     np.dot(ecl[il,:,:],icl_fid[il,:,:]))))
            fisherr_ell[il,i1]=fsky*(l_arr[il]+0.5)*f_v
            for i2 in np.arange(nvary-i1)+i1 :
                f_m=np.trace(np.dot(dcl_arr[i2,il,:,:],
                                    np.dot(icl_fid[il,:,:],
                                           np.dot(dcl_arr[i1,il,:,:],icl_fid[il,:,:]))))
                fisher_ell[il,i1,i2]=fsky*(l_arr[il]+0.5)*f_m
                if i1!=i2 :
                    fisher_ell[:,i2,i1]=fisher_ell[:,i1,i2]
    fisher=np.sum(fisher_ell,axis=0)
    fisher_error=np.sum(fisherr_ell,axis=0)
        
    fishp=FisherParams(fisher,pars_all,l_arr,cl_fid,dcl_arr,fisher_error=fisher_error)

    return fishp
