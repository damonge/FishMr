import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import interp1d
import sys
import os
#import healpy as hp
import numdifftools  as nd

AMIN2RAD=np.pi/180/60.

def cl_plaw(l,lref,A,a) :
    """
    Power law power spectrum, such that
       D_ell = A*(l/lref)^a
    """
    ind=np.where(l<10)[0]
    cl_pl=A*((l+0.01)/(lref+0.01))**a*2*np.pi/((l+0.01)*(l+1.))
    if len(ind)>0 :
        cl_pl[ind]=cl_pl[ind[-1]]
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
        return (nu/nu_0)**(beta+temp*np.log(nu/nu_0))/fcmb
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
def read_cl_teb(fname,lmax=3000) :
    """
    Read TEB power spectrum in CAMB format into l,C_l
    """
    data=np.loadtxt(fname,unpack=True)
    larr=data[0]
    clteb=np.zeros([len(larr)+2,4])
    clteb[2:,0]=data[1]*2*np.pi/(larr*(larr+1))
    clteb[2:,1]=data[2]*2*np.pi/(larr*(larr+1))
    clteb[2:,2]=data[3]*2*np.pi/(larr*(larr+1))
    clteb[2:,3]=data[4]*2*np.pi/(larr*(larr+1))
    
    return np.arange(lmax),clteb[:lmax]

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

class GaussSt(object) :
    """
    Class to manage statistics of multivariate GRFs
    """
    def __init__(self,nmaps) :
        self.nmaps=nmaps
        self.nmaps_pol=2*nmaps
        self.ind_2d=np.zeros([(self.nmaps*(self.nmaps+1))/2,2],dtype=int)
        id1d=0
        for i in np.arange(self.nmaps) :
            for j in np.arange(self.nmaps-i) :
                self.ind_2d[id1d,:]=[j,i+j]
                id1d+=1
        self.ind_2d_pol=np.zeros([(self.nmaps_pol*(self.nmaps_pol+1))/2,2],dtype=int)
        id1d=0
        for i in np.arange(self.nmaps_pol) :
            for j in np.arange(self.nmaps_pol-i) :
                self.ind_2d_pol[id1d,:]=[j,i+j]
                id1d+=1

    def gaussian_covariance(self,mat,ell_weights,use_pol=True) :
        """
        Computes Gaussian covariance matrix for an input set of 
        power spectra.
        mat : matrix of input power spectra [ell,nmaps,nmaps]
        ell_weights : weight of each multipole (e.g. 2./(fsky*(2*ell+1)))
        use_pol : should I assume that mat contains EE, EB and BB?
        """
        if len(mat)!=len(ell_weights) :
            raise ValueError("C_ells and weights are not compatible")
        if use_pol :
            ids_use=self.ind_2d_pol
            nmaps=self.nmaps_pol
        else :
            ids_use=self.ind_2d
            nmaps=self.nmaps
        if len(mat[0]) != nmaps :
            raise ValueError("Input power spectrum doesn't match expected size")
        nvec=(nmaps*(nmaps+1))/2

        covar=np.zeros([len(mat),nvec,nvec])
        for iv1 in np.arange(nvec) :
            i_a=ids_use[iv1,0]; i_b=ids_use[iv1,1]
            for iv2 in np.arange(nvec) :
                i_c=ids_use[iv2,0]; i_d=ids_use[iv2,1]
                covar[:,iv1,iv2]=(mat[:,i_a,i_c]*mat[:,i_b,i_d]+mat[:,i_a,i_d]*mat[:,i_b,i_c])
        covar*=ell_weights[:,None,None]

        return covar

    def wrap_vector(self,vec,use_pol=True,transpose_in=False) :
        """
        Wrap vector of power spectra into a matrix.
        vec : input vector with shape [n_ell,n_maps].
        use_pol : does vec contain EE, EB and BB?
        transpose_in : if True, input vector has shape [n_maps,n_ell]
        """
        if transpose_in :
            vec_use=np.transpose(vec)
        else :
            vec_use=vec
        if use_pol :
            ids_use=self.ind_2d_pol
            nmaps=self.nmaps_pol
        else :
            ids_use=self.ind_2d
            nmaps=self.nmaps
        if len(vec_use[0]) != (nmaps*(nmaps+1))/2 :
            raise ValueError("Input vector doesn't match expected size")

        mat_out=np.zeros([len(vec_use),nmaps,nmaps])
        for ivec in np.arange(len(vec_use[0])) :
            i=ids_use[ivec,0]
            j=ids_use[ivec,1]
            mat_out[:,i,j]=vec_use[:,ivec]
            if j!=i :
                mat_out[:,j,i]=vec_use[:,ivec]
        return mat_out
        
    def unwrap_matrix(self,mat,use_pol=True,transpose_out=False) :
        """
        Unwrap matrix of power spectra into a vector.
        mat : input power spectra with shape [n_ell,n_maps,n_maps]
        use_pol : does mat contain EE, EB and BB?
        transpose_out : if True, the output will have shape [n_spec,n_ell].
                        otherwise [n_ell,n_spec]
        """
        if use_pol :
            id_use=self.ind_2d_pol
            nmaps=self.nmaps_pol
        else :
            id_use=self.ind_2d
            nmaps=self.nmaps
        if len(mat[0]) != nmaps :
            raise ValueError("Input power spectrum doesn't match expected size")

        vec_out=np.zeros([len(mat),(nmaps*(nmaps+1))/2])
        for ivec in np.arange(len(vec_out[0])) :
            vec_out[:,ivec]=mat[:,id_use[ivec,0],id_use[ivec,1]]

        if transpose_out :
            return np.transpose(vec_out)
        else :
            return vec_out

    def cl_gaussian_realization(self,cl_matrix,ell_weights,use_pol=True,return_matrix=True) :
        """
        Generate Gaussian realization of the power spectra.
        cl_matrix : matrix of input power spectra. Shape should be [n_ell,n_maps,n_maps]
        ell_weights : weight of each multipole in the covariance matrix
        use_pol : does cl_matrix include EE, EB and BB?
        return_matrix : do you want the output wrapped into a matrix?
        """
        mean=self.unwrap_matrix(cl_matrix,use_pol=use_pol)
        covar=self.gaussian_covariance(cl_matrix,ell_weights,use_pol=use_pol)
        chol=np.linalg.cholesky(covar)
        uvec=np.random.randn(len(mean),len(mean[0]))
        x_vec=mean+np.sum(chol*uvec[:,None,:],axis=2)

        if return_matrix :
            x_mat=self.wrap_vector(x_vec,use_pol=use_pol)
            return x_mat
        else :
            return x_vec

    def anafast_multimap(self,maps,return_matrix=True,transpose_out=False) :
        """
        Return multi-frequency power spectrum.
        maps : input maps. Either a 2D array of spin-0 maps [nmaps,npix]
               or a 3D array of spin-2 maps [nmaps,2,npix]
        return_matrix : do you want to return the power spectra as a matrix?
        transpose_out : if returning a vector, do you want it with shape [nspec,nell]?
        """
        if len(np.shape(maps))==3 :
            use_pol=True
        else :
            use_pol=False
            
        if use_pol :
            nmaps=len(maps)
            nside=hp.npix2nside(len(maps[0,0]))
            cl_mat=np.zeros([3*nside,2*nmaps,2*nmaps])
            dum_t=np.zeros_like(maps[0,0])
            for i in np.arange(nmaps) :
                mp1=[dum_t,maps[i,0],maps[j,1]]
                for j in np.arange(i,nmaps) :
                    mp2=[dum_t,maps[j,0],maps[j,1]]
                    ctt,cee,cbb,cte,ceb,ctb=hp.anafast(mp1,map2=mp2,pol=True)
                    cl_mat[:,i,j]=cee; cl_mat[:,i+nmaps,j+nmaps]=cbb
                    cl_mat[:,i+nmaps,j]=ceb; cl_mat[:,i,j+nmaps]=ceb
                    if i!=j :
                        cl_mat[:,j,i]=cee; cl_mat[:,j+nmaps,i+nmaps]=cbb
                        cl_mat[:,j+nmaps,i]=ceb; cl_mat[:,j,i+nmaps]=ceb
            if not return_matrix :
                return self.unwrap_matrix(cl_mat,use_pol=use_pol,transpose_out=transpose_out)
            else :
                return cl_mat
        else :
            cl_vec=np.array(hp.anafast(maps,pol=False))
            if return_matrix :
                return self.wrap_vector(cl_vec,use_pol=use_pol,transpose_in=True)
            else :
                if transpose_out :
                    return cl_vec
                else :
                    return np.transpose(cl_vec)
                    
    def synfast_multimap(self,nside,cl_matrix_ee,cl_matrix_bb=None,cl_matrix_eb=None,return_pol=True) :
        """
        Generates a GRF for a matrix of power spectra.
        nside : resolution
        cl_matrix_ee : matrix of power spectra
        cl_matrix_bb, cl_matrix_ee : if present, it will be assumed that the user wants to
                                     generate spin-2 GRFs
        return_pol : flag to force spin-2 GRFs even if cl_matrix_bb and cl_matrix_eb are not provided.
        """
        nell=len(cl_matrix_ee)
        if len(cl_matrix_ee[0]) != self.nmaps :
            raise ValueError("Input power spectrum doesn't match expected size")
        clee_use=cl_matrix_ee
        if return_pol :
            if cl_matrix_bb is None :
                clbb_use=np.zeros_like(cl_matrix_ee)
            else :
                clbb_use=cl_matrix_bb
                if len(cl_matrix_bb[0]) != self.nmaps :
                    raise ValueError("Input power spectrum doesn't match expected size")
            if cl_matrix_eb is None :
                cleb_use=np.zeros_like(cl_matrix_ee)
            else :
                cleb_use=cl_matrix_eb
                if len(cl_matrix_eb[0]) != self.nmaps :
                    raise ValueError("Input power spectrum doesn't match expected size")


        if return_pol :
            nmaps_total=self.nmaps_pol
            cl_matrix=np.zeros([nell,self.nmaps_pol,self.nmaps_pol])
            cl_matrix[:,:self.nmaps,:self.nmaps]=clee_use
            cl_matrix[:,:self.nmaps,self.nmaps:]=cleb_use
            cl_matrix[:,self.nmaps:,:self.nmaps]=cleb_use
            cl_matrix[:,self.nmaps:,self.nmaps:]=clbb_use
            ids_use=self.ind_2d_pol
        else :
            nmaps_total=self.nmaps
            ids_use=self.ind_2d
            cl_matrix=clee_use.copy()

        cl_vector=self.unwrap_matrix(cl_matrix,use_pol=return_pol,transpose_out=True)
        maps=hp.synfast(cl_vector,nside,pol=False,new=True,verbose=False)

        if return_pol :
            maps_out=[]
            for i in np.arange(self.nmaps) :
                mp_e=maps[i]
                mp_b=maps[i+self.nmaps]
                mp_t=np.zeros_like(mp_e)
                mp_t,mp_q,mp_u=hp.alm2map(hp.map2alm([mp_t,mp_e,mp_b],pol=False),
                                          nside,pol=True,verbose=False)
                maps_out.append([mp_q,mp_u])
            maps_out=np.array(maps_out)
        else :
            maps_out=maps.copy()

        return maps_out

class CMBModel(object) :
    def __init__(self,fname_tensors="data/planck1_r1p00_tensCls.dat",fname_lensing="data/planck1_r0p00_lensedtotCls.dat") :
        lp,cteb_prim=read_cl_teb(fname_tensors)
        ll,cteb_lens=read_cl_teb(fname_lensing)
        self.cl_bb_primf=interp1d(lp,cteb_prim[:,2])
        self.cl_bb_lensf=interp1d(ll,cteb_lens[:,2])
        self.cl_ee_primf=interp1d(lp,cteb_prim[:,1])
        self.cl_ee_lensf=interp1d(ll,cteb_lens[:,1])
        
    
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
                 contain_cmb=True,contain_E=False,contain_B=True,
                 cmb_model=None,r_prim=0.,A_lens=1.,
                 contain_sync=True,A_sync_BB=3.80,A_sync_EE=3.80,A_sync_EB=0.,
                 alpha_sync=-0.60,beta_sync=-3.0,nu0_sync=23.,xi_sync=50.,c_sync=0.,
                 contain_dust=True,A_dust_BB=4.25,A_dust_EE=4.25,A_dust_EB=0.,
                 alpha_dust=-0.42,beta_dust=1.59,temp_dust=20.,nu0_dust=353.,xi_dust=50.,
                 contain_dust_sync_corr=False,r_dust_sync=0.0,
                 contain_CO1=False,A_CO1_BB=0.5,A_CO1_EE=0.5,A_CO1_EB=0.0,alpha_CO1=-0.42,
                 contain_CO2=False,A_CO2_BB=0.5,A_CO2_EE=0.5,A_CO2_EB=0.0,alpha_CO2=-0.42,**kwargs) :
        """
        Initializes SkyModel structure
        contain_cmb: include cmb?
        contain_E: include both E and B?
        cmb_model: CMBModel structure (will create a new one if None)
        contain_sync: include synchrotron?
        A_sync_BB, A_sync_EE, A_sync_EB, alpha_sync, beta_sync, nu0_sync, xi_sync, c_sync:
                D^sync_ell(nu1,nu2) = A_sync * (ell/80.)**alpha_sync *
                                      f_sync(nu1,nu0_sync,beta_sync) * f_sync(nu2,nu0_sync,beta_sync,c_sync) *
                                      exp[-0.5*(log(nu1/nu2)/xi_sync)**2]
        contain_dust: include dust?
        A_dust_BB, A_dust_EE, A_dust_EB, alpha_dust, beta_dust, temp_dust, nu0_dust, xi_dust:
                D^dust_ell(nu1,nu2) = A_dust * (ell/80.)**alpha_dust *
                                      f_dust(nu1,nu0_dust,beta_dust,temp_dust) * f_dust(nu2,nu0_dust,beta_dust,temp_dust) *
                                      exp[-0.5*(log(nu1/nu2)/xi_dust)**2]
        contain_dust_sync_corr: include sychrotron-dust correlation?
        r_dust_sync: synchrotron-dust correlation coefficient
        contain_CO1: include CO 1->0 at 115 GHz
        A_CO1_BB, A_CO1_EE, A_CO1_EB, alpha_CO1: see similar parameters for dust
        contain_CO2: include CO 2->1 at 230 GHz
        A_CO2_BB, A_CO2_EE, A_CO2_EB, alpha_CO2: see similar parameters for dust
        """

        self.contain_cmb=contain_cmb
        self.contain_E=contain_E
        self.contain_B=contain_B
        self.contain_sync=contain_sync
        self.contain_dust=contain_dust
        self.contain_CO1=contain_CO1
        self.contain_CO2=contain_CO2
        
        ncomp=0
        if self.contain_cmb :
            if cmb_model is None :
                self.cmb_model=CMBModel()
            else :
                self.cmb_model=cmb_model
            ncomp+=1
            self.r_prim=r_prim
            self.A_lens=A_lens
            
        if self.contain_sync :
            ncomp+=1
            self.A_sync_BB=A_sync_BB
            self.A_sync_EE=A_sync_EE
            self.A_sync_EB=A_sync_EB
            self.alpha_sync=alpha_sync
            self.beta_sync=beta_sync
            self.c_sync=c_sync
            self.nu0_sync=nu0_sync
            self.xi_sync=xi_sync
            
        if self.contain_dust :
            ncomp+=1
            self.A_dust_BB=A_dust_BB
            self.A_dust_EE=A_dust_EE
            self.A_dust_EB=A_dust_EB
            self.alpha_dust=alpha_dust
            self.beta_dust=beta_dust
            self.temp_dust=temp_dust
            self.nu0_dust=nu0_dust
            self.xi_dust=xi_dust

        self.contain_dust_sync_corr=False
        if self.contain_sync and self.contain_dust :
            if contain_dust_sync_corr :
                self.contain_dust_sync_corr=contain_dust_sync_corr
                self.r_dust_sync=r_dust_sync
            
        if self.contain_CO1 :
            ncomp+=1
            self.A_CO1_BB=A_CO1_BB
            self.A_CO1_EE=A_CO1_EE
            self.A_CO1_EB=A_CO1_EB
            self.alpha_CO1=alpha_CO1
            
        if self.contain_CO2 :
            ncomp+=1
            self.A_CO2_BB=A_CO2_BB
            self.A_CO2_EE=A_CO2_EE
            self.A_CO2_EB=A_CO2_EB
            self.alpha_CO2=alpha_CO2

        self.ncomp=ncomp

    def is_consistent(self) :
        """
        Returns True if the parameters of this sky model are consistent (False otherwise)
        """
        if self.contain_cmb :
            if self.A_lens<0 : return False
        if self.contain_sync :
            if self.A_sync_BB<0 : return False
            if self.A_sync_EB<0 : return False
            if self.A_sync_EE<0 : return False
        if self.contain_dust :
            if self.A_dust_BB<0 : return False
            if self.A_dust_EB<0 : return False
            if self.A_dust_EE<0 : return False
        if self.contain_CO1 :
            if self.A_CO1_BB<0 : return False
            if self.A_CO1_EB<0 : return False
            if self.A_CO1_EE<0 : return False
        if self.contain_CO2 :
            if self.A_CO2_BB<0 : return False
            if self.A_CO2_EB<0 : return False
            if self.A_CO2_EE<0 : return False

        return True
            
        
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
            if name=='A_sync_BB' :
                self.A_sync_BB=value
                found=True
            if name=='A_sync_EE' :
                self.A_sync_EE=value
                found=True
            if name=='A_sync_EB' :
                self.A_sync_EB=value
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
            if name=='A_dust_BB' :
                self.A_dust_BB=value
                found=True
            if name=='A_dust_EE' :
                self.A_dust_EE=value
                found=True
            if name=='A_dust_EB' :
                self.A_dust_EB=value
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
            if name=='A_CO1_BB' :
                self.A_CO1_BB=value
                found=True
            if name=='A_CO1_EE' :
                self.A_CO1_EE=value
                found=True
            if name=='A_CO1_EB' :
                self.A_CO1_EB=value
                found=True
            if name=='alpha_CO1' :
                self.alpha_CO1=value
                found=True
            
        if self.contain_CO2 :
            if name=='A_CO2_BB' :
                self.A_CO2_BB=value
                found=True
            if name=='A_CO2_EE' :
                self.A_CO2_EE=value
                found=True
            if name=='A_CO2_EB' :
                self.A_CO2_EB=value
                found=True
            if name=='alpha_CO2' :
                self.alpha_CO2=value
                found=True

        if not found :
            ValueError("Parameter "+name+" not found")
        
    def cl_cmb(self,l) :
        clp=(self.cmb_model.cl_bb_primf)(l)
        cll=(self.cmb_model.cl_bb_lensf)(l)
        return self.r_prim*clp+self.A_lens*cll

    def cl_cmb_ee(self,l) :
        clp=(self.cmb_model.cl_ee_primf)(l)
        cll=(self.cmb_model.cl_ee_lensf)(l)
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
        cl_comp_ee=np.zeros([nell,self.ncomp])
        cl_comp_eb=np.zeros([nell,self.ncomp])
        cl_comp_bb=np.zeros([nell,self.ncomp])

        #Fill F matrix
        f_matrix=np.zeros([nnu,self.ncomp])
        icomp=0
        if self.contain_cmb :
            f_matrix[:,icomp]=freq_evolve("BB",None,None,None,None,nus)
            icomp_cmb=icomp
            icomp+=1
        if self.contain_sync :
            f_matrix[:,icomp]=freq_evolve("PL",self.nu0_sync,self.beta_sync,self.c_sync,None,nus)
            icomp_sync=icomp
            icomp+=1
        if self.contain_dust :
            f_matrix[:,icomp]=freq_evolve("mBB",self.nu0_dust,self.beta_dust,self.temp_dust,None,nus)
            icomp_dust=icomp
            icomp+=1
        if self.contain_CO1 :
            f_matrix[:,icomp]=freq_evolve("CO1",None,None,None,None,nus)
            icomp_CO1=icomp
            icomp+=1
        if self.contain_CO2 :
            f_matrix[:,icomp]=freq_evolve("CO2",None,None,None,0.5,nus)
            icomp_CO2=icomp
            icomp+=1

        #Fill decorrelation matrix
        decorr_nu=np.ones([nnu,nnu,self.ncomp])
        if self.contain_sync :
            decorr_nu[:,:,icomp_sync]=np.exp(-0.5*(np.log(nus[:,None]/nus[None,:])/self.xi_sync)**2)
        if self.contain_dust :
            decorr_nu[:,:,icomp_dust]=np.exp(-0.5*(np.log(nus[:,None]/nus[None,:])/self.xi_dust)**2)

        #Fill component power spectra
        if self.contain_cmb :
            if self.contain_B :
                cl_comp_bb[:,icomp_cmb]=self.cl_cmb(ls)
            if self.contain_E :
                cl_comp_ee[:,icomp_cmb]=self.cl_cmb_ee(ls)
        if self.contain_sync :
            if self.contain_B :
                cl_comp_bb[:,icomp_sync]=cl_plaw(ls,80.,self.A_sync_BB,self.alpha_sync)
                if self.contain_E :
                    cl_comp_eb[:,icomp_sync]=cl_plaw(ls,80.,self.A_sync_EB,self.alpha_sync)
            if self.contain_E :
                cl_comp_ee[:,icomp_sync]=cl_plaw(ls,80.,self.A_sync_EE,self.alpha_sync)
        if self.contain_dust :
            if self.contain_B :
                cl_comp_bb[:,icomp_dust]=cl_plaw(ls,80.,self.A_dust_BB,self.alpha_dust)
                if self.contain_E :
                    cl_comp_eb[:,icomp_dust]=cl_plaw(ls,80.,self.A_dust_EB,self.alpha_dust)
            if self.contain_E :
                cl_comp_ee[:,icomp_dust]=cl_plaw(ls,80.,self.A_dust_EE,self.alpha_dust)
        if self.contain_CO1 :
            if self.contain_B :
                cl_comp_bb[:,icomp_CO1]=cl_plaw(ls,80.,self.A_CO1_BB,self.alpha_CO1)
                if self.contain_E :
                    cl_comp_eb[:,icomp_CO1]=cl_plaw(ls,80.,self.A_CO1_EB,self.alpha_CO1)
            if self.contain_E :
                cl_comp_ee[:,icomp_CO1]=cl_plaw(ls,80.,self.A_CO1_EE,self.alpha_CO1)
        if self.contain_CO2 :
            if self.contain_B :
                cl_comp_bb[:,icomp_CO2]=cl_plaw(ls,80.,self.A_CO2_BB,self.alpha_CO2)
                if self.contain_E :
                    cl_comp_eb[:,icomp_CO2]=cl_plaw(ls,80.,self.A_CO2_EB,self.alpha_CO2)
            if self.contain_E :
                cl_comp_ee[:,icomp_CO2]=cl_plaw(ls,80.,self.A_CO2_EE,self.alpha_CO2)

        rcorr=np.identity(self.ncomp)
        if self.contain_dust_sync_corr :
            rcorr[icomp_sync,icomp_dust]=rcorr[icomp_dust,icomp_sync]=self.r_dust_sync

        a_nu=decorr_nu[:,:,:]*f_matrix[:,None,:]*f_matrix[None,:]
        csqr_matrix_comp_bb=np.sqrt(np.fabs(cl_comp_bb[:,None,None,:]*a_nu[None,:,:,:]))
        c_matrix_comp_bb=np.sum(np.dot(csqr_matrix_comp_bb,rcorr)*csqr_matrix_comp_bb,axis=3)
        csqr_matrix_comp_eb=np.sqrt(np.fabs(cl_comp_eb[:,None,None,:]*a_nu[None,:,:,:]))
        c_matrix_comp_eb=np.sum(np.dot(csqr_matrix_comp_eb,rcorr)*csqr_matrix_comp_eb,axis=3)
        csqr_matrix_comp_ee=np.sqrt(np.fabs(cl_comp_ee[:,None,None,:]*a_nu[None,:,:,:]))
        c_matrix_comp_ee=np.sum(np.dot(csqr_matrix_comp_ee,rcorr)*csqr_matrix_comp_ee,axis=3)

        c_out=np.zeros([nell,2*nnu,2*nnu])
        c_out[:,:nnu,:nnu]=c_matrix_comp_ee
        c_out[:,:nnu,nnu:]=c_matrix_comp_eb
        c_out[:,nnu:,:nnu]=c_matrix_comp_eb
        c_out[:,nnu:,nnu:]=c_matrix_comp_bb

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
        self.beam_sigmas=np.array([1.4])
        self.gains=np.array([1.])
        self.gst=GaussSt(len(self.freqs))
        
    def getFsky(self) :
        return self.fsky

    def getFrequencies(self) :
        return self.freqs

    def getNoiseFlat(self) :
        return self.noi_flat

    def getNoiseCls(self,ells) :
        ells=np.asarray(ells)
        scalar_input=False
        if ells.ndim == 0 :
            ells=ells[None]
            scalar_input=True
        
        noises=np.zeros([ells,len(self.freqs),len(self.freqs)])

        if scalar_input :
            noises=noises[0,:,:]

        return noises

    def beamCell(self,ells,cl_in) :
        return cl_in
    
    def observeCl(self,ells,sky) :
        """
        Produce cross-band powers as observed by this experiment.
        This implies generating the true C_ells defined by a SkyModel
        object (sky) and perturbing them with this experiment's gains and noise.
        """
        nnu=len(self.freqs)
        noi=self.getNoiseCls(ells)
        ctrue=sky.get_Cl(self.freqs,ells)
        gainmat=self.gains[:,None]*self.gains[None,:]
        ctrue[:,:nnu,:nnu]*=gainmat[None,:,:]
        ctrue[:,:nnu,nnu:]*=gainmat[None,:,:]
        ctrue[:,nnu:,:nnu]*=gainmat[None,:,:]
        ctrue[:,nnu:,nnu:]*=gainmat[None,:,:]
        ctrue[:,nnu:,nnu:]+=noi
        ctrue[:,:nnu,:nnu]+=noi
        return ctrue

    def genNoiseMaps(self,nside) :
        ells=np.arange(3*nside)
        nlmat=self.beamCell(ells,self.getNoiseCls(ells))
        mps_out=self.gst.synfast_multimap(nside,nlmat,cl_matrix_bb=nlmat)
        return mps_out

    def genSkyMaps(self,nside,sky) :
        ells=np.arange(3*nside)
        nnu=len(self.freqs)
        cl_sky=sky.get_Cl(self.freqs,ells)
        cl_ee=self.beamCell(ells,cl_sky[:,:nnu,:nnu])
        cl_bb=self.beamCell(ells,cl_sky[:,nnu:,nnu:])
        cl_eb=self.beamCell(ells,cl_sky[:,:nnu,nnu:])
        maps=self.gst.synfast_multimap(nside,cl_ee,cl_matrix_bb=cl_bb,cl_matrix_eb=cl_eb)
        mps_out=np.zeros([len(self.freqs),2,hp.nside2npix(nside)])
        mps_out[:,0,:]=maps[:,0,:]; mps_out[:,1,:]=maps[:,1,:];
        return mps_out

def array_from_kws(string,dictio) :
    ix=0
    xs=[]
    while string+"%d"%ix in dictio :
        xs.append(dictio[string+"%d"%ix])
        ix+=1
    return np.array(xs)

def beamsFromFreq(freqs,diam) :
    beam_fwhm=1315.0/(freqs*diam)
    return beam_fwhm

class ExperimentSimple(ExperimentBase) :
    """
    Simple experiment with a bunch of frequencies, noise levels,
    ell_knees and alpha_knees and a constant aperture
    """

    def __init__(self,fsky=None,name=None,freqs=None,gains=None,
                 noi_flat=None,alpha_knee=None,ell_knee=None,beams=None,**kwargs) :
        """
        fsky: sky fraction
        name: experiment's name
        freqs: frequencies in GHz
        gains: relative gains at each frequency
        noi_flat: flat noise levels at each frequency (in uK_CMB arcmin)
        alpha_knee: knee exponent for non-flat noise
        ell_knee: knee multipole for non-flat noise
        beams: beam FWHM in arcmin
        """
        self.typestr="Simple"
        if name is None : self.name="default"
        else : self.name=name
        if fsky is None : self.fsky=1.
        else : self.fsky=fsky
        
        self.freqs=array_from_kws("freq",kwargs)
        if len(self.freqs)==0 :
            if freqs is None :
                raise ValueError("No frequencies found")
            else :
                freqs=np.asarray(freqs)
                if freqs.ndim==0 :
                    freqs=freqs[None]
                self.freqs=freqs.copy()
            
        if gains is None :
            self.gains=array_from_kws("gain",kwargs)
            if len(self.gains)==0 :
                self.gains=np.ones(len(self.freqs))*1.
        else :
            if np.asarray(gains).ndim==0 :
                self.gains=gains*np.ones(len(self.freqs))
            else :
                self.gains=gains.copy()
        if len(self.gains)!=len(self.freqs) :
            raise ValueError("gains and freqs should have the same number of elements")
        
        if noi_flat is None :
            self.noi_flat=array_from_kws("noi_flat",kwargs)
            if len(self.noi_flat)==0 :
                self.noi_flat=np.ones(len(self.freqs))*1E20
        else :
            if np.asarray(noi_flat).ndim==0 :
                self.noi_flat=noi_flat*np.ones(len(self.freqs))
            else :
                self.noi_flat=noi_flat.copy()
        if len(self.noi_flat)!=len(self.freqs) :
            raise ValueError("noi_flat and freqs should have the same number of elements")

        if beams is None :
            beams_fwhm=array_from_kws("beam",kwargs)
            if len(beams_fwhm)==0 :
                self.beam_sigmas=np.ones(len(self.freqs))*1E-4
        else :
            if np.asarray(beams).ndim==0 :
                self.beam_sigmas=np.ones(len(self.freqs))*AMIN2RAD*beams/2.355
            else :
                self.beam_sigmas=AMIN2RAD*beams/2.355
        if len(self.beam_sigmas)!=len(self.freqs) :
            raise ValueError("beams and freqs should have the same number of elements")

        if alpha_knee is None : self.alpha_knee=-1.9*np.ones(len(self.freqs))
        else :
            if np.asarray(alpha_knee).ndim==0 :
                self.alpha_knee=alpha_knee*np.ones(len(self.freqs))
            else :
                if len(alpha_knee)!=len(self.freqs) :
                    raise ValueError("alpha_knee and freqs should have the same number of elements")
                self.alpha_knee=alpha_knee.copy()
        if ell_knee is None : self.ell_knee=np.ones(len(self.freqs))*0.01
        else :
            if np.asarray(ell_knee).ndim==0 :
                self.ell_knee=ell_knee*np.ones(len(self.freqs))
            else :
                if len(ell_knee)!=len(self.freqs) :
                    raise ValueError("ell_knee and freqs should have the same number of elements")
                self.ell_knee=ell_knee.copy()
        
        self.gst=GaussSt(len(self.freqs))
                
    def getNoiseCls(self,ells) :
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

        nell=(np.ones(len(ells))[:,None]+((ells[:,None]+0.01)/self.ell_knee[None,:])**self.alpha_knee[None,:])
        nell*=np.exp((ells*(ells+1.))[:,None]*((self.beam_sigmas[None,:])**2))
        nell*=(nlevels**2)[None,:]

        noises=np.zeros([len(ells),len(self.freqs),len(self.freqs)])
        for i,f in enumerate(self.freqs) :
            noises[:,i,i]=nell[:,i]

        if scalar_input :
            noises=noises[0,:,:]

        return noises

    def beamCell(self,ells,cl_in) :
        if len(cl_in[0])!=len(self.freqs) :
            raise ValueError("Incompatible C_ell")
        if len(cl_in)!=len(ells) :
            raise ValueError("Incompatible C_ell")

        bm=np.exp(-0.5*(ells*(ells+1.))[:,None,None]*((self.beam_sigmas**2)[None,:,None]+(self.beam_sigmas**2)[None,None,:]))
        return cl_in*bm

class ExperimentDouble(ExperimentBase) :
    """
    Experiment made up of two separate telescopes.
    Both telescopes should have the same frequency channels,
    but not necessarily the same noise properties or beams.
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
                                   alpha_knee=exp1.alpha_knee,ell_knee=exp1.ell_knee,beams=exp1.beam_sigmas*2.355/AMIN2RAD)
        self.exp2=ExperimentSimple(name=name,freqs=freqs,fsky=fsky,
                                   gains=gains,noi_flat=exp2.noi_flat,
                                   alpha_knee=exp2.alpha_knee,ell_knee=exp2.ell_knee,beams=exp2.beam_sigmas*2.355/AMIN2RAD)
        self.typestr="Double"
        self.name=name
        self.fsky=fsky
        self.freqs=freqs.copy()
        self.gains=self.exp1.gains.copy()
        self.noi_flat=1./np.sqrt(1./self.exp1.noi_flat**2+1./self.exp2.noi_flat**2)
        self.gst=GaussSt(len(self.freqs))

    def getNoiseCls(self,ells) :
        noi1=self.exp1.getNoiseCls(ells)
        noi2=self.exp2.getNoiseCls(ells)
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

def construct_parameters(sky,pars0=[]) :
    pars_arr=pars0[:]
    if sky.contain_cmb :
        if sky.r_prim>0.001 :
            drprim=0.2*sky.r_prim
            onesided=oFalse
        else :
            drprim=5E-4
            onesided=True
        pars_arr.append({'val':sky.r_prim ,'dval':drprim,'name':'r_prim'  ,
                         'label':'$r$'                  ,'vary':True,
                         'prior_th':[-1.,1.],'prior_gau':None})
        pars_arr.append({'val':sky.A_lens,'dval':0.01 ,'name':'A_lens'  ,
                         'label':'$A_{\\rm lens}$'      ,'vary':True,
                         'prior_th':[0.,2.],'prior_gau':None})
    if sky.contain_sync :
        pars_arr.append({'val':sky.A_sync_BB ,'dval':0.1  ,'name':'A_sync_BB'  ,
                         'label':'$A^{\\rm BB}_{\\rm sync}$'      ,'vary':True,
                         'prior_th':[0.,1000.],'prior_gau':None})
        if sky.contain_E :
            pars_arr.append({'val':sky.A_sync_EB ,'dval':0.1  ,'name':'A_sync_EB'  ,
                             'label':'$A^{\\rm EB}_{\\rm sync}$'      ,'vary':False,
                             'prior_th':None,'prior_gau':None})
            pars_arr.append({'val':sky.A_sync_EE ,'dval':0.1  ,'name':'A_sync_EE'  ,
                             'label':'$A^{\\rm EE}_{\\rm sync}$'      ,'vary':True,
                             'prior_th':[0.,1000.],'prior_gau':None})
        pars_arr.append({'val':sky.alpha_sync,'dval':0.01 ,'name':'alpha_sync'  ,
                         'label':'$\\alpha_{\\rm sync}$','vary':True,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.beta_sync,'dval':0.005,'name':'beta_sync'  ,
                         'label':'$\\beta_{\\rm sync}$' ,'vary':True,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.nu0_sync,'dval':-1   ,'name':'nu0_sync',
                         'label':'$\\nu_{\\rm sync}$'   ,'vary':False,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.xi_sync  ,'dval':0.2  ,'name':'xi_sync'   ,
                         'label':'$\\xi_{\\rm sync}$'   ,'vary':False,
                         'prior_th':None,'prior_gau':None})
    if sky.contain_dust :
        pars_arr.append({'val':sky.A_dust_BB,'dval':0.1  ,'name':'A_dust_BB'  ,
                         'label':'$A^{\\rm BB}_{\\rm dust}$'      ,'vary':True,
                         'prior_th':[0,1000.],'prior_gau':None})
        if sky.contain_E :
            pars_arr.append({'val':sky.A_dust_EB,'dval':0.1  ,'name':'A_dust_EB'  ,
                             'label':'$A^{\\rm EB}_{\\rm dust}$'      ,'vary':False,
                             'prior_th':None,'prior_gau':None})
            pars_arr.append({'val':sky.A_dust_EE,'dval':0.1  ,'name':'A_dust_EE'  ,
                             'label':'$A^{\\rm EE}_{\\rm dust}$'      ,'vary':True,
                             'prior_th':[0.,1000.],'prior_gau':None})
        pars_arr.append({'val':sky.alpha_dust,'dval':0.01 ,'name':'alpha_dust'  ,
                         'label':'$\\alpha_{\\rm dust}$','vary':True,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.beta_dust,'dval':0.002,'name':'beta_dust'  ,
                         'label':'$\\beta_{\\rm dust}$' ,'vary':True,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.temp_dust,'dval':0.1  ,'name':'temp_dust'  ,
                         'label':'$T_{\\rm dust}$'      ,'vary':True,
                         'prior_th':[10.,30.],'prior_gau':None})
        pars_arr.append({'val':sky.nu0_dust,'dval':-1   ,'name':'nu0_dust',
                         'label':'$\\nu_{\\rm dust}$'   ,'vary':False,
                         'prior_th':None,'prior_gau':None})
        pars_arr.append({'val':sky.xi_dust ,'dval':0.2  ,'name':'xi_dust'   ,
                         'label':'$\\xi_{\\rm dust}$'   ,'vary':False,
                         'prior_th':None,'prior_gau':None})
    if sky.contain_dust_sync_corr :
        pars_arr.append({'val':sky.r_dust_sync,'dval':0.1  ,'name':'r_dust_sync'  ,
                         'label':'$r_{\\rm d-s}$'       ,'vary':True,
                         'prior_th':None,'prior_gau':None})
    if sky.contain_CO1 :
        pars_arr.append({'val':sky.A_CO1_BB,'dval':0.05 ,'name':'A_CO1_BB'   ,
                         'label':'$A^{1BB}_{\\rm CO}$'      ,'vary':True,
                         'prior_th':[0,1000.],'prior_gau':None})
        if sky.contain_E :
            pars_arr.append({'val':sky.A_CO1_EB,'dval':0.05 ,'name':'A_CO1_EB'   ,
                             'label':'$A^{1EB}_{\\rm CO}$'      ,'vary':False,
                             'prior_th':None,'prior_gau':None})
            pars_arr.append({'val':sky.A_CO1_EE,'dval':0.05 ,'name':'A_CO1_EE'   ,
                             'label':'$A^{1EE}_{\\rm CO}$'      ,'vary':True,
                             'prior_th':[0.,1000.],'prior_gau':None})
        pars_arr.append({'val':sky.alpha_CO1,'dval':0.1  ,'name':'alpha_CO1'   ,
                         'label':'$\\alpha^1_{\\rm CO}$','vary':True,
                         'prior_th':None,'prior_gau':None})
    if sky.contain_CO2 :
        pars_arr.append({'val':sky.A_CO2_BB,'dval':0.05 ,'name':'A_CO2_BB'   ,
                         'label':'$A^{2BB}_{\\rm CO}$'      ,'vary':True,
                         'prior_th':[0.,1000.],'prior_gau':None})
        if sky.contain_E :
            pars_arr.append({'val':sky.A_CO2_EB,'dval':0.05 ,'name':'A_CO2_EB'   ,
                             'label':'$A^{2EB}_{\\rm CO}$'      ,'vary':False,
                             'prior_th':None,'prior_gau':None})
            pars_arr.append({'val':sky.A_CO2_EE,'dval':0.05 ,'name':'A_CO2_EE'   ,
                             'label':'$A^{2EE}_{\\rm CO}$'      ,'vary':True,
                             'prior_th':[0.,1000.],'prior_gau':None})
        pars_arr.append({'val':sky.alpha_CO2,'dval':0.1  ,'name':'alpha_CO2'   ,
                         'label':'$\\alpha^2_{\\rm CO}$','vary':True,
                         'prior_th':None,'prior_gau':None})
    pars_out=pd.DataFrame(pars_arr).to_records()

    return pars_out

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

    pars_all=construct_parameters(sky)
        
    fsky=xpr.getFsky()
    nu_arr=xpr.getFrequencies()
    nnu=len(nu_arr)
    l_arr=np.arange(lmin_use,lmax)
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
            for i2 in np.arange(i1,nvary) :
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

def get_fisher_from_like(l_k,p_0,d_p,a_d) :
    n_p=len(p_0)
    fisher_v=np.zeros(n_p)
    fisher_m=np.zeros([n_p,n_p])
    l1o2o=l_k(p_0,a_d)
    for i1 in np.arange(n_p) :
        dp1=d_p[i1]
        p1m2o=p_0.copy(); p1m2o[i1]-=dp1; l1m2o=l_k(p1m2o,a_d)
        p1p2o=p_0.copy(); p1p2o[i1]+=dp1; l1p2o=l_k(p1p2o,a_d)
        fisher_m[i1,i1]=-(l1p2o-2*l1o2o+l1m2o)/dp1**2
        fisher_v[i1]=(l1p2o-l1m2o)/(2*dp1)
        for i2 in np.arange(i1+1,n_p) :
            dp2=d_p[i2]
            p1m2m=p1m2o.copy(); p1m2m[i2]-=dp2; l1m2m=l_k(p1m2m,a_d)
            p1m2p=p1m2o.copy(); p1m2p[i2]+=dp2; l1m2p=l_k(p1m2p,a_d)
            p1p2m=p1p2o.copy(); p1p2m[i2]-=dp2; l1p2m=l_k(p1p2m,a_d)
            p1p2p=p1p2o.copy(); p1p2p[i2]+=dp2; l1p2p=l_k(p1p2p,a_d)
            fisher_m[i1,i2]=-(l1p2p-l1p2m-l1m2p+l1m2m)/(4*dp1*dp2)
            fisher_m[i2,i1]=fisher_m[i1,i2]
    return fisher_m,fisher_v

def get_fisher_from_like_precise(l_k,p_0,a_d) :
    n_p=len(p_0)
    def lkh(p):
#        return np.sum(p**2)
        return l_k(p,a_d)
    fisher_v=nd.Gradient(lkh)(p_0)
    fisher_m=-nd.Hessian(lkh)(p_0)
    return fisher_m,fisher_v
