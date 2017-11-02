import numpy as np

def sens_calc_LAT(N_LF,N_MF,N_HF,N_UHF,f_sky,n_years=5.0):
    S_LF_27 = 20.8
    S_LF_39 = 14.3
    S_MF_90 = 6.5
    S_MF_150 = 8.1
    S_HF_150 = 6.9
    S_HF_220 = 15.8
    S_UHF_220 = 13.8
    S_UHF_270 = 35.6

    efficiency = 0.4*.5*.85
    total_tubes = N_LF+ N_MF+ N_HF+ N_UHF

    S_27 = S_39 = S_90 = S_150 = S_220 = S_270 = 1e9 ## e.g., make the noise irrelvently high by default
    S_27 = 1./np.sqrt( N_LF * S_LF_27**-2. + S_27**-2.)
    S_39 = 1./np.sqrt( N_LF * S_LF_39**-2. + S_39**-2.)
    S_90 = 1./np.sqrt( N_MF * S_MF_90**-2. + S_90**-2.)
    S_150 = 1./np.sqrt( N_MF * S_MF_150**-2. + S_150**-2.)
    S_150 = 1./np.sqrt( N_HF * S_HF_150**-2. + S_150**-2.)
    S_220 = 1./np.sqrt( N_HF * S_HF_220**-2. + S_220**-2.)
    S_220 = 1./np.sqrt( N_UHF * S_UHF_220**-2. + S_220**-2.)
    S_270 = 1./np.sqrt( N_UHF * S_UHF_270**-2. + S_270**-2.)
    integration_time = n_years *365.* 24. * 3600. * efficiency
    sky_area = 4.*np.pi * (180/np.pi)**2. * 3600. * f_sky
    N_27 = S_27 * np.sqrt(sky_area / integration_time)
    N_39 = S_39 * np.sqrt(sky_area / integration_time)
    N_90 = S_90 * np.sqrt(sky_area / integration_time)
    N_150 = S_150 * np.sqrt(sky_area / integration_time)
    N_220 = S_220 * np.sqrt(sky_area / integration_time)
    N_270 = S_270 * np.sqrt(sky_area / integration_time)
    bands = np.array([27,39,90,150,220,270.]) ## in GHz
    noise_per_arcminute = np.array([N_27,N_39,N_90,N_150,N_220,N_270])
    return(bands,noise_per_arcminute)

def sens_calc_SAT(N_LF,N_MF,N_HF,N_UHF,f_sky,n_years=5.0):
    S_LF_27 = 18.0
    S_LF_39 = 10.6
    S_MF_90 = 3.9
    S_MF_150 = 4.6
    S_HF_150 = 4.6
    S_HF_220 = 6.8
    S_UHF_220 = 6.8
    S_UHF_270 = 16.8

    efficiency = 0.4*.5*.85
    total_tubes = N_LF+ N_MF+ N_HF+ N_UHF
    
    S_27 = S_39 = S_90 = S_150 = S_220 = S_270 = 1e9 ## e.g., make the noise irrelvently high by default
    S_27 = 1./np.sqrt( N_LF * S_LF_27**-2. + S_27**-2.)
    S_39 = 1./np.sqrt( N_LF * S_LF_39**-2. + S_39**-2.)
    S_90 = 1./np.sqrt( N_MF * S_MF_90**-2. + S_90**-2.)
    S_150 = 1./np.sqrt( N_MF * S_MF_150**-2. + S_150**-2.)
    S_150 = 1./np.sqrt( N_HF * S_HF_150**-2. + S_150**-2.)
    S_220 = 1./np.sqrt( N_HF * S_HF_220**-2. + S_220**-2.)
    S_220 = 1./np.sqrt( N_UHF * S_UHF_220**-2. + S_220**-2.)
    S_270 = 1./np.sqrt( N_UHF * S_UHF_270**-2. + S_270**-2.)
    integration_time = n_years *365.* 24. * 3600. * efficiency
    sky_area = 4.*np.pi * (180/np.pi)**2. * 3600. * f_sky
    N_27 = S_27 * np.sqrt(sky_area / integration_time)
    N_39 = S_39 * np.sqrt(sky_area / integration_time)
    N_90 = S_90 * np.sqrt(sky_area / integration_time)
    N_150 = S_150 * np.sqrt(sky_area / integration_time)
    N_220 = S_220 * np.sqrt(sky_area / integration_time)
    N_270 = S_270 * np.sqrt(sky_area / integration_time)
    bands = np.array([27,39,90,150,220,270.]) ## in GHz
    noise_per_arcminute = np.array([N_27,N_39,N_90,N_150,N_220,N_270])
    return(bands,noise_per_arcminute)
