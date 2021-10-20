'''
Collection of small functions that are useful when reducing and plotting data. 
'''


#########################################
# Chirp mass
#########################################
def Mchirp(m1, m2):
    chirp_mass = np.divide(np.power(np.multiply(m1, m2), 3./5.), np.power(np.add(m1, m2), 1./5.))
    return chirp_mass    
   

################################################
def hdf5_to_astropy(hdf5_file, group = 'SystemParameters' ):
    """convert your hdf5 table to astropy.table for easy indexing etc
    hdf5_file  =  Hdf5 file you would like to convert
    group      =  Data group that you want to acces
    """
    Data         = hdf5_file[group]#
    table = Table()
    for key in list(Data.keys()):
        table[key] =  Data[key]
    return table


#########################################
# Read data
#########################################
def read_data(loc = data_dir+'/output/COMPAS_Output_wWeights.h5', rate_key = 'Rates_mu00.025_muz-0.05_alpha-1.77_sigma01.125_sigmaz0.05_zBinned', verbose=False):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                  --> [string] Location of data
            rate_key             --> [string] group key name of COMPAS HDF5 data that contains your merger rate
            verbose              --> [bool] If you want to print statements while reading in 

        Returns:
            crude_rate_density   --> [2D float array] Intrinsic merger rate density for each binary at new crude redshiftbins in 1/yr/Gpc^3

    """


    if verbose: print('Reading ',loc)
    ################################################
    ## Open hdf5 file
    File        = h5.File(loc ,'r')
    if verbose: print(File.keys(), File[rate_key].keys())
        
    ################################################
    ## Read merger rate related data
    DCO_mask                  = File[rate_key]['DCOmask'][()] # Mask from DCO to merging BBH 
    redshifts                 = File[rate_key]['redshifts'][()]
    Average_SF_mass_needed    = File[rate_key]['Average_SF_mass_needed'][()]
    intrinsic_rate_density    = File[rate_key]['merger_rate'][()]
    intrinsic_rate_density_z0 = File[rate_key]['merger_rate_z0'][()] #Rate density at z=0 for the smallest z bin in your simulation


    ################################################
    ## Get DCO info and pour into astropy table
    DCO         = hdf5_to_astropy(File, group = 'DoubleCompactObjects' )  
    #SYS         = hdf5_to_astropy(File, group = 'SystemParameters' )  
    ################################################
    ## Add extra columns of interest to your DCO ##
    SYS_DCO_seeds_bool          = np.in1d(File['SystemParameters']['SEED'][()], DCO['SEED']) #Bool to point SYS to DCO
    q_init                      = File['SystemParameters']['Mass@ZAMS(2)'][()]/File['SystemParameters']['Mass@ZAMS(1)'][()]
    DCO['SemiMajorAxis@ZAMS']   = File['SystemParameters']['SemiMajorAxis@ZAMS'][SYS_DCO_seeds_bool]
    DCO['Mass@ZAMS(1)']         = File['SystemParameters']['Mass@ZAMS(1)'][SYS_DCO_seeds_bool]
    DCO['Mass@ZAMS(2)']         = File['SystemParameters']['Mass@ZAMS(2)'][SYS_DCO_seeds_bool]
    
    DCO['q_init']               = q_init[SYS_DCO_seeds_bool]
    DCO['Stellar_Type@ZAMS(1)'] = File['SystemParameters']['Stellar_Type@ZAMS(1)'][SYS_DCO_seeds_bool]
    DCO['Stellar_Type@ZAMS(2)'] = File['SystemParameters']['Stellar_Type@ZAMS(2)'][SYS_DCO_seeds_bool]
    DCO['tDelay']               = DCO['Coalescence_Time'] + DCO['Time'] #Myr
    DCO['M_moreMassive']        = np.maximum(DCO['Mass(1)'], DCO['Mass(2)'])
    DCO['M_lessMassive']        = np.minimum(DCO['Mass(1)'], DCO['Mass(2)'])
    DCO['q_final']              = DCO['M_lessMassive']/DCO['M_moreMassive']
    ################################################
    
    File.close()
    
    return DCO, DCO_mask, redshifts, Average_SF_mass_needed, intrinsic_rate_density, intrinsic_rate_density_z0 

    
