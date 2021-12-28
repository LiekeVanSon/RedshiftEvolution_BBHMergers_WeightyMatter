'''
Collection of small functions that are useful when reducing and plotting data. 
'''

######################################
## Imports
import numpy as np
import h5py as h5
from astropy.table import Table

import astropy.units as u

# Chosen cosmology 
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
import sys 

######################################
## locations
save_loc    =  '../plots/'
data_dir    = '../output/'


#########################################
# Class to show PDF images
#########################################
class PDF(object):
  def __init__(self, pdf, size=(200,200)):
    self.pdf = pdf
    self.size = size

  def _repr_html_(self):
    return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

  def _repr_latex_(self):
    return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)


#########################################
# Chirp mass
#########################################
def Mchirp(m1, m2):
    chirp_mass = np.divide(np.power(np.multiply(m1, m2), 3./5.), np.power(np.add(m1, m2), 1./5.))
    return chirp_mass    
   

#########################################
# read all groups and subgroups 
# from hdf5 file, and put in astropytable
#########################################
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
# Nice little progressbar script 
# to know how far you are with bootstrapping
#########################################
def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()

#########################################
# Read data
#########################################
def read_data(loc = data_dir+'/output/COMPAS_Output_wWeights.h5', rate_key = 'Rates_mu00.025_muz-0.05_alpha-1.77_sigma01.125_sigmaz0.05_zBinned', verbose=False):
    """
        Read DCO, SYS and merger rate data, necesarry to make the plots in this 
        
        Args:
            loc                       --> [string] Location of data
            rate_key                  --> [string] group key name of COMPAS HDF5 data that contains your merger rate
            verbose                   --> [bool] If you want to print statements while reading in 

        Returns:
            DCO                       --> [astropy table obj.]
            DCO_mask                  --> [list of bools]
            redshifts                 --> [list of floats]
            Average_SF_mass_needed    --> [float] 
            intrinsic_rate_density    --> [2D array of floats]
            intrinsic_rate_density_z0 --> [1D array of floats] 
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





#########################################
# Bin rate density over crude z-bin
#########################################
def get_crude_rate_density(intrinsic_rate_density, fine_redshifts, crude_redshifts):
    """
        A function to take the 'volume averaged' intrinsic rate density for large (crude) redshift bins. 
        This takes into account the change in volume for different redshift shells

        !! This function assumes an integrer number of fine redshifts fit in a crude redshiftbin !!
        !! We also assume the fine redshift bins and crude redshift bins are spaced equally in redshift !!
        
        Args:
            intrinsic_rate_density    --> [2D float array] Intrinsic merger rate density for each binary at each redshift in 1/yr/Gpc^3
            fine_redshifts            --> [list of floats] Edges of redshift bins at which the rates where evaluated
            crude_redshifts           --> [list of floats] Merger rate for each binary at each redshift in 1/yr/Gpc^3

        Returns:
            crude_rate_density       --> [2D float array] Intrinsic merger rate density for each binary at new crude redshiftbins in 1/yr/Gpc^3

    """
    # Calculate the volume of the fine redshift bins
    fine_volumes       = cosmo.comoving_volume(fine_redshifts).to(u.Gpc**3).value
    fine_shell_volumes = np.diff(fine_volumes) #same len in z dimension as weight

    # Multiply intrinsic rate density by volume of the redshift shells, to get the number of merging BBHs in each z-bin
    N_BBH_in_z_bin         = (intrinsic_rate_density[:,:] * fine_shell_volumes[:])
    
    # !! the following asusmes your redshift bins are equally spaced in both cases!!
    # get the binsize of 
    fine_binsize, crude_binsize    = np.diff(fine_redshifts), np.diff(crude_redshifts) 
    if np.logical_and(np.all(np.round(fine_binsize,8) == fine_binsize[0]),  np.all(np.round(crude_binsize,8) == crude_binsize[0]) ):
        fine_binsize    = fine_binsize[0]
        crude_binsize   = crude_binsize[0] 
    else:
        print('Your fine redshifts or crude redshifts are not equally spaced!,',
              'fine_binsize:', fine_binsize, 'crude_binsize', crude_binsize)
        return -1

    # !! also check that your crude redshift bin is made up of an integer number of fine z-bins !!
    i_per_crude_bin = crude_binsize/fine_binsize 
    print('i_per_crude_bin', i_per_crude_bin)
    if (i_per_crude_bin).is_integer():
        i_per_crude_bin = int(i_per_crude_bin)
    else: 
        print('your crude redshift bin is NOT made up of an integer number of fine z-bins!: i_per_crude_bin,', i_per_crude_bin)
        return -1
    
    
    # add every i_per_crude_bin-th element together, to get the number of merging BBHs in each crude redshift bin
    N_BBH_in_crudez_bin    = np.add.reduceat(N_BBH_in_z_bin, np.arange(0, len(N_BBH_in_z_bin[0,:]), int(i_per_crude_bin) ), axis = 1)
    
    
    # Convert crude redshift bins to volumnes and ensure all volumes are in Gpc^3
    crude_volumes       = cosmo.comoving_volume(crude_redshifts).to(u.Gpc**3).value
    crude_shell_volumes = np.diff(crude_volumes)# split volumes into shells 
    
    
    # Finally tunr rate back into an average (crude) rate density, by dividing by the new z-volumes
    # In case your crude redshifts don't go all the way to z_first_SF, just use N_BBH_in_crudez_bin up to len(crude_shell_volumes)
    crude_rate_density     = N_BBH_in_crudez_bin[:, :len(crude_shell_volumes)]/crude_shell_volumes
    
    return crude_rate_density



