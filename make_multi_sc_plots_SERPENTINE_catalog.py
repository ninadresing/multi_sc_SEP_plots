import datetime as dt
import os
import warnings

import cdflib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy
from matplotlib.ticker import AutoMinorLocator
from soho_loader import calc_av_en_flux_ERNE, soho_load
from solo_epd_loader import epd_load
from stereo_loader import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from stereo_loader import calc_av_en_flux_SEPT, stereo_load
# import astropy.units as u
# from cdflib.epochs import CDFepoch
# from sunpy import log
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
# from sunpy.util.exceptions import warn_user
from tqdm import tqdm
from wind_3dp_loader import wind3dp_load

'''
June 2022
this loops over the mission time and creates
multi-sc plots to be used for the SERPENTINE
multi-sc SEP event catalog
moved this file to Deepnote June 15 2022
'''

# make selections
#############################################################
first_date = dt.datetime(2021, 9, 17)
last_date = dt.datetime(2021, 9, 18)
plot_period = '7D'
averaging = '1H'  # None

Bepi = False  # not included yet!
PSP = True
SOHO = True
SOLO = True
STEREO = True
WIND = True


# SOHO:
erne = True
ephin_p = False  # not included yet!
ephin_e = False  # not included yet!

# SOLO:
ept = True
het = True
ept_use_corr_e = False  # not included yet!

# STEREO:
sept_e = True
sept_p = False
stereo_het = True
let = False

wind3dp_p = False
wind3dp_e = True
#############################################################

# omit some warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=sunpy.util.SunpyUserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# manually define seaborn-colorblind colors
seaborn_colorblind = ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']  # blue, green, orange, magenta, yello, light blue
# change some matplotlib plotting settings
SIZE = 20
plt.rc('font', size=SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['xtick.major.width'] = 2
plt.rcParams['xtick.minor.size'] = 5
plt.rcParams['xtick.minor.width'] = 1
plt.rcParams['ytick.major.size'] = 10
plt.rcParams['ytick.major.width'] = 2
plt.rcParams['ytick.minor.size'] = 5
plt.rcParams['ytick.minor.width'] = 1
plt.rcParams['axes.linewidth'] = 2.0


def _fillval_nan(data, fillval):
    try:
        data[data == fillval] = np.nan
    except ValueError:
        # This happens if we try and assign a NaN to an int type
        pass
    return data


def _get_cdf_vars(cdf):
    # Get list of all the variables in an open CDF file
    var_list = []
    cdf_info = cdf.cdf_info()
    for attr in list(cdf_info.keys()):
        if 'variable' in attr.lower() and len(cdf_info[attr]) > 0:
            for var in cdf_info[attr]:
                var_list += [var]

    return var_list


def _cdf2df_3d_psp(cdf, index_key, dtimeindex=True, ignore=None, include=None):
    """
    Converts a cdf file to a pandas dataframe.
    Note that this only works for 1 dimensional data, other data such as
    distribution functions or pitch angles will not work properly.
    Parameters
    ----------
    cdf : cdf
        Opened CDF file.
    index_key : str
        The CDF key to use as the index in the output DataFrame.
    dtimeindex : bool
        If ``True``, the DataFrame index is parsed as a datetime.
        Default is ``True``.
    ignore : list
        In case a CDF file has columns that are unused / not required, then
        the column names can be passed as a list into the function.
    include : str, list
        If only specific columns of a CDF file are desired, then the column
        names can be passed as a list into the function. Should not be used
        with ``ignore``.
    Returns
    -------
    df : :class:`pandas.DataFrame`
        Data frame with read in data.
    """
    if include is not None:
        if ignore is not None:
            raise ValueError('ignore and include are incompatible keywords')
        if isinstance(include, str):
            include = [include]
        if index_key not in include:
            include.append(index_key)

    # Extract index values
    index_info = cdf.varinq(index_key)
    if index_info['Last_Rec'] == -1:
        warnings.warn(f"No records present in CDF file {cdf.cdf_info()['CDF'].name}")
        return_df = pd.DataFrame()
    else:
        index = cdf.varget(index_key)
        try:
            # If there are multiple indexes, take the first one
            # TODO: this is just plain wrong, there should be a way to get all
            # the indexes out
            index = index[...][:, 0]
        except IndexError:
            pass

        if dtimeindex:
            index = cdflib.epochs.CDFepoch.breakdown(index, to_np=True)
            index_df = pd.DataFrame({'year': index[:, 0],
                                    'month': index[:, 1],
                                    'day': index[:, 2],
                                    'hour': index[:, 3],
                                    'minute': index[:, 4],
                                    'second': index[:, 5],
                                    'ms': index[:, 6],
                                    })
            # Not all CDFs store pass milliseconds
            try:
                index_df['us'] = index[:, 7]
                index_df['ns'] = index[:, 8]
            except IndexError:
                pass
            index = pd.DatetimeIndex(pd.to_datetime(index_df), name='Time')
        data_dict = {}
        npoints = len(index)

        var_list = _get_cdf_vars(cdf)
        keys = {}
        # Get mapping from each attr to sub-variables
        for cdf_key in var_list:
            if ignore:
                if cdf_key in ignore:
                    continue
            elif include:
                if cdf_key not in include:
                    continue
            if cdf_key == 'Epoch':
                keys[cdf_key] = 'Time'
            else:
                keys[cdf_key] = cdf_key
        # Remove index key, as we have already used it to create the index
        keys.pop(index_key)
        # Remove keys for data that doesn't have the right shape to load in CDF
        # Mapping of keys to variable data
        vars = {}
        for cdf_key in keys.copy():
            try:
                vars[cdf_key] = cdf.varget(cdf_key)
            except ValueError:
                vars[cdf_key] = ''
        for cdf_key in keys:
            var = vars[cdf_key]
            if type(var) is np.ndarray:
                key_shape = var.shape
                if len(key_shape) == 0 or key_shape[0] != npoints:
                    vars.pop(cdf_key)
            else:
                vars.pop(cdf_key)

        # Loop through each key and put data into the dataframe
        for cdf_key in vars:
            df_key = keys[cdf_key]
            # Get fill value for this key
            # First catch string FILLVAL's
            if type(cdf.varattsget(cdf_key)['FILLVAL']) is str:
                fillval = cdf.varattsget(cdf_key)['FILLVAL']
            else:
                try:
                    fillval = float(cdf.varattsget(cdf_key)['FILLVAL'])
                except KeyError:
                    fillval = np.nan

            if isinstance(df_key, list):
                for i, subkey in enumerate(df_key):
                    data = vars[cdf_key][...][:, i]
                    data = _fillval_nan(data, fillval)
                    data_dict[subkey] = data
            else:
                # If ndims is 1, we just have a single column of data
                # If ndims is 2, have multiple columns of data under same key
                # If ndims is 3, have multiple columns of data under same key, with 2 sub_keys (e.g., energy and pitch-angle)
                key_shape = vars[cdf_key].shape
                ndims = len(key_shape)
                if ndims == 1:
                    data = vars[cdf_key][...]
                    data = _fillval_nan(data, fillval)
                    data_dict[df_key] = data
                elif ndims == 2:
                    for i in range(key_shape[1]):
                        data = vars[cdf_key][...][:, i]
                        data = _fillval_nan(data, fillval)
                        data_dict[f'{df_key}_{i}'] = data
                elif ndims == 3:
                    for i in range(key_shape[2]):
                        for j in range(key_shape[1]):
                            data = vars[cdf_key][...][:, j, i]
                            data = _fillval_nan(data, fillval)
                            data_dict[f'{df_key}_E{i}_P{j}'] = data
        return_df = pd.DataFrame(index=index, data=data_dict)

    return return_df


def psp_isois_load(dataset, startdate, enddate, epilo_channel='F', epilo_threshold=None, path=None, resample=None):
    """
    Downloads CDF files via SunPy/Fido from CDAWeb for CELIAS, EPHIN, ERNE onboard SOHO
    Parameters
    ----------
    dataset : {str}
        Name of PSP dataset:
            - 'PSP_ISOIS-EPIHI_L2-HET-RATES60'
            - 'PSP_ISOIS-EPIHI_L2-HET-RATES3600' (higher coverage than 'RATES60' before mid-2021)
            - 'PSP_ISOIS-EPIHI_L2-LET1-RATES60' (not yet supported)
            - 'PSP_ISOIS-EPIHI_L2-LET2-RATES60' (not yet supported)
            - 'PSP_ISOIS-EPILO_L2-PE'
            - 'PSP_ISOIS-EPILO_L2-IC' (not yet supported)
    startdate, enddate : {datetime or str}
        Datetime object (e.g., dt.date(2021,12,31) or dt.datetime(2021,4,15)) or "standard"
        datetime string (e.g., "2021/04/15") (enddate must always be later than startdate)
    epilo_channel : string
        'E', 'F', 'G'. EPILO chan, by default 'F'
    epilo_threshold : {int or float}, optional
        Replace ALL flux/countrate values above 'epilo_threshold' with np.nan, by default None
    path : {str}, optional
        Local path for storing downloaded data, by default None
    resample : {str}, optional
        resample frequency in format understandable by Pandas, e.g. '1min', by default None
    Returns
    -------
    df : {Pandas dataframe}
        See links above for the different datasets for a description of the dataframe columns
    energies_dict : {dictionary}
        Dictionary containing energy information.
        NOTE: For EPIHI energy values are only loaded from the first day of the interval! 
        For EPILO energy values are the mean of the whole loaded interval.
    """
    trange = a.Time(startdate, enddate)
    cda_dataset = a.cdaweb.Dataset(dataset)
    try:
        result = Fido.search(trange, cda_dataset)
        filelist = [i[0].split('/')[-1] for i in result.show('URL')[0]]
        filelist.sort()
        if path is None:
            filelist = [sunpy.config.get('downloads', 'download_dir') + os.sep + file for file in filelist]
        elif type(path) is str:
            filelist = [path + os.sep + f for f in filelist]
        downloaded_files = filelist

        for i, f in enumerate(filelist):
            if os.path.exists(f) and os.path.getsize(f) == 0:
                os.remove(f)
            if not os.path.exists(f):
                downloaded_file = Fido.fetch(result[0][i], path=path, max_conn=1)

        # loading for EPIHI
        if dataset.split('-')[1] == 'EPIHI_L2':
            # downloaded_files = Fido.fetch(result, path=path, max_conn=1)
            # downloaded_files.sort()
            data = TimeSeries(downloaded_files, concatenate=True)
            df = data.to_dataframe()
            # df = read_cdf(downloaded_files[0])

            # reduce data frame to only H_Flux, H_Uncertainty, Electron_Counts, and Electron_Rate.
            # There is no Electron_Uncertainty, maybe one could use at least the Poission error from Electron_Counts for that.
            # df = df.filter(like='H_Flux') + df.filter(like='H_Uncertainty') + df.filter(like='Electrons')
            if dataset.split('-')[2].upper() == 'HET':
                if dataset.split('-')[3] == 'RATES60':
                    selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
                if dataset.split('-')[3] == 'RATES3600':
                    selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            if dataset.split('-')[2].upper() == 'LET1':
                selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            if dataset.split('-')[2].upper() == 'LET2':
                selected_cols = ["A_H_Flux", "B_H_Flux", "A_H_Uncertainty", "B_H_Uncertainty", "A_Electrons", "B_Electrons"]
            df = df[df.columns[df.columns.str.startswith(tuple(selected_cols))]]

            cdf = cdflib.CDF(downloaded_files[0])

            # remove this (i.e. following line) when sunpy's read_cdf is updated,
            # and FILLVAL will be replaced directly, see
            # https://github.com/sunpy/sunpy/issues/5908
            df = df.replace(cdf.varattsget('A_H_Flux')['FILLVAL'], np.nan)

            # get info on energies and units
            energies_dict = {"H_ENERGY":
                             cdf['H_ENERGY'],
                             "H_ENERGY_DELTAPLUS":
                             cdf['H_ENERGY_DELTAPLUS'],
                             "H_ENERGY_DELTAMINUS":
                             cdf['H_ENERGY_DELTAMINUS'],
                             "H_ENERGY_LABL":
                             cdf['H_ENERGY_LABL'],
                             "H_FLUX_UNITS":
                             cdf.varattsget('A_H_Flux')['UNITS'],
                             "Electrons_ENERGY":
                             cdf['Electrons_ENERGY'],
                             "Electrons_ENERGY_DELTAPLUS":
                             cdf['Electrons_ENERGY_DELTAPLUS'],
                             "Electrons_ENERGY_DELTAMINUS":
                             cdf['Electrons_ENERGY_DELTAMINUS'],
                             "Electrons_ENERGY_LABL":
                             cdf['Electrons_ENERGY_LABL'],
                             "Electrons_Rate_UNITS":
                             cdf.varattsget('A_Electrons_Rate')['UNITS']
                             }

        # loading for EPILO
        if dataset.split('-')[1] == 'EPILO_L2':
            if len(downloaded_files) > 0:
                ignore = ['Epoch_ChanF_DELTA', 'RTN_ChanF', 'HCI_ChanF', 'HCI_R_ChanF', 'HCI_Lat_ChanF', 'HCI_Lon_ChanF', 'HGC_R_ChanF', 'HGC_Lat_ChanF', 'HGC_Lon_ChanF', 'Electron_ChanF_Energy_LABL', 'Electron_Counts_ChanF']
                # read 0th cdf file
                cdf = cdflib.CDF(downloaded_files[0])
                df = _cdf2df_3d_psp(cdf, f"Epoch_Chan{epilo_channel.upper()}", ignore=ignore)

                # read additional cdf files
                if len(downloaded_files) > 1:
                    for f in downloaded_files[1:]:
                        cdf = cdflib.CDF(f)
                        t_df = _cdf2df_3d_psp(cdf, f"Epoch_Chan{epilo_channel.upper()}", ignore=ignore)
                        df = pd.concat([df, t_df])

                # columns of returned df for EPILO PE
                # -----------------------------------
                # PA_ChanF_0 to PA_ChanF_7
                # SA_ChanF_0 to SA_ChanF_7
                # Electron_ChanF_Energy_E0_P0 to Electron_ChanF_Energy_E47_P7
                # Electron_ChanF_Energy_DELTAMINUS_E0_P0 to Electron_ChanF_Energy_DELTAMINUS_E47_P7
                # Electron_ChanF_Energy_DELTAPLUS_E0_P0 to Electron_ChanF_Energy_DELTAPLUS_E47_P7
                # Electron_CountRate_ChanF_E0_P0 to Electron_CountRate_ChanF_E47_P7
                energies_dict = {}
                for k in [f'Electron_Chan{epilo_channel.upper()}_Energy_E',
                          f'Electron_Chan{epilo_channel.upper()}_Energy_DELTAMINUS',
                          f'Electron_Chan{epilo_channel.upper()}_Energy_DELTAPLUS']:
                    energies_dict[k] = df[df.columns[df.columns.str.startswith(k)]].mean()
                    df.drop(df.columns[df.columns.str.startswith(k)], axis=1, inplace=True)
                # rename energy column (removing trailing '_E')
                energies_dict[f'Electron_Chan{epilo_channel.upper()}_Energy'] = energies_dict.pop(f'Electron_Chan{epilo_channel.upper()}_Energy_E')

                # replace outlier data points above given threshold with np.nan
                # note: df.where(cond, np.nan) replaces all values where the cond is NOT fullfilled with np.nan
                # following Pandas Dataframe work is not too elegant, but works...
                if epilo_threshold:
                    # create new dataframe of FLUX columns only with removed outliers
                    df2 = df.filter(like='Electron_CountRate_').where(df.filter(like='Electron_CountRate_') <= epilo_threshold, np.nan)
                    # drop these FLUX columns from original dataframe
                    flux_cols = df.filter(like='Electron_CountRate_').columns
                    df.drop(labels=flux_cols, axis=1, inplace=True)
                    # add cleaned new FLUX columns to original dataframe
                    df = pd.concat([df2, df], axis=1)
            else:
                df = ''
                energies_dict = ''

        if isinstance(resample, str):
            df = resample_df(df, resample)
    except RuntimeError:
        print(f'Unable to obtain "{dataset}" data!')
        downloaded_files = []
        df = pd.DataFrame()
        energies_dict = []
    return df, energies_dict


def resample_df(df, resample):
    """
    Resample Pandas Dataframe
    """
    try:
        # _ = pd.Timedelta(resample)  # test if resample is proper Pandas frequency
        df = df.resample(resample).mean()
        df.index = df.index + pd.tseries.frequencies.to_offset(pd.Timedelta(resample)/2)
    except ValueError:
        raise Warning(f"Your 'resample' option of [{resample}] doesn't seem to be a proper Pandas frequency!")
    return df


def calc_av_en_flux_PSP_EPIHI(df, energies, en_channel, species, instrument, viewing):
    """
    This function averages the flux of several energy channels into a combined energy channel
    channel numbers counted from 0

    So far only works for EPIHI-HET

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing PSP data
    energies : dict
        Energy dict returned from psp_loader
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'het'
    viewing : string
        'A', 'B'psp

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux
    """
    if instrument.lower() == 'het':
        if species.lower() in ['e', 'electrons']:
            species_str = 'Electrons'
            flux_key = 'Electrons_Rate'
        if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
            species_str = 'H'
            flux_key = 'H_Flux'
    if type(en_channel) == list:
        en_str = energies[f'{species_str}_ENERGY_LABL']
        energy_low = en_str[en_channel[0]][0].split('-')[0]
        energy_up = en_str[en_channel[-1]][0].split('-')[-1]
        en_channel_string = energy_low + '-' + energy_up
        # replace multiple whitespaces with single ones
        en_channel_string = ' '.join(en_channel_string.split())

        DE = energies[f'{species_str}_ENERGY_DELTAPLUS']+energies[f'{species_str}_ENERGY_DELTAMINUS']

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            try:
                df = df[df.columns[df.columns.str.startswith(f'{viewing.upper()}_{flux_key}')]]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{viewing.upper()}_{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{viewing.upper()}_{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[f'{viewing.upper()}_{flux_key}_{en_channel}']}, index=df.index)
    else:
        flux_out = pd.DataFrame({'flux': df[f'{viewing.upper()}_{flux_key}_{en_channel}']}, index=df.index)
        en_channel_string = en_str[en_channel]
    return flux_out, en_channel_string


def calc_av_en_flux_PSP_EPILO(df, en_dict, en_channel, species, mode, chan, viewing):
    """
    This function averages the flux of several energy channels into a combined energy channel
    channel numbers counted from 0

    So far only works for EPILO PE chanF electrons

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing PSP data
    energies : dict
        Energy dict returned from psp_loader
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons'
    mode : string
        'pe'. EPILO mode
    chan : string
        'E', 'F', 'G'. EPILO chan
    viewing : integer
        EPILO viewing. 0 to 7 for 'E' & 'F'; 80 for 'G'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux
    """

    if mode.lower() == 'pe':
        if species.lower() in ['e', 'electrons']:
            species_str = 'Electron'
            flux_key = 'Electron_CountRate'
        # if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        #     species_str = 'H'
        #     flux_key = 'H_Flux'
    # if mode.lower() == 'ic':
        # if species.lower() in ['e', 'electrons']:
        #     species_str = 'Electrons'
        #     flux_key = 'Electrons_Rate'
        # if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        #     species_str = 'H'
        #     flux_key = 'H_Flux'
        if type(en_channel) == int:
            en_channel = [en_channel]
        if type(en_channel) == list:
            energy = en_dict[f'{species_str}_Chan{chan}_Energy'].filter(like=f'_P{viewing}').values
            energy_low = energy - en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'].filter(like=f'_P{viewing}').values
            energy_high = energy + en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'].filter(like=f'_P{viewing}').values
            DE = en_dict[f'{species_str}_Chan{chan}_Energy_DELTAMINUS'].filter(like=f'_P{viewing}').values + en_dict[f'{species_str}_Chan{chan}_Energy_DELTAPLUS'].filter(like=f'_P{viewing}').values

            # build energy string of combined channel
            en_channel_string = np.round(energy_low[en_channel[0]],1).astype(str) + ' - ' + np.round(energy_high[en_channel[-1]],1).astype(str) + ' keV'

            # select viewing direction
            # df = df.filter(like=f'_P{viewing}')

            if len(en_channel) > 2:
                raise Exception("en_channel must have length 2 or less! Define first and last channel to use (don't list all of them)")
            if len(en_channel) == 2:
                # try:
                #     df = df[df.columns[df.columns.str.startswith(f'{viewing.upper()}_{flux_key}')]]
                #     # df = df[df.columns[df.columns.str.startswith(f'{flux_key}_Chan{chan}_')]]
                # except (AttributeError, KeyError):
                #     None
                for bins in np.arange(en_channel[0], en_channel[-1]+1):
                    if bins == en_channel[0]:
                        I_all = df[f"{flux_key}_Chan{chan}_E{bins}_P{viewing}"] * DE[bins]
                    else:
                        I_all = I_all + df[f"{flux_key}_Chan{chan}_E{bins}_P{viewing}"] * DE[bins]
                DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
                flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
            if len(en_channel) == 1:
                en_channel = en_channel[0]
                flux_out = pd.DataFrame({'flux': df[f"{flux_key}_Chan{chan}_E{en_channel}_P{viewing}"]}, index=df.index)
    return flux_out, en_channel_string


def calc_av_en_flux_EPD(df, energies, en_channel, species, instrument):  # original from Nina Slack Feb 9, 2022, rewritten Jan Apr 8, 2022
    """This function averages the flux of several energy channels of HET into a combined energy channel
    channel numbers counted from 0

    Parameters
    ----------
    df : pd.DataFrame DataFrame containing HET data
        DataFrame containing HET data
    energies : dict
        Energy dict returned from epd_loader (from Jan)
    en_channel : int or list
        energy channel number(s) to be used
    species : string
        'e', 'electrons', 'p', 'i', 'protons', 'ions'
    instrument : string
        'ept' or 'het'

    Returns
    -------
    pd.DataFrame
        flux_out: contains channel-averaged flux

    Raises
    ------
    Exception
        [description]
    """
    if species.lower() in ['e', 'electrons']:
        en_str = energies['Electron_Bins_Text']
        bins_width = 'Electron_Bins_Width'
        flux_key = 'Electron_Flux'
    if species.lower() in ['p', 'protons', 'i', 'ions', 'h']:
        if instrument.lower() == 'het':
            en_str = energies['H_Bins_Text']
            bins_width = 'H_Bins_Width'
            flux_key = 'H_Flux'
        if instrument.lower() == 'ept':
            en_str = energies['Ion_Bins_Text']
            bins_width = 'Ion_Bins_Width'
            flux_key = 'Ion_Flux'
    if type(en_channel) == list:
        energy_low = en_str[en_channel[0]][0].split('-')[0]

        energy_up = en_str[en_channel[-1]][0].split('-')[-1]

        en_channel_string = energy_low + '-' + energy_up

        if len(en_channel) > 2:
            raise Exception('en_channel must have len 2 or less!')
        if len(en_channel) == 2:
            DE = energies[bins_width]
            try:
                df = df[flux_key]
            except (AttributeError, KeyError):
                None
            for bins in np.arange(en_channel[0], en_channel[-1]+1):
                if bins == en_channel[0]:
                    I_all = df[f'{flux_key}_{bins}'] * DE[bins]
                else:
                    I_all = I_all + df[f'{flux_key}_{bins}'] * DE[bins]
            DE_total = np.sum(DE[(en_channel[0]):(en_channel[-1]+1)])
            flux_out = pd.DataFrame({'flux': I_all/DE_total}, index=df.index)
        else:
            en_channel = en_channel[0]
            flux_out = pd.DataFrame({'flux': df[f'{flux_key}_{en_channel}']}, index=df.index)
    else:
        flux_out = pd.DataFrame({'flux': df[f'{flux_key}_{en_channel}']}, index=df.index)
        en_channel_string = en_str[en_channel]
    return flux_out, en_channel_string


# some plot options
intensity_label = 'Flux\n/(s cm² sr MeV)'
linewidth = 1.5
outpath = None  # os.getcwd()
plot_e_100 = True
plot_e_1 = True
plot_p = True
save_fig = True
outpath = 'plots'  # '/Users/dresing/Documents/Proposals/SERPENTINE_H2020/Cycle25_Multi-SC_SEP_Event_List/Multi_sc_plots'

dates = pd.date_range(start=first_date, end=last_date, freq=plot_period)
for startdate in tqdm(dates.to_pydatetime()):
    enddate = startdate + pd.Timedelta(plot_period)
    outfile = f'{outpath}{os.sep}Multi_sc_plot_{startdate.date()}_{plot_period}_{averaging}-av.png'

    if Bepi:
        # av_bepi = 10
        sixs_resample = averaging  # '10min'
        # 'E1':'71 keV', 'E2':'106 keV', 'E3':'169 keV', 'E4':'280 keV', 'E5':'960 keV', 'E6':'2240 keV', 'E7':'8170 keV'
        sixs_ch_e1 = 'E5'  # 'E2' #
        sixs_ch_e100 = 'E2'
        sixs_side_e = 2
        sixs_color = seaborn_colorblind[4]
        # 'P1':'1.1 MeV', 'P2':'1.2 MeV', 'P3':'1.5 MeV', 'P4':'2.3 MeV', 'P5':'4.0 MeV', 'P6':'8.0 MeV', 'P7':'15.0 MeV','P8':'25.1 MeV', 'P9':'37.3 MeV'
        sixs_ch_p = ['P8', 'P9']  # we want 'P8'-'P9' averaged (to do!)
        sixs_side_p = 3
    if SOHO:
        soho_ephin_color = 'k'
        soho_erne_color = 'k'  # seaborn_colorblind[5]  # 'green'
        # av_soho = av
        soho_erne_resample = averaging  # '30min'
        soho_path = '/home/gieseler/uni/soho/data/'
        if erne:
            erne_p_ch = [3, 4]  # [0]  # [4,5]  # 2
        if ephin_e:
            ephin_ch_e1 = 'e150'
            ephin_e_intercal = 1/14.
        if ephin_p:
            ephin_ch_p = 'p25'
    if SOLO:
        solo_ept_color = seaborn_colorblind[5]  # 'blue'
        solo_het_color = seaborn_colorblind[0]  # 'blue' # seaborn_colorblind[1]
        sector = 'sun'
        ept_ch_e100 = [14, 18]  # [25]
        het_ch_e1 = [0, 1]
        ept_ch_p = [50, 56]  # 50-56
        het_ch_p = [19, 24]  # [18, 19]
        solo_ept_resample = averaging
        solo_het_resample = averaging
        solo_path = '/home/gieseler/uni/solo/data/'
    if STEREO:
        stereo_sept_color = 'orangered'  # seaborn_colorblind[3]  #
        stereo_het_color = 'orangered'  # seaborn_colorblind[3]  # 'coral'
        stereo_let_color = 'orangered'  # seaborn_colorblind[3]  # 'coral'
        sector = 'sun'
        sept_ch_e100 = [6, 7]  # [12, 16]
        sept_ch_p = [25, 30]
        st_het_ch_e = [0, 1]
        st_het_ch_p = [5, 8]  # 3  #7 #3
        let_ch = 5  # 1
        sta_het_resample = averaging
        sta_sept_resample = averaging
        sta_let_resample = averaging
        stereo_path = '/home/gieseler/uni/stereo/data/'
    if WIND:
        wind_color = 'dimgrey'
        wind3dp_ch_e100 = 3
        wind3dp_ch_p = 6
        wind_3dp_resample = averaging  # '30min'
        wind_3dp_threshold = 1e3/1e6  # None
        wind_path = '/home/gieseler/uni/wind/data/'
    if PSP:
        psp_epilo_ch_e100 = [4, 5]  # cf. psp_epilo_energies
        psp_het_ch_e = [3, 10]  # cf. psp_het_energies
        psp_het_ch_p = [8, 9]  # cf. psp_het_energies

        psp_epilo_channel = 'F'
        psp_epilo_viewing = 3  # 3="sun", 7="antisun"
        psp_epilo_threshold = None  # None
        psp_path = '/home/gieseler/uni/psp/data/'
        psp_het_resample = averaging
        psp_epilo_resample = averaging
        psp_het_color = 'blueviolet'

    # LOAD DATA
    ##################################################################

    if WIND:
        if wind3dp_e:
            print('loading wind/3dp e')
            wind3dp_e_df, wind3dp_e_meta = wind3dp_load(dataset="WI_SFSP_3DP",
                                                        startdate=startdate,
                                                        enddate=enddate,
                                                        resample=wind_3dp_resample,
                                                        threshold=wind_3dp_threshold,
                                                        multi_index=False,
                                                        path=wind_path,
                                                        max_conn=1)
            wind3dp_ch_e = wind3dp_ch_e100

        if wind3dp_p:
            print('loading wind/3dp p')
            wind3dp_p_df, wind3dp_p_meta = wind3dp_load(dataset="WI_SOSP_3DP", startdate=startdate, enddate=enddate, resample=wind_3dp_resample, multi_index=False, path=wind_path, max_conn=1)

    if STEREO:
        if stereo_het:
            print('loading stereo/het')
            sta_het_e_labels = ['0.7-1.4 MeV', '1.4-2.8 MeV', '2.8-4.0 MeV']
            sta_het_p_labels = ['13.6-15.1 MeV', '14.9-17.1 MeV', '17.0-19.3 MeV', '20.8-23.8 MeV', '23.8-26.4 MeV', '26.3-29.7 MeV', '29.5-33.4 MeV', '33.4-35.8 MeV', '35.5-40.5 MeV', '40.0-60.0 MeV']

            sta_het_df, sta_het_meta = stereo_load(instrument='het', startdate=startdate, enddate=enddate, spacecraft='sta', resample=sta_het_resample, path=stereo_path, max_conn=1)

        if let:
            print('loading stereo/let')
            # for H and He4:
            let_chstring = ['1.8-2.2 MeV', '2.2-2.7 MeV', '2.7-3.2 MeV', '3.2-3.6 MeV', '3.6-4.0 MeV', '4.0-4.5 MeV', '4.5-5.0 MeV', '5.0-6.0 MeV', '6.0-8.0 MeV', '8.0-10.0 MeV', '10.0-12.0 MeV', '12.0-15.0 MeV']

            sta_let_df, sta_let_meta = stereo_load(instrument='let', startdate=startdate, enddate=enddate, spacecraft='sta', resample=sta_let_resample, path=stereo_path, max_conn=1)
        if sept_e:
            print('loading stereo/sept e')

            sta_sept_df_e, sta_sept_dict_e = stereo_load(instrument='sept', startdate=startdate, enddate=enddate, spacecraft='sta', sept_species='e', sept_viewing=sector, resample=sta_sept_resample, path=stereo_path, max_conn=1)
            sept_ch_e = sept_ch_e100

        if sept_p:
            print('loading stereo/sept p')

            sta_sept_df_p, sta_sept_dict_p = stereo_load(instrument='sept', startdate=startdate, enddate=enddate, spacecraft='sta', sept_species='p', sept_viewing=sector, resample=sta_sept_resample, path=stereo_path, max_conn=1)

        sectors = {'sun': 0, 'asun': 1, 'north': 2, 'south': 3}
        sector_num = sectors[sector]

    if SOHO:
        if ephin_e or ephin_p:
            print('loading soho/ephin')
            ephin = eph_rl2_loader(startdate.year, startdate.timetuple().tm_yday, doy2=enddate.timetuple().tm_yday, av=av_soho)
        if erne:
            print('loading soho/erne')
            erne_chstring = ['13-16 MeV', '16-20 MeV', '20-25 MeV', '25-32 MeV', '32-40 MeV', '40-50 MeV', '50-64 MeV', '64-80 MeV', '80-100 MeV', '100-130 MeV']
            # soho_p = ERNE_HED_loader(startdate.year, startdate.timetuple().tm_yday, doy2=enddate.timetuple().tm_yday, av=av_soho)
            soho_erne, erne_energies = soho_load(dataset="SOHO_ERNE-HED_L2-1MIN", startdate=startdate, enddate=enddate, path=soho_path, resample=soho_erne_resample, max_conn=1)

    if PSP:
        print('loading PSP/EPIHI-HET data')
        psp_het, psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES60', startdate, enddate, path=psp_path, resample=None)
        # psp_let1, psp_let1_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-LET1-RATES60', startdate, enddate, path=psp_path, resample=psp_resample)
        if len(psp_het) == 0:
            print(f'No PSP/EPIHI-HET 60s data found for {startdate.date()} - {enddate.date()}. Trying 3600s data.')
            psp_het, psp_het_energies = psp_isois_load('PSP_ISOIS-EPIHI_L2-HET-RATES3600', startdate, enddate, path=psp_path, resample=None)
            psp_3600 = True
            psp_het_resample = None

        print('loading PSP/EPILO PE data')
        psp_epilo, psp_epilo_energies = psp_isois_load('PSP_ISOIS-EPILO_L2-PE',
                                                       startdate, enddate,
                                                       epilo_channel=psp_epilo_channel,
                                                       epilo_threshold=psp_epilo_threshold,
                                                       path=psp_path, resample=None)
        if len(psp_epilo) == 0:
            print(f'No PSP/EPILO PE data for {startdate.date()} - {enddate.date()}')

    if SOLO:
        data_product = 'l2'
        sdate = startdate
        edate = enddate
        if ept:
            if plot_e_100 or plot_p:
                print('loading solo/ept e & p')
                try:
                    ept_p, ept_e, ept_energies = epd_load(sensor='EPT', viewing=sector, level=data_product, startdate=sdate, enddate=edate, path=solo_path, autodownload=True)
                except(Exception):
                    print(f'No SOLO/EPT data for {startdate.date()} - {enddate.date()}')
                    ept_e = []
        if het:
            if plot_e_1 or plot_p:
                print('loading solo/het e & p')
                try:
                    het_p, het_e, het_energies = epd_load(sensor='HET', viewing=sector, level=data_product, startdate=sdate, enddate=edate, path=solo_path, autodownload=True)
                except(Exception):
                    print(f'No SOLO/HET data for {startdate.date()} - {enddate.date()}')
                    het_e = []
                    het_p = []

    if Bepi:
        print('loading Bepi/SIXS')
        sixs_e, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_e, av=sixs_resample)
        sixs_p, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_p, av=sixs_resample)
        sixs_ch_e = sixs_ch_e100

########## AVERAGE ENERGY CHANNELS
####################################################
    if SOLO:
        if len(ept_e) > 0:
            if plot_e_100:
                df_ept_e = ept_e['Electron_Flux']
                ept_en_str_e = ept_energies['Electron_Bins_Text'][:]

                if ept_use_corr_e:
                    print('correcting e')
                    ion_cont_corr_matrix = np.loadtxt('EPT_ion_contamination_flux_paco.dat')
                    Electron_Flux_cont = np.zeros(np.shape(df_ept_e))
                    for tt in range(len(df_ept_e)):
                        Electron_Flux_cont[tt, :] = np.matmul(ion_cont_corr_matrix, df_ept_p.values[tt, :])
                    df_ept_e = df_ept_e - Electron_Flux_cont

                df_ept_e, ept_chstring_e = calc_av_en_flux_EPD(ept_e, ept_energies, ept_ch_e100, 'e', 'ept')

                if isinstance(solo_ept_resample, str):
                    df_ept_e = resample_df(df_ept_e, solo_ept_resample)
            if plot_p:
                df_ept_p = ept_p['Ion_Flux']
                ept_en_str_p = ept_energies['Ion_Bins_Text'][:]
                df_ept_p, ept_chstring_p = calc_av_en_flux_EPD(ept_p, ept_energies, ept_ch_p, 'p', 'ept')
                if isinstance(solo_ept_resample, str):
                    df_ept_p = resample_df(df_ept_p, solo_ept_resample)

        if len(het_e) > 0:
            if plot_e_1:
                print('calc_av_en_flux_HET e')
                df_het_e, het_chstring_e = calc_av_en_flux_EPD(het_e, het_energies, het_ch_e1, 'e', 'het')
                if isinstance(solo_het_resample, str):
                    df_het_e = resample_df(df_het_e, solo_het_resample)
            if plot_p:
                print('calc_av_en_flux_HET p')
                df_het_p, het_chstring_p = calc_av_en_flux_EPD(het_p, het_energies, het_ch_p, 'p', 'het')
                if isinstance(solo_het_resample, str):
                    df_het_p = resample_df(df_het_p, solo_het_resample)

    if STEREO:
        if sept_e:
            if type(sept_ch_e) == list and len(sta_sept_df_e) > 0:
                sta_sept_avg_e, sept_chstring_e = calc_av_en_flux_SEPT(sta_sept_df_e, sta_sept_dict_e, sept_ch_e)
            else:
                sta_sept_avg_e = []
                sept_chstring_e = ''

        if sept_p:
            if type(sept_ch_p) == list and len(sta_sept_df_p) > 0:
                sta_sept_avg_p, sept_chstring_p = calc_av_en_flux_SEPT(sta_sept_df_p, sta_sept_dict_p, sept_ch_p)
            else:
                sta_sept_avg_p = []
                sept_chstring_p = ''

        if stereo_het:
            if type(st_het_ch_e) == list and len(sta_het_df) > 0:
                sta_het_avg_e, st_het_chstring_e = calc_av_en_flux_ST_HET(sta_het_df.filter(like='Electron'),
                                                                          sta_het_meta['channels_dict_df_e'],
                                                                          st_het_ch_e, species='e')
            else:
                sta_het_avg_e = []
                st_het_chstring_e = ''
            if type(st_het_ch_p) == list and len(sta_het_df) > 0:
                sta_het_avg_p, st_het_chstring_p = calc_av_en_flux_ST_HET(sta_het_df.filter(like='Proton'),
                                                                          sta_het_meta['channels_dict_df_p'],
                                                                          st_het_ch_p, species='p')
            else:
                sta_het_avg_p = []
                st_het_chstring_p = ''
    if SOHO:
        if erne:
            if type(erne_p_ch) == list and len(soho_erne) > 0:
                soho_erne_avg_p, soho_erne_chstring_p = calc_av_en_flux_ERNE(soho_erne.filter(like='PH_'),
                                                                             erne_energies['channels_dict_df_p'],
                                                                             erne_p_ch,
                                                                             species='p',
                                                                             sensor='HET')
    if PSP:
        if len(psp_het) > 0:
            if plot_e_1:
                print('calc_av_en_flux_PSP_EPIHI e 1 MeV')
                df_psp_het_e, psp_het_chstring_e = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, psp_het_ch_e, 'e', 'het', 'A')
                if isinstance(psp_het_resample, str):
                    df_psp_het_e = resample_df(df_psp_het_e, psp_het_resample)
            if plot_p:
                print('calc_av_en_flux_PSP_EPIHI p')
                df_psp_het_p, psp_het_chstring_p = calc_av_en_flux_PSP_EPIHI(psp_het, psp_het_energies, psp_het_ch_p, 'p', 'het', 'A')
                if isinstance(psp_het_resample, str):
                    df_psp_het_p = resample_df(df_psp_het_p, psp_het_resample)
        if len(psp_epilo) > 0:
            if plot_e_100:
                print('calc_av_en_flux_PSP_EPILO e 100 keV')            
                df_psp_epilo_e, psp_epilo_chstring_e = calc_av_en_flux_PSP_EPILO(psp_epilo,
                                                                                 psp_epilo_energies,
                                                                                 psp_epilo_ch_e100,
                                                                                 species='e',
                                                                                 mode='pe',
                                                                                 chan=psp_epilo_channel,
                                                                                 viewing=psp_epilo_viewing)


                # select energy channel
                # TODO: introduce calc_av_en_flux_PSP_EPILO(). ATM, if list of channels, only first one is selected
                # if type(psp_epilo_ch_e100) is list:
                    # psp_epilo_ch_e100 = psp_epilo_ch_e100[0]
                # df_psp_epilo_e = df_psp_epilo_e.filter(like=f'_E{psp_epilo_ch_e100}_')

                # energy = en_dict['Electron_ChanF_Energy'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # energy_low = energy - en_dict['Electron_ChanF_Energy_DELTAMINUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # energy_high = energy + en_dict['Electron_ChanF_Energy_DELTAPLUS'].filter(like=f'_E{en_channel}_P{viewing}').values[0]
                # chstring_e = np.round(energy_low,1).astype(str) + ' - ' + np.round(energy_high,1).astype(str) + ' keV'

                if isinstance(psp_epilo_resample, str):
                    df_psp_epilo_e = resample_df(df_psp_epilo_e, psp_epilo_resample)

    ##########################################################################################


    panels = 0
    if plot_e_1:
        panels = panels + 1
    if plot_e_100:
        panels = panels + 1
    if plot_p:
        panels = panels + 1
    fig, axes = plt.subplots(panels, figsize=(24, 15), dpi=200, sharex=True)
    axnum = 0
    # Intensities
    ####################################################################
    # 100 KEV ELECTRONS
    #################################################################
    if plot_e_100:
        if panels == 1:
            ax = axes
        else:
            ax = axes[axnum]
        species_string = 'Electrons'
        if ept_use_corr_e:
            species_string = 'Electrons (corrected)'

        if PSP:
            if len(psp_epilo) > 0:
                ax.plot(df_psp_epilo_e.index, df_psp_epilo_e*100, color=psp_het_color, linewidth=linewidth,
                        label='PSP '+r"$\bf{(count\ rate\ *100)}$"+'\nISOIS-EPILO '+psp_epilo_chstring_e+f'\nF (W{psp_epilo_viewing})',
                        drawstyle='steps-mid')

        if Bepi:
            ax.plot(sixs_e.index, sixs_e[sixs_ch_e], color='orange', linewidth=linewidth, label='BepiColombo\nSIXS '+sixs_chstrings[sixs_ch_e]+f'\nside {sixs_side_e}', drawstyle='steps-mid')
        if SOLO:
            if ept and (len(ept_e) > 0):
                flux_ept = df_ept_e.values
                try:
                    for ch in ept_ch_e100:
                        ax.plot(df_ept_e.index.values, flux_ept[:, ch], linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_en_str_e[ch, 0]+f'\n{sector}', drawstyle='steps-mid')
                except IndexError:
                    ax.plot(df_ept_e.index.values, flux_ept, linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_chstring_e+f'\n{sector}', drawstyle='steps-mid')
        if STEREO:
            if sept_e:
                if type(sept_ch_e) == list and len(sta_sept_avg_e) > 0:
                    ax.plot(sta_sept_avg_e.index, sta_sept_avg_e, color=stereo_sept_color, linewidth=linewidth,
                            label='STEREO/SEPT '+sept_chstring_e+f' {sector}', drawstyle='steps-mid')
                elif type(sept_ch_e) == int:
                    ax.plot(sta_sept_df_e.index, sta_sept_df_e[f'ch_{sept_ch_e}'], color=stereo_sept_color,
                            linewidth=linewidth, label='STEREO/SEPT '+sta_sept_dict_e.loc[sept_ch_e]['ch_strings']+f' {sector}', drawstyle='steps-mid')

        if SOHO:
            if ephin_e:
                ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color,
                        linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}',
                        drawstyle='steps-mid')
        if WIND:
            if len(wind3dp_e_df) > 0:
                # multiply by 1e6 to get per MeV
                ax.plot(wind3dp_e_df.index, wind3dp_e_df[f'FLUX_{wind3dp_ch_e}']*1e6, color=wind_color, linewidth=linewidth, label='Wind/3DP '+wind3dp_e_meta['channels_dict_df']['Bins_Text'][wind3dp_ch_e], drawstyle='steps-mid')

        # ax.set_ylim(7.9e-3, 4.7e1)
        # ax.set_ylim(0.3842003987966555, 6333.090511873226)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='100 keV '+species_string)
        axnum = axnum + 1

    # 1 MEV ELECTRONS
    #################################################################
    if plot_e_1:
        if panels == 1:
            ax = axes
        else:
            ax = axes[axnum]
            species_string = 'Electrons'
        if ept_use_corr_e:
            species_string = 'Electrons (corrected)'

        if PSP:
            if len(psp_het) > 0:
                # ax.plot(psp_het.index, psp_het[f'A_Electrons_Rate_{psp_het_ch_e}'], color=psp_het_color, linewidth=linewidth,
                #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_energies['Electrons_ENERGY_LABL'][psp_het_ch_e][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
                #         drawstyle='steps-mid')
                ax.plot(df_psp_het_e.index, df_psp_het_e*10, color=psp_het_color, linewidth=linewidth,
                        label='PSP '+r"$\bf{(count\ rate\ *10)}$"+'\nISOIS-EPIHI-HET '+psp_het_chstring_e+'\nA (sun)',
                        drawstyle='steps-mid')
        if Bepi:
            ax.plot(sixs_e.index, sixs_e[sixs_ch_e100], color='orange', linewidth=linewidth,
                    label='Bepi/SIXS '+sixs_chstrings[sixs_ch_e100]+f' side {sixs_side_e}', drawstyle='steps-mid')
        if SOLO:
            if het and (len(het_e) > 0):
                ax.plot(df_het_e.index.values, df_het_e.flux, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_e+f' {sector}', drawstyle='steps-mid')
        if STEREO:
            if stereo_het:
                if len(sta_het_avg_e) > 0:
                    ax.plot(sta_het_avg_e.index, sta_het_avg_e, color=stereo_het_color, linewidth=linewidth,
                            label='STEREO/HET '+st_het_chstring_e, drawstyle='steps-mid')
        if SOHO:
            if ephin_e:
                ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}', drawstyle='steps-mid')

        # ax.set_ylim(7.9e-3, 4.7e1)
        # ax.set_ylim(0.3842003987966555, 6333.090511873226)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='1 MeV '+species_string)
        axnum = axnum + 1

    # PROTONS
    #################################################################
    if plot_p:
        if panels == 1:
            ax = axes
        else:
            ax = axes[axnum]
        if PSP:
            if len(psp_het) > 0:
                # ax.plot(psp_het.index, psp_het[f'A_H_Flux_{psp_het_ch_p}'], color=psp_het_color, linewidth=linewidth,
                #         label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_energies['H_ENERGY_LABL'][psp_het_ch_p][0].replace(' ', '').replace('-', ' - ').replace('MeV', ' MeV')+'\nA (sun)',
                #         drawstyle='steps-mid')
                ax.plot(df_psp_het_p.index, df_psp_het_p, color=psp_het_color, linewidth=linewidth,
                        label='PSP '+r"$\bf{(count\ rates)}$"+'\nISOIS-EPIHI-HET '+psp_het_chstring_p+'\nA (sun)',
                        drawstyle='steps-mid')
        if Bepi:
            ax.plot(sixs_p.index, sixs_p[sixs_ch_p], color='orange', linewidth=linewidth, label='BepiColombo/SIXS '+sixs_chstrings[sixs_ch_p]+f' side {sixs_side_p}', drawstyle='steps-mid')
        if SOLO:
            if het and (len(ept_e) > 0):
                ax.plot(df_het_p.index, df_het_p, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_p+f' {sector}', drawstyle='steps-mid')
        if STEREO:
            if sept_p:
                if type(sept_ch_p) == list and len(sta_sept_avg_p) > 0:
                    ax.plot(sta_sept_df_p.index, sta_sept_avg_p, color=stereo_sept_color, linewidth=linewidth, label='STEREO/SEPT '+sept_chstring_p+f' {sector}', drawstyle='steps-mid')
                elif type(sept_ch_p) == int:
                    ax.plot(sta_sept_df_p.index, sta_sept_df_p[f'ch_{sept_ch_p}'], color=stereo_sept_color, linewidth=linewidth, label='STEREO/SEPT '+sta_sept_dict_p.loc[sept_ch_p]['ch_strings']+f' {sector}', drawstyle='steps-mid')
            if stereo_het:
                if len(sta_het_avg_p) > 0:
                    ax.plot(sta_het_avg_p.index, sta_het_avg_p, color=stereo_het_color,
                            linewidth=linewidth, label='STEREO/HET '+st_het_chstring_p, drawstyle='steps-mid')
            if let:
                str_ch = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
                ax.plot(sta_let_df.index, sta_let_df[f'H_unsec_flux_{let_ch}'], color=stereo_let_color, linewidth=linewidth, label='STERE/LET '+let_chstring[let_ch], drawstyle='steps-mid')
        if SOHO:
            if erne:
                if type(erne_p_ch) == list and len(soho_erne_avg_p) > 0:
                    ax.plot(soho_erne_avg_p.index, soho_erne_avg_p, color=soho_erne_color, linewidth=linewidth, label='SOHO/ERNE/HED '+soho_erne_chstring_p, drawstyle='steps-mid')
                elif type(erne_p_ch) == int:
                    if len(soho_erne) > 0:
                        ax.plot(soho_erne.index, soho_erne[f'PH_{erne_p_ch}'], color=soho_erne_color, linewidth=linewidth, label='SOHO/ERNE/HED '+erne_chstring[erne_p_ch], drawstyle='steps-mid')
            if ephin_p:
                ax.plot(ephin['date'], ephin[ephin_ch_p][0], color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_p][1], drawstyle='steps-mid')
        #if WIND:
            # multiply by 1e6 to get per MeV
        #    ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label='Wind\n3DP '+str(round(wind3dp_p_df[f'ENERGY_{wind3dp_ch_p}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
        # ax.set_ylim(2.05e-5, 4.8e0)
        # ax.set_ylim(0.00033920545179055416, 249.08996960298424)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='>25 MeV Protons/Ions')
        axnum = axnum+1
    # pos = get_horizons_coord('Solar Orbiter', startdate, 'id')
    # dist = np.round(pos.radius.value, 2)
    # fig.suptitle(f'Solar Orbiter/EPD {sector} (R={dist} au)')
    ax.set_xlim(startdate, enddate)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%H:%M'))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel('Date / Time in year '+str(startdate.year))
    plt.tight_layout()
    fig.subplots_adjust(hspace=0.1)
    if save_fig:
        species = ''
        if plot_e_1 or plot_e_100:
            species = species+'e'
        if plot_p:
            species = species+'p'
        plt.savefig(outfile)
        plt.close()
        print('')
        print('Saved '+outfile)
    else:
        plt.show()
