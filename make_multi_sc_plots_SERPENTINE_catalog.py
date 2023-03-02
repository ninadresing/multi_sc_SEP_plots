import datetime as dt
import os
import warnings

# import cdflib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sunpy
from seppy.tools import inf_inj_time
from matplotlib.ticker import AutoMinorLocator
from matplotlib.transforms import blended_transform_factory
from seppy.loader.psp import calc_av_en_flux_PSP_EPIHI, calc_av_en_flux_PSP_EPILO, psp_isois_load, resample_df
from seppy.loader.soho import calc_av_en_flux_ERNE, soho_load
from solo_epd_loader import epd_load
from seppy.loader.stereo import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from seppy.loader.stereo import calc_av_en_flux_SEPT, stereo_load
# import astropy.units as u
# from cdflib.epochs import CDFepoch
# from sunpy import log
# from sunpy.net import Fido
# from sunpy.net import attrs as a
# from sunpy.timeseries import TimeSeries
# from sunpy.util.exceptions import warn_user
from tqdm import tqdm
from seppy.loader.wind import wind3dp_load

'''
June 2022
this loops over the mission time and creates
multi-sc plots to be used for the SERPENTINE
multi-sc SEP event catalog
moved this file to Deepnote June 15 2022
'''

# make selections
#############################################################

# processing mode: 'regular' (e.g. weekly) or 'events'
mode = 'regular'

lower_proton = False  # True if 13 MeV protons should be used instead of 25+ MeV

if mode == 'regular':
    first_date = dt.datetime(2022, 11, 19)
    last_date = dt.datetime(2022, 12, 31)
    plot_period = '7D'
    averaging = '1h'  # '5min'  # None

if mode == 'events':
    averaging = '20min'  # '5min' None

Bepi = True
PSP = True
SOHO = True
SOLO = True
STEREO = True
WIND = True


# SOHO:
erne = True
ephin_p = False  # not included yet!
ephin_e = True  # not included yet!

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

# plot vertical lines with previously found onset and peak times
plot_times = True
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


def bepicolombo_sixs_stack(path, date, side):
    # def bepicolombo_sixs_stack(path, date, side, species):

    # side is the index of the file here
    try:
        try:
            filename = f"{path}/sixs_phys_data_{date}_side{side}.csv"
            df = pd.read_csv(filename)
        except FileNotFoundError:
            # try alternative file name format
            filename = f"{path}/{date.strftime('%Y%m%d')}_side{side}.csv"
            df = pd.read_csv(filename)
            times = pd.to_datetime(df['TimeUTC'])

        # list comprehension because the method can't be applied onto the array "times"
        times = [t.tz_convert(None) for t in times]
        df.index = np.array(times)
        df = df.drop(columns=['TimeUTC'])

        # choose the subset of desired particle species
        # if species=="ion":
        #     df = df[[f"P{i}" for i in range(1,10)]]
        # if species=="ele":
        #     df = df[[f"E{i}" for i in range(1,8)]]

    except FileNotFoundError:
        print(f'Unable to open {filename}')
        df = pd.DataFrame()
        filename = ''

    return df, filename


def bepi_sixs_load(startdate, enddate, side, path):
    dates = pd.date_range(startdate, enddate)

    # read files into Pandas dataframes:
    df, file = bepicolombo_sixs_stack(path, startdate, side=side)
    if len(dates) > 1:
        for date in dates[1:]:
            t_df, file = bepicolombo_sixs_stack(path, date.date(), side=side)
            df = pd.concat([df, t_df])

    channels_dict = {"Energy_Bin_str": {'E1': '71 keV', 'E2': '106 keV', 'E3': '169 keV', 'E4': '280 keV', 'E5': '960 keV', 'E6': '2240 keV', 'E7': '8170 keV',
                                        'P1': '1.1 MeV', 'P2': '1.2 MeV', 'P3': '1.5 MeV', 'P4': '2.3 MeV', 'P5': '4.0 MeV', 'P6': '8.0 MeV', 'P7': '14.5 MeV', 'P8': '25.1 MeV', 'P9': '37.3 MeV'},
                     "Electron_Bins_Low_Energy": np.array([55, 78, 134, 235, 1000, 1432, 4904]),
                     "Electron_Bins_High_Energy": np.array([92, 143, 214, 331, 1193, 3165, 10000]),
                     "Ion_Bins_Low_Energy": np.array([0.001, 1.088, 1.407, 2.139, 3.647, 7.533, 13.211, 22.606, 29.246]),
                     "Ion_Bins_High_Energy": np.array([1.254, 1.311, 1.608, 2.388, 4.241, 8.534, 15.515, 28.413, 40.0])}
    return df, channels_dict


def calc_av_en_flux_sixs(df, meta, channel, species):
    """
    This function averages the flux of two energy channels of BepiColombo/SIXS into a combined energy channel
    channel numbers counted from 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing HET data
    channel : int or list
        energy channel or list with first and last channel to be used
    species : string
        'e', 'electrons', 'p', 'protons'

    Returns
    -------
    flux: pd.DataFrame
        channel-averaged flux
    en_channel_string: str
        string containing the energy information of combined channel
    """

    # define constant geometric factors
    GEOMFACTOR_PROT8 = 5.97E-01
    GEOMFACTOR_PROT9 = 4.09E+00
    GEOMFACTOR_ELEC5 = 1.99E-02
    GEOMFACTOR_ELEC6 = 1.33E-01
    GEOMFACTOR_PROT_COMB89 = 3.34
    GEOMFACTOR_ELEC_COMB56 = 0.0972

    if type(channel) == list and len(channel) == 1:
        channel = channel[0]

    if species in ['p', 'protons']:
        if channel == [8, 9]:
            countrate = df['P8'] * GEOMFACTOR_PROT8 + df['P9'] * GEOMFACTOR_PROT9
            flux = countrate / GEOMFACTOR_PROT_COMB89
            en_channel_string = '37 MeV'
        else:
            flux = df[f'P{channel}']
            en_channel_string = sixs_meta['Energy_Bin_str'][f'P{channel}']

    if species in ['e', 'electrons']:
        if channel == [5, 6]:
            countrate = df['E5'] * GEOMFACTOR_ELEC5 + df['E6'] * GEOMFACTOR_ELEC6
            flux = countrate / GEOMFACTOR_ELEC_COMB56
            en_channel_string = '1.4 MeV'
        else:
            flux = df[f'E{channel}']
            en_channel_string = sixs_meta['Energy_Bin_str'][f'E{channel}']

    return flux, en_channel_string


# some plot options
intensity_label = 'Flux\n/(s cm² sr MeV)'
linewidth = 1.5
outpath = None  # os.getcwd()
plot_e_100 = True
plot_e_1 = True
plot_p = True
save_fig = True
outpath = 'plots'  # '/Users/dresing/Documents/Proposals/SERPENTINE_H2020/Cycle25_Multi-SC_SEP_Event_List/Multi_sc_plots'


"""
START LOAD ONSET TIMES
"""
if plot_times:
    # First you have to manually (!) bring spreadsheet in the same form as the example file!!!

    # Load modified spreadhseet
    # df = pd.read_csv('WP2_multi-sc_SEP_event_list_D2.3.csv')
    df = pd.read_csv('WP2_multi_sc_catalog - WP2_multi_sc_event_list_draft.csv')
    # df = pd.read_excel('WP2_multi-sc_SEP_event_list_D2.3.xlsx')

    # get list of flare times
    df_flare_date_str = df['flare date\n(yyyy-mm-dd)'].values.astype(str)
    df_flare_date_str = np.delete(df_flare_date_str, np.where(df_flare_date_str == 'nan'))
    df_flare_times_str = df['flare time\n(HH:MM:SS)'].values.astype(str)
    df_flare_times_str = np.delete(df_flare_times_str, np.where(df_flare_times_str == 'nan'))
    df_flare_times = []
    for i in range(len(df_flare_date_str)):
        df_flare_times.append(dt.datetime.strptime(f'{df_flare_date_str[i]} {df_flare_times_str[i]}', '%Y-%m-%d %H:%M:%S'))

    def get_times_from_csv_list(df, observer):
        """
        df_onset_p, df_peak_p, df_onset_e100, df_peak_e100, df_onset_e1000, df_peak_e1000 = get_times_from_csv_list(df, 'SOLO')
        """

        df_solo = df[df.Observer == observer]

        # protons
        df_solo_onset_date_p_str = df_solo['p25MeV onset date (yyyy-mm-dd)'].values.astype(str)
        df_solo_onset_date_p_str = np.delete(df_solo_onset_date_p_str, np.where(df_solo_onset_date_p_str == 'nan'))
        df_solo_onset_time_p_str = df_solo['p25MeV onset time (HH:MM:SS)'].values.astype(str)
        df_solo_onset_time_p_str = np.delete(df_solo_onset_time_p_str, np.where(df_solo_onset_time_p_str == 'nan'))

        df_solo_peak_date_p_str = df_solo['p25MeV peak date (yyyy-mm-dd)'].values.astype(str)
        df_solo_peak_date_p_str = np.delete(df_solo_peak_date_p_str, np.where(df_solo_peak_date_p_str == 'nan'))
        df_solo_peak_time_p_str = df_solo['p25MeV peak time (HH:MM:SS)'].values.astype(str)
        df_solo_peak_time_p_str = np.delete(df_solo_peak_time_p_str, np.where(df_solo_peak_time_p_str == 'nan'))

        df_solo_onset_p = []
        for i in range(len(df_solo_onset_date_p_str)):
            df_solo_onset_p.append(dt.datetime.strptime(f'{df_solo_onset_date_p_str[i]} {df_solo_onset_time_p_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_onset_date_p_str[i]} {df_solo_onset_time_p_str[i]} => {df_solo_onset_p[i]}')

        df_solo_peak_p = []
        for i in range(len(df_solo_peak_date_p_str)):
            df_solo_peak_p.append(dt.datetime.strptime(f'{df_solo_peak_date_p_str[i]} {df_solo_peak_time_p_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_peak_date_p_str[i]} {df_solo_peak_time_p_str[i]} => {df_solo_peak_p[i]}')

        # 100 keV electrons
        df_solo_onset_date_e100_str = df_solo['e100keV onset date (yyyy-mm-dd)'].values.astype(str)
        df_solo_onset_date_e100_str = np.delete(df_solo_onset_date_e100_str, np.where(df_solo_onset_date_e100_str == 'nan'))
        df_solo_onset_time_e100_str = df_solo['e100keV onset time (HH:MM:SS)'].values.astype(str)
        df_solo_onset_time_e100_str = np.delete(df_solo_onset_time_e100_str, np.where(df_solo_onset_time_e100_str == 'nan'))

        df_solo_peak_date_e100_str = df_solo['e100keV peak date (yyyy-mm-dd)'].values.astype(str)
        df_solo_peak_date_e100_str = np.delete(df_solo_peak_date_e100_str, np.where(df_solo_peak_date_e100_str == 'nan'))
        df_solo_peak_time_e100_str = df_solo['e100keV peak time (HH:MM:SS)'].values.astype(str)
        df_solo_peak_time_e100_str = np.delete(df_solo_peak_time_e100_str, np.where(df_solo_peak_time_e100_str == 'nan'))

        df_solo_onset_e100 = []
        for i in range(len(df_solo_onset_date_e100_str)):
            df_solo_onset_e100.append(dt.datetime.strptime(f'{df_solo_onset_date_e100_str[i]} {df_solo_onset_time_e100_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_onset_date_e100_str[i]} {df_solo_onset_time_e100_str[i]} => {df_solo_onset_e100[i]}')

        df_solo_peak_e100 = []
        for i in range(len(df_solo_peak_date_e100_str)):
            df_solo_peak_e100.append(dt.datetime.strptime(f'{df_solo_peak_date_e100_str[i]} {df_solo_peak_time_e100_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_peak_date_e100_str[i]} {df_solo_peak_time_e100_str[i]} => {df_solo_peak_e100[i]}')

        # 1000 keV (1 MeV) electrons
        df_solo_onset_date_e1000_str = df_solo['e1MeV onset date (yyyy-mm-dd)'].values.astype(str)
        df_solo_onset_date_e1000_str = np.delete(df_solo_onset_date_e1000_str, np.where(df_solo_onset_date_e1000_str == 'nan'))
        df_solo_onset_time_e1000_str = df_solo['e1MeV onset time (HH:MM:SS)'].values.astype(str)
        df_solo_onset_time_e1000_str = np.delete(df_solo_onset_time_e1000_str, np.where(df_solo_onset_time_e1000_str == 'nan'))

        df_solo_peak_date_e1000_str = df_solo['e1MeV peak date (yyyy-mm-dd)'].values.astype(str)
        df_solo_peak_date_e1000_str = np.delete(df_solo_peak_date_e1000_str, np.where(df_solo_peak_date_e1000_str == 'nan'))
        df_solo_peak_time_e1000_str = df_solo['e1MeV peak time (HH:MM:SS)'].values.astype(str)
        df_solo_peak_time_e1000_str = np.delete(df_solo_peak_time_e1000_str, np.where(df_solo_peak_time_e1000_str == 'nan'))

        df_solo_onset_e1000 = []
        for i in range(len(df_solo_onset_date_e1000_str)):
            df_solo_onset_e1000.append(dt.datetime.strptime(f'{df_solo_onset_date_e1000_str[i]} {df_solo_onset_time_e1000_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_onset_date_e1000_str[i]} {df_solo_onset_time_e1000_str[i]} => {df_solo_onset_e1000[i]}')

        df_solo_peak_e1000 = []
        for i in range(len(df_solo_peak_date_e1000_str)):
            df_solo_peak_e1000.append(dt.datetime.strptime(f'{df_solo_peak_date_e1000_str[i]} {df_solo_peak_time_e1000_str[i]}', '%Y-%m-%d %H:%M:%S'))
            # print(f'{df_solo_peak_date_e1000_str[i]} {df_solo_peak_time_e1000_str[i]} => {df_solo_peak_e1000[i]}')

        return df_solo_onset_p, df_solo_peak_p, df_solo_onset_e100, df_solo_peak_e100, df_solo_onset_e1000, df_solo_peak_e1000

    df_bepi_onset_p, df_bepi_peak_p, df_bepi_onset_e100, df_bepi_peak_e100, df_bepi_onset_e1000, df_bepi_peak_e1000 = get_times_from_csv_list(df, 'BepiColombo')
    df_solo_onset_p, df_solo_peak_p, df_solo_onset_e100, df_solo_peak_e100, df_solo_onset_e1000, df_solo_peak_e1000 = get_times_from_csv_list(df, 'SOLO')
    df_sta_onset_p, df_sta_peak_p, df_sta_onset_e100, df_sta_peak_e100, df_sta_onset_e1000, df_sta_peak_e1000 = get_times_from_csv_list(df, 'STEREO-A')
    df_psp_onset_p, df_psp_peak_p, df_psp_onset_e100, df_psp_peak_e100, df_psp_onset_e1000, df_psp_peak_e1000 = get_times_from_csv_list(df, 'PSP')
    df_wind_onset_p, df_wind_peak_p, df_wind_onset_e100, df_wind_peak_e100, df_wind_onset_e1000, df_wind_peak_e1000 = get_times_from_csv_list(df, 'L1 (SOHO/Wind)')  # previously 'Wind'
    df_soho_onset_p, df_soho_peak_p, df_soho_onset_e100, df_soho_peak_e100, df_soho_onset_e1000, df_soho_peak_e1000 = get_times_from_csv_list(df, 'L1 (SOHO/Wind)')  # previously 'SOHO'

    # obtain a list of all onset datetime
    all_onsets = df_bepi_onset_p + df_bepi_onset_e100 + df_bepi_onset_e1000 + df_solo_onset_p + df_solo_onset_e100 + df_solo_onset_e1000 + df_sta_onset_p + df_sta_onset_e100 + df_sta_onset_e1000 + df_psp_onset_p + df_psp_onset_e100 + df_psp_onset_e1000 + df_wind_onset_p + df_wind_onset_e100 + df_wind_onset_e1000 + df_soho_onset_p + df_soho_onset_e100 + df_soho_onset_e1000
    all_onsets.sort()
    # obtain a list of all onset dates
    all_onset_dates_org = np.array([i.date() for i in all_onsets])
    all_onset_dates_org.sort()
    # remove duplicates
    all_onset_dates = np.array([*set(all_onset_dates_org)])
    all_onset_dates.sort()  # <--really needed!
    # obtain list of datetimes with only the earliest onset time for each date
    all_onset_dates_first = []
    for i, date in enumerate(all_onset_dates):
        all_onset_dates_first.append(all_onsets[np.where(all_onset_dates_org == date)[0][0]])
"""
END LOAD ONSET TIMES
"""


"""
START CALC INF_INJ_TIMES
"""
calc_inf_inj_time = False
if calc_inf_inj_time:
    df = pd.read_csv('WP2_multi_sc_catalog - WP2_multi_sc_event_list_draft.csv')

    fixed_mean_energies_p = {'SOLO': np.sqrt(25.09*41.18),
                             'PSP': np.sqrt(26.91*38.05),
                             'STEREO-A': np.sqrt(26.3*40.5),
                             'L1 (SOHO/Wind)': np.sqrt(25*40),
                             'BepiColombo': 37.0
                             }
    fixed_mean_energies_e1000 = {'SOLO': np.sqrt(0.4533*2.4010),
                                 'PSP': np.sqrt(0.7071*2.8284),
                                 'STEREO-A': np.sqrt(0.7*2.8),
                                 'L1 (SOHO/Wind)': np.sqrt(0.67*10.4),
                                 'BepiColombo': 1.4
                                 }
    fixed_mean_energies_e100 = {'SOLO': np.sqrt(85.6*130.5)/1000.,
                                'PSP': np.sqrt(65.91*153.50)/1000.,
                                'STEREO-A': np.sqrt(85.*125.)/1000.,
                                'L1 (SOHO/Wind)': np.sqrt(75.63*140.46)/1000.,
                                'BepiColombo': 0.106
                                }
    sw = 400

    # inj_times_p = []
    # distances_p = []
    # inj_times_e100 = []
    # distances_e100 = []
    # inj_times_e1000 = []
    # distances_e1000 = []

    for i in tqdm(range(df.shape[0])):
        mission = df['Observer'].iloc[i]
        if mission == 'L1 (SOHO/Wind)':
            mission_p = 'SOHO'
            mission_e1000 = 'SOHO'
            mission_e100 = 'Wind'
        else:
            mission_p = mission
            mission_e1000 = mission
            mission_e100 = mission

        onset_date_p_str = df['p25MeV onset date (yyyy-mm-dd)'].iloc[i]
        onset_time_p_str = df['p25MeV onset time (HH:MM:SS)'].iloc[i]
        if type(onset_date_p_str) != str or type(onset_time_p_str) != str:
            onset_p = pd.NaT
        else:
            onset_p = dt.datetime.strptime(f'{onset_date_p_str} {onset_time_p_str}', '%Y-%m-%d %H:%M:%S')

        onset_date_e1000_str = df['e1MeV onset date (yyyy-mm-dd)'].iloc[i]
        onset_time_e1000_str = df['e1MeV onset time (HH:MM:SS)'].iloc[i]
        if type(onset_date_e1000_str) != str or type(onset_time_e1000_str) != str:
            onset_e1000 = pd.NaT
        else:
            onset_e1000 = dt.datetime.strptime(f'{onset_date_e1000_str} {onset_time_e1000_str}', '%Y-%m-%d %H:%M:%S')

        onset_date_e100_str = df['e100keV onset date (yyyy-mm-dd)'].iloc[i]
        onset_time_e100_str = df['e100keV onset time (HH:MM:SS)'].iloc[i]
        if type(onset_date_e100_str) != str or type(onset_time_e100_str) != str:
            onset_e100 = pd.NaT
        else:
            onset_e100 = dt.datetime.strptime(f'{onset_date_e100_str} {onset_time_e100_str}', '%Y-%m-%d %H:%M:%S')

        if not type(onset_p) == pd._libs.tslibs.nattype.NaTType:
            inj_time_p, distance_p = inf_inj_time(mission_p, onset_p, 'p', fixed_mean_energies_p[mission], sw)
        else:
            inj_time_p = pd.NaT
            distance_p = np.nan
        if not type(onset_e100) == pd._libs.tslibs.nattype.NaTType:
            inj_time_e100, distance_e100 = inf_inj_time(mission_e100, onset_e100, 'e', fixed_mean_energies_e100[mission], sw)
            # use different energy channels for PSP before 14 June 2021:
            if mission == 'PSP' and onset_e100 < dt.datetime(2021, 6, 14):
                inj_time_e100, distance_e100 = inf_inj_time(mission_e100, onset_e100, 'e', np.sqrt(84.1*131.6)/1000., sw)
        else:
            inj_time_e100 = pd.NaT
            distance_e100 = np.nan
        if not type(onset_e1000) == pd._libs.tslibs.nattype.NaTType:
            inj_time_e1000, distance_e1000 = inf_inj_time(mission_e1000, onset_e1000, 'e', fixed_mean_energies_e1000[mission], sw)
        else:
            inj_time_e1000 = pd.NaT
            distance_e1000 = np.nan

        if not type(inj_time_p) == pd._libs.tslibs.nattype.NaTType:
            df['p25MeV inferred injection time (HH:MM:SS)'].iloc[i] = inj_time_p.strftime('%H:%M:%S')
            df['p25MeV inferred injection date (yyyy-mm-dd)'].iloc[i] = inj_time_p.strftime('%Y-%m-%d')
            df['p25MeV pathlength used for inferred injection time (au)'].iloc[i] = distance_p.value
        if not type(inj_time_e1000) == pd._libs.tslibs.nattype.NaTType:
            df['e1MeV inferred injection time (HH:MM:SS)'].iloc[i] = inj_time_e1000.strftime('%H:%M:%S')
            df['e1MeV inferred injection date (yyyy-mm-dd)'].iloc[i] = inj_time_e1000.strftime('%Y-%m-%d')
            df['e1MeV pathlength used for inferred injection time (au)'].iloc[i] = distance_e1000.value
        if not type(inj_time_e100) == pd._libs.tslibs.nattype.NaTType:
            df['e100keV inferred injection time (HH:MM:SS)'].iloc[i] = inj_time_e100.strftime('%H:%M:%S')
            df['e100keV inferred injection date (yyyy-mm-dd)'].iloc[i] = inj_time_e100.strftime('%Y-%m-%d')
            df['e100keV pathlength used for inferred injection time (au)'].iloc[i] = distance_e100.value

        # inj_times_p.append(inj_time_p)
        # distances_p.append(distance_p)
        # inj_times_e100.append(inj_time_e100)
        # distances_e100.append(distance_e100)
        # inj_times_e1000.append(inj_time_e1000)
        # distances_e1000.append(distance_e1000)

        # print('p    :', onset_date_p_str, onset_time_p_str, onset_p, distance_p, inj_time_p)
        # print('e1000:', onset_date_e1000_str, onset_time_e1000_str, onset_e1000, distance_e1000, inj_time_e1000)
        # print('e100 :', onset_date_e100_str, onset_time_e100_str, onset_e100, distance_e100, inj_time_e100)

        df.to_csv('new.csv', index=False)

# calc_inf_inj_time = False
# if calc_inf_inj_time:
#     #
#     def get_inf_inj_time(mission, onset_list, particle, energy, solarwind_speed=400):
#         inf_inj_times = []
#         distances = []
#         for i in tqdm(onset_list):
#             inj_time, distance = inf_inj_time(mission, i, particle, energy, solarwind_speed)
#             inf_inj_times.append(inj_time)
#             distances.append(distance)
#             # print(i, inj_time, distance)
#         return inf_inj_times, distances
#     #
#     sta_inf_inj_times_p, sta_distances_p = get_inf_inj_time('STEREO-A', df_sta_onset_p, 'p', np.sqrt(26.3*40.5))
#     sta_inf_inj_times_e1000, sta_distances_e1000 = get_inf_inj_time('STEREO-A', df_sta_onset_e1000, 'e', np.sqrt(26.3*40.5))
#     sta_inf_inj_times_e100, sta_distances_e100 = get_inf_inj_time('STEREO-A', df_sta_onset_e100, 'e', np.sqrt(85.*125.)/1000.)
#     #
#     soho_inf_inj_times_p, soho_distances_p = get_inf_inj_time('SOHO', df_soho_onset_p, 'p', np.sqrt(25*40))
#     soho_inf_inj_times_e1000, soho_distances_e1000 = get_inf_inj_time('SOHO', df_soho_onset_e1000, 'e', np.sqrt(0.67*10.4))
#     wind_inf_inj_times_e100, wind_distances_e100 = get_inf_inj_time('Wind', df_wind_onset_e100, 'e', np.sqrt(75.63*140.46)/1000.)
#     #
#     solo_inf_inj_times_p, solo_distances_p = get_inf_inj_time('Solar Orbiter', df_solo_onset_p, 'p', np.sqrt(25.09*41.18))
#     solo_inf_inj_times_e1000, solo_distances_e1000 = get_inf_inj_time('Solar Orbiter', df_solo_onset_e1000, 'e', np.sqrt(0.4533*2.4010))
#     solo_inf_inj_times_e100, solo_distances_e100 = get_inf_inj_time('Solar Orbiter', df_solo_onset_e100, 'e', np.sqrt(85.6*130.5)/1000.)
#     #
#     psp_inf_inj_times_p, psp_distances_p = get_inf_inj_time('Parker Solar Probe', df_psp_onset_p, 'p', np.sqrt(26.91*38.05))
#     psp_inf_inj_times_e1000, psp_distances_e1000 = get_inf_inj_time('Parker Solar Probe', df_psp_onset_e1000, 'e', np.sqrt(0.7071*2.8284))
#     psp_inf_inj_times_e100, psp_distances_e100 = get_inf_inj_time('Parker Solar Probe', df_psp_onset_e100, 'e', np.sqrt(65.91*153.50)/1000.)
#     #
#     bepi_inf_inj_times_p, bepi_distances_p = get_inf_inj_time('BepiColombo', df_bepi_onset_p, 'p', 37.0)
#     bepi_inf_inj_times_e1000, bepi_distances_e1000 = get_inf_inj_time('BepiColombo', df_bepi_onset_e1000, 'e', 1.4)
#     bepi_inf_inj_times_e100, bepi_distances_e100 = get_inf_inj_time('BepiColombo', df_bepi_onset_e100, 'e', 0.106)
#     #
#     # e.g:
#     # for i in range(len(df_bepi_onset_e100)):
#     #     print(df_bepi_onset_e100[i], bepi_inf_inj_times_e100[i], bepi_distances_e100[i])
"""
END CALC INF_INJ_TIMES
"""


"""
START PLOT PEAK FLUXES VS DATES
"""
plot_peak_vs_time = False
if plot_peak_vs_time:
    fig, ax = plt.subplots(figsize=(10, 6))
    for obs in ['SOLO', 'STEREO-A', 'Wind', 'BepiColombo', 'PSP']:  # 'PSP' removed bc. scaled count rates only
        df_obs = df[df.Observer == obs]
        nindex = np.where(df_obs['ELECTRONS 100 keV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_obs['ELECTRONS 100 keV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_obs['ELECTRONS 100 keV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        ax.plot(dates, fluxes, 'o', label=f'{obs}')
    ax.set_title('100 keV electrons')
    ax.set_ylabel('peak flux')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for obs in ['SOLO', 'STEREO-A', 'SOHO', 'BepiColombo']:  # 'PSP' removed bc. scaled count rates only
        df_obs = df[df.Observer == obs]
        nindex = np.where(df_obs['ELECTRONS 1 MeV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_obs['ELECTRONS 1 MeV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_obs['ELECTRONS 1 MeV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        ax.plot(dates, fluxes, 'o', label=f'{obs}')
    ax.set_title('1 MeV electrons')
    ax.set_ylabel('peak flux')
    ax.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for obs in ['SOLO', 'STEREO-A', 'SOHO', 'BepiColombo', 'PSP']:
        df_obs = df[df.Observer == obs]
        nindex = np.where(df_obs['PROTONS 25-40 MeV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_obs['PROTONS 25-40 MeV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_obs['PROTONS 25-40 MeV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        ax.plot(dates, fluxes, 'o', label=f'{obs}')
    ax.set_title('> 25 MeV protons')
    ax.set_ylabel('peak flux')
    ax.legend()
    plt.show()

plot_peak_vs_time2 = False
if plot_peak_vs_time2:
    event_number = list(dict.fromkeys(df['Event number'].values))

    fig, ax = plt.subplots(figsize=(10, 6))
    for event in event_number:
        df_event = df[df['Event number'] == event]
        nindex = np.where(df_event['ELECTRONS 100 keV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_event['ELECTRONS 100 keV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_event['ELECTRONS 100 keV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        if len(fluxes) > 0:
            ax.bar(dates[np.argmax(fluxes)], fluxes[np.argmax(fluxes)], width=3, color='k', label=f'{event}')
    ax.set_title('100 keV electrons')
    ax.set_xlim(dt.datetime(2020, 11, 1), dt.datetime(2022, 6, 1))
    ax.set_yscale('log')
    ax.set_ylabel('Peak Flux / (s cm² sr MeV)')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for event in event_number:
        df_event = df[df['Event number'] == event]
        nindex = np.where(df_event['ELECTRONS 1 MeV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_event['ELECTRONS 1 MeV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_event['ELECTRONS 1 MeV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        if len(fluxes) > 0:
            ax.bar(dates[np.argmax(fluxes)], fluxes[np.argmax(fluxes)], width=3, color='k', label=f'{event}')
    ax.set_title('1 MeV electrons')
    ax.set_xlim(dt.datetime(2020, 11, 1), dt.datetime(2022, 6, 1))
    ax.set_yscale('log')
    ax.set_ylabel('Peak Flux / (s cm² sr MeV)')
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    for event in event_number:
        df_event = df[df['Event number'] == event]
        nindex = np.where(df_event['PROTONS 25-40 MeV Peak date (yyyy-mm-dd)'].values.astype(str) != 'nan')[0]
        dates = [dt.datetime.strptime(df_event['PROTONS 25-40 MeV Peak date (yyyy-mm-dd)'].iloc[nindex].values[i], '%Y-%m-%d') for i in range(len(nindex))]
        fluxes = [float(df_event['PROTONS 25-40 MeV Peak flux'].iloc[nindex].values[i].replace(',', '.')) for i in range(len(nindex))]
        if len(fluxes) > 0:
            ax.bar(dates[np.argmax(fluxes)], fluxes[np.argmax(fluxes)], width=3, color='k', label=f'{event}')
    ax.set_title('>25 MeV protons')
    if lower_proton:
        ax.set_title('14 MeV protons')
    ax.set_xlim(dt.datetime(2020, 11, 1), dt.datetime(2022, 6, 1))
    ax.set_yscale('log')
    ax.set_ylabel('Peak Flux / (s cm² sr MeV)')
    plt.show()
"""
END PLOT PEAK FLUXES VS DATES
"""


if mode == 'regular':
    dates = pd.date_range(start=first_date, end=last_date, freq=plot_period)
if mode == 'events':
    dates = all_onset_dates_first
# for startdate in tqdm(dates.to_pydatetime()):  # not in use any more
for i in tqdm(range(len(dates))):  # standard
# for i in tqdm(range(7, len(dates))):
# for i in tqdm([3, 25, 27, 30, 32, 34, 41, 43, 48]):  # replot some events which automatically are replaced with day+1 plots
# for i in tqdm([3, 25, 27, 30, 32, 34, 41, 43, 48]):  # replot some events which automatically are replaced with day+1 plots
    # i=24
    print(i, dates[i])
    if mode == 'regular':
        startdate = dates[i].to_pydatetime()
    if mode == 'events':
        startdate = dt.datetime.fromisoformat(dates[i].isoformat()) - pd.Timedelta('5h')
        plot_period = ('48h')
    enddate = startdate + pd.Timedelta(plot_period)
    outfile = f'{outpath}{os.sep}multi_sc_plot_{startdate.date()}_{plot_period}_{averaging}-av.png'
    if lower_proton:
        outfile = f'{outpath}{os.sep}multi_sc_plot_{startdate.date()}_{plot_period}_{averaging}-av_p-mod.png'

    if Bepi:
        # av_bepi = 10
        sixs_resample = averaging  # '10min'
        sixs_ch_e1 = [5, 6]
        sixs_ch_e100 = 2
        sixs_ch_p = [8, 9]  # we want 'P8'-'P9' averaged
        if lower_proton:
            sixs_ch_p = [7]
        sixs_side = 2
        sixs_color = 'orange'  # seaborn_colorblind[4]  # orange?
        sixs_path = '/home/gieseler/uni/bepi/data/bc_mpo_sixs/data_csv/cruise/sixs-p/raw'
    if SOHO:
        soho_ephin_color = 'k'
        soho_erne_color = 'k'  # seaborn_colorblind[5]  # 'green'
        # av_soho = av
        soho_erne_resample = averaging  # '30min'
        soho_ephin_resample = averaging  # '30min'
        soho_path = '/home/gieseler/uni/soho/data/'
        if erne:
            erne_p_ch = [3, 4]  # [0]  # [4,5]  # 2
            if lower_proton:
                erne_p_ch = [0]
        if ephin_e:
            ephin_ch_e1 = 'E1300'
            # ephin_e_intercal = 1/14.
        # if ephin_p:
        #     ephin_ch_p = 'p25'
    if SOLO:
        solo_ept_color = seaborn_colorblind[5]  # 'blue'
        solo_het_color = seaborn_colorblind[0]  # 'blue' # seaborn_colorblind[1]
        sector = 'sun'
        ept_ch_e100 = [14, 18]  # [25]
        het_ch_e1 = [0, 1]
        ept_ch_p = [50, 56]  # 50-56
        het_ch_p = [19, 24]  # [18, 19]
        if lower_proton:
            het_ch_p = [11, 12]
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
        if lower_proton:
            st_het_ch_p = [0]
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
        wind_3dp_threshold = None  # 1e3/1e6  # None
        wind_path = '/home/gieseler/uni/wind/data/'
    if PSP:
        psp_epilo_ch_e100 = [4, 5]  # cf. psp_epilo_energies
        psp_het_ch_e = [3, 10]  # cf. psp_het_energies
        psp_het_ch_p = [8, 9]  # cf. psp_het_energies
        if lower_proton:
            psp_het_ch_p = [4]
        psp_epilo_channel = 'F'
        psp_epilo_viewing = 3  # 3="sun", 7="antisun"
        psp_epilo_threshold = None  # 1e2  # None
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
            # ephin = eph_rl2_loader(startdate.year, startdate.timetuple().tm_yday, doy2=enddate.timetuple().tm_yday, av=av_soho)
            soho_ephin, ephin_energies = soho_load(dataset="SOHO_COSTEP-EPHIN_L2-1MIN",
                                                   startdate=startdate,
                                                   enddate=enddate,
                                                   path=soho_path,
                                                   resample=soho_ephin_resample,
                                                   pos_timestamp='center')

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
        # sixs_e, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_e, av=sixs_resample)
        # sixs_p, sixs_chstrings = bepi_sixs_loader(startdate.year, startdate.month, startdate.day, sixs_side_p, av=sixs_resample)
        # sixs_ch_e = sixs_ch_e100
        sixs_df, sixs_meta = bepi_sixs_load(startdate=startdate,
                                            enddate=enddate,
                                            side=sixs_side,
                                            path=sixs_path)
        if len(sixs_df) > 0:
            sixs_df_p = sixs_df[[f"P{i}" for i in range(1, 10)]]
            sixs_df_e = sixs_df[[f"E{i}" for i in range(1, 8)]]

    """
    ########## AVERAGE ENERGY CHANNELS ##########
    #############################################
    """
    if SOLO:
        if len(ept_e) > 0:
            if plot_e_100:
                df_ept_e = ept_e['Electron_Flux']
                ept_en_str_e = ept_energies['Electron_Bins_Text'][:]

                # if ept_use_corr_e:
                #     print('correcting e')
                #     ion_cont_corr_matrix = np.loadtxt('EPT_ion_contamination_flux_paco.dat')
                #     Electron_Flux_cont = np.zeros(np.shape(df_ept_e))
                #     for tt in range(len(df_ept_e)):
                #         Electron_Flux_cont[tt, :] = np.matmul(ion_cont_corr_matrix, df_ept_p.values[tt, :])
                #     df_ept_e = df_ept_e - Electron_Flux_cont

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
    if Bepi:
        if len(sixs_df) > 0:
            # 1 MeV electrons:
            sixs_df_e1, sixs_e1_en_channel_string = calc_av_en_flux_sixs(sixs_df_e, sixs_meta, sixs_ch_e1, 'e')
            # >25 MeV protons:
            sixs_df_p25, sixs_p25_en_channel_string = calc_av_en_flux_sixs(sixs_df_p, sixs_meta, sixs_ch_p, 'p')
            # 100 keV electrons withouth averaging:
            sixs_df_e100 = sixs_df_e[f'E{sixs_ch_e100}']
            sixs_e100_en_channel_string = sixs_meta['Energy_Bin_str'][f'E{sixs_ch_e100}']

            if isinstance(sixs_resample, str):
                sixs_df_e100 = resample_df(sixs_df_e100, sixs_resample)
                sixs_df_e1 = resample_df(sixs_df_e1, sixs_resample)
                sixs_df_p25 = resample_df(sixs_df_p25, sixs_resample)

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
                #     psp_epilo_ch_e100 = psp_epilo_ch_e100[0]
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

        # plot flare times with arrows on top
        if mode == 'events':
            trans = blended_transform_factory(x_transform=ax.transData, y_transform=ax.transAxes)
            ind = np.where((np.array(df_flare_times) < enddate) & (np.array(df_flare_times) > startdate))
            [ax.annotate('',
                         xy=[mdates.date2num(i), 1.0], xycoords=trans,
                         xytext=[mdates.date2num(i), 1.07], textcoords=trans,
                         arrowprops=dict(arrowstyle="->", lw=2)) for i in np.array(df_flare_times)[ind]]

        if PSP:
            if len(psp_epilo) > 0:
                ax.plot(df_psp_epilo_e.index, df_psp_epilo_e*100, color=psp_het_color, linewidth=linewidth,
                        label='PSP '+r"$\bf{(count\ rate\ *100)}$"+'\nISOIS-EPILO '+psp_epilo_chstring_e+f'\nF (W{psp_epilo_viewing})',
                        drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=psp_het_color) for i in df_psp_onset_e100]
                [ax.axvline(i, lw=2, ls=':', color=psp_het_color) for i in df_psp_peak_e100]

        if Bepi:
            # ax.plot(sixs_e.index, sixs_e[sixs_ch_e], color='orange', linewidth=linewidth, label='BepiColombo\nSIXS '+sixs_chstrings[sixs_ch_e]+f'\nside {sixs_side_e}', drawstyle='steps-mid')
            if len(sixs_df) > 0:
                ax.plot(sixs_df_e100.index, sixs_df_e100, color=sixs_color, linewidth=linewidth, label='BepiColombo/SIXS '+sixs_e100_en_channel_string+f' side {sixs_side}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=sixs_color) for i in df_bepi_onset_e100]
                [ax.axvline(i, lw=2, ls=':', color=sixs_color) for i in df_bepi_peak_e100]
        if SOLO:
            if ept and (len(ept_e) > 0):
                flux_ept = df_ept_e.values
                try:
                    for ch in ept_ch_e100:
                        ax.plot(df_ept_e.index.values, flux_ept[:, ch], linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_en_str_e[ch, 0]+f'\n{sector}', drawstyle='steps-mid')
                except IndexError:
                    ax.plot(df_ept_e.index.values, flux_ept, linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_chstring_e+f'\n{sector}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=solo_ept_color) for i in df_solo_onset_e100]
                [ax.axvline(i, lw=2, ls=':', color=solo_ept_color) for i in df_solo_peak_e100]
        if STEREO:
            if sept_e:
                if type(sept_ch_e) == list and len(sta_sept_avg_e) > 0:
                    ax.plot(sta_sept_avg_e.index, sta_sept_avg_e, color=stereo_sept_color, linewidth=linewidth,
                            label='STEREO/SEPT '+sept_chstring_e+f' {sector}', drawstyle='steps-mid')
                elif type(sept_ch_e) == int:
                    ax.plot(sta_sept_df_e.index, sta_sept_df_e[f'ch_{sept_ch_e}'], color=stereo_sept_color,
                            linewidth=linewidth, label='STEREO/SEPT '+sta_sept_dict_e.loc[sept_ch_e]['ch_strings']+f' {sector}', drawstyle='steps-mid')
                if plot_times:
                    [ax.axvline(i, lw=2, color=stereo_sept_color) for i in df_sta_onset_e100]
                    [ax.axvline(i, lw=2, ls=':', color=stereo_sept_color) for i in df_sta_peak_e100]

        # if SOHO:
            # if ephin_e:
            #     ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color,
            #             linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}',
            #             drawstyle='steps-mid')
        if WIND:
            if len(wind3dp_e_df) > 0:
                # multiply by 1e6 to get per MeV
                ax.plot(wind3dp_e_df.index, wind3dp_e_df[f'FLUX_{wind3dp_ch_e}']*1e6, color=wind_color, linewidth=linewidth, label='Wind/3DP '+wind3dp_e_meta['channels_dict_df']['Bins_Text'][wind3dp_ch_e], drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=wind_color) for i in df_wind_onset_e100]
                [ax.axvline(i, lw=2, ls=':', color=wind_color) for i in df_wind_peak_e100]

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
            if plot_times:
                [ax.axvline(i, lw=2, color=psp_het_color) for i in df_psp_onset_e1000]
                [ax.axvline(i, lw=2, ls=':', color=psp_het_color) for i in df_psp_peak_e1000]
        if Bepi:
            # ax.plot(sixs_e.index, sixs_e[sixs_ch_e100], color='orange', linewidth=linewidth,
            #         label='Bepi/SIXS '+sixs_chstrings[sixs_ch_e100]+f' side {sixs_side_e}', drawstyle='steps-mid')
            if len(sixs_df) > 0:
                ax.plot(sixs_df_e1.index, sixs_df_e1, color=sixs_color, linewidth=linewidth,
                        label='BepiColombo/SIXS '+sixs_e1_en_channel_string+f' side {sixs_side}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=sixs_color) for i in df_bepi_onset_e1000]
                [ax.axvline(i, lw=2, ls=':', color=sixs_color) for i in df_bepi_peak_e1000]
        if SOLO:
            if het and (len(het_e) > 0):
                ax.plot(df_het_e.index.values, df_het_e.flux, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_e+f' {sector}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=solo_het_color) for i in df_solo_onset_e1000]
                [ax.axvline(i, lw=2, ls=':', color=solo_het_color) for i in df_solo_peak_e1000]
        if STEREO:
            if stereo_het:
                if len(sta_het_avg_e) > 0:
                    ax.plot(sta_het_avg_e.index, sta_het_avg_e, color=stereo_het_color, linewidth=linewidth,
                            label='STEREO/HET '+st_het_chstring_e, drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=stereo_het_color) for i in df_sta_onset_e1000]
                [ax.axvline(i, lw=2, ls=':', color=stereo_het_color) for i in df_sta_peak_e1000]
        if SOHO:
            if ephin_e:
                # ax.plot(ephin['date'], ephin[ephin_ch_e][0]*ephin_e_intercal, color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_e][1]+f'/{ephin_e_intercal}', drawstyle='steps-mid')
                if len(soho_ephin) > 0:
                    ax.plot(soho_ephin.index, soho_ephin[ephin_ch_e1], color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin_energies[ephin_ch_e1], drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=soho_ephin_color) for i in df_soho_onset_e1000]
                [ax.axvline(i, lw=2, ls=':', color=soho_ephin_color) for i in df_soho_peak_e1000]

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
                        label='PSP '+'\nISOIS-EPIHI-HET '+psp_het_chstring_p+'\nA (sun)',
                        drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=psp_het_color) for i in df_psp_onset_p]
                [ax.axvline(i, lw=2, ls=':', color=psp_het_color) for i in df_psp_peak_p]
        if Bepi:
            # ax.plot(sixs_p.index, sixs_p[sixs_ch_p], color='orange', linewidth=linewidth, label='BepiColombo/SIXS '+sixs_chstrings[sixs_ch_p]+f' side {sixs_side_p}', drawstyle='steps-mid')
            if len(sixs_df) > 0:
                ax.plot(sixs_df_p25.index, sixs_df_p25, color=sixs_color, linewidth=linewidth, label='BepiColombo/SIXS '+sixs_p25_en_channel_string+f' side {sixs_side}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=sixs_color) for i in df_bepi_onset_p]
                [ax.axvline(i, lw=2, ls=':', color=sixs_color) for i in df_bepi_peak_p]
        if SOLO:
            if het and (len(ept_e) > 0):
                ax.plot(df_het_p.index, df_het_p, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_p+f' {sector}', drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=solo_het_color) for i in df_solo_onset_p]
                [ax.axvline(i, lw=2, ls=':', color=solo_het_color) for i in df_solo_peak_p]
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
            if plot_times:
                [ax.axvline(i, lw=2, color=stereo_het_color) for i in df_sta_onset_p]
                [ax.axvline(i, lw=2, ls=':', color=stereo_het_color) for i in df_sta_peak_p]
        if SOHO:
            if erne:
                if type(erne_p_ch) == list and len(soho_erne_avg_p) > 0:
                    ax.plot(soho_erne_avg_p.index, soho_erne_avg_p, color=soho_erne_color, linewidth=linewidth, label='SOHO/ERNE/HED '+soho_erne_chstring_p, drawstyle='steps-mid')
                elif type(erne_p_ch) == int:
                    if len(soho_erne) > 0:
                        ax.plot(soho_erne.index, soho_erne[f'PH_{erne_p_ch}'], color=soho_erne_color, linewidth=linewidth, label='SOHO/ERNE/HED '+erne_chstring[erne_p_ch], drawstyle='steps-mid')
            # if ephin_p:
            #     ax.plot(ephin['date'], ephin[ephin_ch_p][0], color=soho_ephin_color, linewidth=linewidth, label='SOHO/EPHIN '+ephin[ephin_ch_p][1], drawstyle='steps-mid')
            if plot_times:
                [ax.axvline(i, lw=2, color=soho_erne_color) for i in df_soho_onset_p]
                [ax.axvline(i, lw=2, ls=':', color=soho_erne_color) for i in df_soho_peak_p]
        # if WIND:
            # multiply by 1e6 to get per MeV
        #    ax.plot(wind3dp_p_df.index, wind3dp_p_df[f'FLUX_{wind3dp_ch_p}']*1e6, color=wind_color, linewidth=linewidth, label='Wind\n3DP '+str(round(wind3dp_p_df[f'ENERGY_{wind3dp_ch_p}'].mean()/1000., 2)) + ' keV', drawstyle='steps-mid')
        # ax.set_ylim(2.05e-5, 4.8e0)
        # ax.set_ylim(0.00033920545179055416, 249.08996960298424)
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='>25 MeV Protons/Ions')
        if lower_proton:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='14 MeV Protons/Ions')
        axnum = axnum+1
    # pos = get_horizons_coord('Solar Orbiter', startdate, 'id')
    # dist = np.round(pos.radius.value, 2)
    # fig.suptitle(f'Solar Orbiter/EPD {sector} (R={dist} au)')
    ax.set_xlim(startdate, enddate)
    # ax.set_xlim(dt.datetime(2021, 10, 9, 6, 0), dt.datetime(2021, 10, 9, 11, 0))
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
