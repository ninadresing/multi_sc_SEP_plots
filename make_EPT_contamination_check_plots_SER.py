# make_EPT_contamination_check_plots_SERPENTINE_catalog.py
'''
June 2022
this loops over the mission time and creates 
plots of EPT electrons and corrected electrons 
to check for contamination periods which need to 
be taken into account in the SERPENTINE catalog
'''
import datetime as dt

# make selections
#############################################################
first_date = dt.datetime(2021, 6, 4)
last_date = dt.datetime(2021, 12, 30)
plot_period = '7D'
averaging = '1H'  # None


SOLO = True

# SOLO:
ept = True
het = False
ept_use_corr_e = False  # not included yet!

#############################################################

import sunpy
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator
from soho_loader import soho_load
from solo_epd_loader import epd_load
from stereo_loader import calc_av_en_flux_SEPT, stereo_load
from stereo_loader import calc_av_en_flux_HET as calc_av_en_flux_ST_HET
from wind_3dp_loader import wind3dp_load
import cdflib
import warnings
import astropy.units as u
from cdflib.epochs import CDFepoch
from sunpy import log
from sunpy.net import Fido
from sunpy.net import attrs as a
from sunpy.timeseries import TimeSeries
from sunpy.util.exceptions import warn_user

# omit some warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=sunpy.util.SunpyUserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# manually define seaborn-colorblind colors
seaborn_colorblind =  ['#0072B2', '#009E73', '#D55E00', '#CC79A7', '#F0E442', '#56B4E9']
                        # blue, green, orange, magenta, yello, light blue
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
plot_e_1 = False
plot_p = False
save_fig = True
outpath = '/Users/dresing/Documents/Proposals/SERPENTINE_H2020/Cycle25_Multi-SC_SEP_Event_List/EPT_conta_corr_plots/'

dates = pd.date_range(start=first_date, end=last_date, freq=plot_period)
for startdate in dates.to_pydatetime():
    enddate = startdate + pd.Timedelta(plot_period)
    outfile = f'{outpath}SOLO_EPT_conta_corr_{startdate.date()}_{plot_period}_{averaging}-av.png'

    if SOLO:
        solo_ept_color = seaborn_colorblind[5]  # 'blue'
        solo_het_color = seaborn_colorblind[0]  # 'blue' # seaborn_colorblind[1]
        sector = 'sun'
        ept_ch_e100 = [14,18]  # [25]
        het_ch_e1 = [0,1]
        ept_ch_p = [50, 56]  # 50-56
        het_ch_p = [19,24]  # [18, 19]
        solo_ept_resample = averaging
        solo_het_resample = averaging
        solo_path = 'data/'
    
    # LOAD DATA
    ##################################################################
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
                    print(f'No SOLO/EPT data for {startdate} - {enddate}')
                    ept_e = []
        if het:
            if plot_e_1 or plot_p:
                print('loading solo/het e & p')
                try:
                    het_p, het_e, het_energies = epd_load(sensor='HET', viewing=sector, level=data_product, startdate=sdate, enddate=edate, path=solo_path, autodownload=True)
                except(Exception):
                    print(f'No SOLO/HET data for {startdate} - {enddate}')
                    het_e = []
                    het_p = []
   

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

                                                                 
    ##########################################################################################
    panels = 0
    if plot_e_1:
        panels = panels +1
    if plot_e_100:
        panels = panels +1
    if plot_p:
        panels = panels +1
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

        if SOLO:
            if ept and (len(ept_e) > 0):
                flux_ept = df_ept_e.values
                try:
                    for ch in ept_ch_e100:
                        ax.plot(df_ept_e.index.values, flux_ept[:, ch], linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_en_str_e[ch, 0]+f'\n{sector}', drawstyle='steps-mid')
                except IndexError:
                    ax.plot(df_ept_e.index.values, flux_ept, linewidth=linewidth, color=solo_ept_color, label='SOLO\nEPT '+ept_chstring_e+f'\n{sector}', drawstyle='steps-mid')

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

      
        if SOLO:
            if het and (len(het_e) > 0):
                ax.plot(df_het_e.index.values, df_het_e.flux, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_e+f' {sector}', drawstyle='steps-mid')
    
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
        if SOLO:
            if het and (len(ept_e) > 0):
                ax.plot(df_het_p.index, df_het_p, linewidth=linewidth, color=solo_het_color, label='SOLO/HET '+het_chstring_p+f' {sector}', drawstyle='steps-mid')
        
        ax.set_yscale('log')
        ax.set_ylabel(intensity_label)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='>25 MeV Protons/Ions')
        axnum = axnum+1

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
        print('Saved '+f'{outpath}/multi_sc_{str(startdate)}{species}{averaging}.png')
    else:
        plt.show()