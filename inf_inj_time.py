from sunpy.coordinates.ephemeris import get_horizons_coord
from numpy import sqrt, log, pi
from astropy import constants as const
from astropy import units as u
import sunpy.sun.constants as sconst
import datetime

SOLAR_ROT = sconst.get('sidereal rotation rate').to(u.rad/u.s)

def get_sun_coords(time='now'):
    '''
    Gets the astropy Sun coordinates.

    Args:
        time (datetime.datetime): time at which coordinates are fetched.

    Returns:
        sun coordinates.
    '''

    return get_horizons_coord("Sun", time=time)

def radial_distance_to_sun(spacecraft, time='now'):
    '''
    Gets the 3D radial distance of a spacecraft to the Sun.
    3D here means that it's the real spatial distance and not
    a projection on, say, the solar equatorial plane.

    Args:
        spacecraft (str): spacecraft to look for.
        time (datetime.datetime): time at which to look for.

    Returns:
        astropy units: radial distance.
    '''

    sc_coords = get_horizons_coord(spacecraft, time)

    return sc_coords.separation_3d(get_sun_coords(time=time))

def calc_spiral_length(radial_dist, sw_speed):
    '''
    Calculates the Parker spiral length from the Sun up to a given radial distance.

    Args:
        radial_dist (astropy units): radial distance to the Sun.
        sw_speed (astropy units): solar wind speed.

    Returns:
        astropy units: Parker spiral length.
    '''

    temp_const = ((SOLAR_ROT/sw_speed)*(radial_dist.to(u.km)-const.R_sun)).value 
    sqrt_temp_const = sqrt(temp_const**2 + 1)

    return 0.5*u.rad * (sw_speed/SOLAR_ROT) * (temp_const*sqrt_temp_const + log(temp_const + sqrt_temp_const))

def calc_particle_speed(mass, kinetic_energy):
    '''
    Calculates the relativistic particle speed.

    Args:
        mass (astropy units): mass of the particle.
        kinetic_energy (astropy units): kinetic energy of the particle.

    Returns:
        astropy units: relativistic particle speed.
    '''

    gamma = sqrt(1 - (mass*const.c**2/(kinetic_energy + mass*const.c**2))**2)
    
    return gamma*const.c

def inf_inj_time(spacecraft, onset_time, species, kinetic_energy, sw_speed):
    '''
    Main function of the script.
    Calculates the inferred injection time of a particle (electron or proton) from the Sun
    given a detection time at some spacecraft.

    Args:
        spacecraft (str): name of the spacecraft.
        onset_time (datetime.datetime): time of onset/detection.
        species (str): particle species, 'p' or 'e'.
        kinetic_energy (astropy units): kinetic energy of particle.
                                        If no unit is supplied, is converted
                                        to MeV.
        sw_speed (astropy units): solar wind speed.
                                  If no unit is supplied, is converted
                                  to km/s.

    Returns:
        datetime.datetime: inferred injection time.
    '''

    if not type(kinetic_energy)==u.quantity.Quantity:
        
        kinetic_energy = kinetic_energy * u.MeV

    if not type(sw_speed)==u.quantity.Quantity:

        sw_speed = sw_speed * u.km/u.s

    mass_dict = {
                'p': const.m_p,
                'e': const.m_e
            }

    radial_distance = radial_distance_to_sun(spacecraft, time=onset_time)

    spiral_length = calc_spiral_length(radial_distance, sw_speed)
    particle_speed = calc_particle_speed(mass_dict[species], kinetic_energy)

    travel_time = spiral_length/particle_speed
    travel_time = travel_time.to(u.s)

    return onset_time - datetime.timedelta(seconds=travel_time.value), spiral_length.to(u.AU)
