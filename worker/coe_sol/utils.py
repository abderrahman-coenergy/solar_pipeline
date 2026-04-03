from calendar import monthrange
from logging import warning
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad
from scipy.optimize import fmin
from tqdm import tqdm
from parse import parse
import json
from math import cos, sin, tan, atan, pi
import warnings
try:
    import sg2
except ImportError:
    print('Could not import module sg2. Install with "pip install sg2 -f https://pip.oie-lab.net/python/"')
    exit(1)

def get_errors_kd(settings, variations, kd_list):
    errors = np.array([np.sum(
                np.array([np.abs(variations[k][pyr - 1] - settings.measures[f'pyrano-fit-{pyr}_value']) for pyr in range(1, settings.n_fit + 1)])
            ,axis=0) for k in range(len(kd_list))])
    errors = errors.transpose()
    opt_index = [np.argmin(errors[i]) for i in range(len(errors))]
    return errors, opt_index

def convert_to_watt(intensities: np.ndarray, hours: np.ndarray) -> np.ndarray:
    """
    Convert the intensities to watts.
    Intensities should be in the form of an array with one column per pyranometer.
    """
    warnings.warn("Deprectated: this function uses the old data format which will not be used in future versions.", DeprecationWarning)
    for k in range(int(hours[0, 0]) ,int(hours[-1][0])+1): #Tous les jours de la semaine prise
        converted = intensities[ hours[:,0] == float(k) ,: ] #On prend que le tableau du jour correspondant
        if converted.shape[0] != 0:
            converted= (converted - np.amin(converted, axis=0 ))*0 #En W/m2
            intensities[ hours[:,0] == k ,: ] = converted
    return intensities


def convert_data_to_watt(intensities: np.ndarray, time: list[str])->np.ndarray: # TODO: etalonnage des instruments
    minimum = np.min(intensities)
    intensities = (intensities - minimum) * 100
    return intensities

def moving_average(array: np.array, intervalle: int):
    return np.convolve(array, np.ones(intervalle), 'valid') / intervalle

def get_day_rank(hours: np.ndarray, month: int, months: list[int])->list[float]:
    """
    Return the day rank of the day of the year.
    """
    return hours[:, 0] + np.sum(months[:month-1]) + (hours[:, 1])/(3600*24)


def get_omega_and_declinaison(
    month: int,
    measures: np.ndarray,
    hours: np.ndarray,
    longitude:  float,
    latitude: float,
    year: int = 2022
) -> tuple[float, float]:
    months = [monthrange(year, i)[1] for i in range(1, 13)]

    J = get_day_rank(hours, month, months)
    declinaison  = np.deg2rad(23.45*np.sin(2*np.pi*(J+284)/365))
    solar_time = hours[:, 1]/3600 - 0 + longitude*4/60 + (- 7.53* np.cos( 2*np.pi*(J-81)/365) - 1.5 * np.sin( 2*np.pi*(J-81)/365) + 9.87 * np.sin(2* 2*np.pi*(J-81)/365))/60
    omega = 180*(12-solar_time)/12 * 2*np.pi/360 #solar angle (rad)
    # below we truncate nighttime from the simulation
    h0 = np.arccos(-np.tan((declinaison)) * np.tan(np.deg2rad(latitude)))
    mask = hours[:, 1] < 3600 * (12 + (24 * h0 /(2 * np.pi) - 1))

    m = [measures[i][mask] for i in range(len(measures))]
    declinaison = declinaison[mask]
    omega = omega[mask]
    solar_time = solar_time[mask]
    h0 = h0[mask]
    h = hours[mask]

    mask = h[:, 1] > 3600 * (12 - (24 * h0 /(2 * np.pi) - 1))
    m = [m[i][mask] for i in range(len(m))]
    declinaison = declinaison[mask]
    omega = omega[mask]
    solar_time = solar_time[mask]
    h = h[mask]

    return(omega, declinaison, solar_time, h, m)

def hour_to_str(hour: tuple[float, float], year=2022, month=1)->str:
    day = int(hour[0])
    hours = int(hour[1] // 3600)
    minutes = int((hour[1] - hours * 3600) // 60)
    seconds = (hour[1] - hours * 3600 - minutes * 60)
    if hours >= 24:
        day += 1
        hours -= 24
    month_len = monthrange(year, month)[1]
    if day > month_len:
        day -= month_len
        month += 1
    if month > 12:
        month -= 12
        year += 1
    return f'{year}-{month:02d}-{day:02d}T{hours:02d}:{minutes:02d}:{seconds:06.3f}'

def format_hour_no_24(hours: str):
    """
    hour is expected in the format YYYY-MM-DDTHH:mm:ss.sss
    """
    new_hours = []
    ###MODIF
    hours = list(hours)
    ###FIN DE MODIF
    for hour in hours:
        parsed = parse("{}-{}-{} {}:{}:{}", hour)
        if parsed == None:
            raise ValueError(f"Invalid time format: {hour}")
        year = int(parsed[0])
        month = int(parsed[1])
        day = int(parsed[2])
        h = int(parsed[3])
        minutes = int(parsed[4])
        seconds = float(parsed[5])
        if h >= 24:
            day += 1
            h -= 24
        month_len = monthrange(year, month)[1]
        if day > month_len:
            day -= month_len
            month += 1
        if month > 12:
            month -= 12
            year += 1
        new_hours.append(f'{year}-{month:02d}-{day:02d}T{h:02d}:{minutes:02d}:{seconds:06.3f}')
    return new_hours


def restore_night(time_str: list[str], model, curves: list[list[float]]):
    npcurves = np.array(curves)
    tstr_format = format_hour_no_24(time_str)
    curves_with_night = np.zeros((np.shape(npcurves)[0], len(tstr_format)))
    for i in range(len(model.time)):
        if model.time[i] in tstr_format:
            idx = tstr_format.index(model.time[i])
            curves_with_night[:, idx] = npcurves[:, i]
    return curves_with_night


def get_solar_parameters(hours, longitude, latitude, elevation, measures, month=1, year=2022, bonus_curves_to_cut=None):
    hours_str = [np.datetime64(hour_to_str(h, month=month)) for h in hours]
    print(f'month = {month}')
    longitude: float = 2.153139
    latitude: float = 48.503565
    elevation: float = 63.7
    positions = sg2.sun_position(
        [[longitude, latitude, elevation]],
        hours_str,
        ['topoc.gamma_S0', 'topoc.omega', 'topoc.delta', 'topoc.alpha_S', ]
    )
    gamma_s = positions.topoc.gamma_S0[0]
    omega = np.mod(positions.topoc.omega[0], 2 * np.pi)
    declinaison = positions.topoc.delta[0]
    azimut = positions.topoc.alpha_S[0]
    # solar_time = -(omega * 12 / 180 - 12)

    months = [monthrange(year, i)[1] for i in range(1, 13)]

    J = get_day_rank(hours, month, months)
    solar_time = hours[:, 1]/3600 - 0 + longitude*4/60 + (- 7.53* np.cos( 2*np.pi*(J-81)/365) - 1.5 * np.sin( 2*np.pi*(J-81)/365) + 9.87 * np.sin(2* 2*np.pi*(J-81)/365))/60
    omega = 180*(12-solar_time)/12 * 2*np.pi/360 #solar angle (rad)

    # below we truncate nighttime from the simulation
    h0 = np.arccos(-np.tan((declinaison)) * np.tan(np.deg2rad(latitude)))
    mask = hours[:, 1] < 3600 * (12 + (24 * h0 /(2 * np.pi) - 1))

    m = [measures[i][mask] for i in range(len(measures))]
    declinaison = declinaison[mask]
    gamma_s = gamma_s[mask]
    azimut = azimut[mask]
    omega = omega[mask]
    solar_time = solar_time[mask]
    h0 = h0[mask]
    h = hours[mask]
    bonus = []
    try:
        for c in bonus_curves_to_cut:
            bonus.append(c[mask])
        bonus_curves_to_cut = bonus
    except:
        pass # nothing to deal with

    mask = h[:, 1] > 3600 * (12 - (24 * h0 /(2 * np.pi) - 1))
    m = [m[i][mask] for i in range(len(m))]
    declinaison = declinaison[mask]
    gamma_s = gamma_s[mask]
    azimut = azimut[mask]
    omega = omega[mask]
    solar_time = solar_time[mask]
    h = h[mask]
    bonus = []
    try:
        for c in bonus_curves_to_cut:
            bonus.append(c[mask])
    except:
        pass
    return omega, declinaison, gamma_s, azimut, solar_time, h, m, bonus

def get_sun_elevation(
    _declinaison: np.ndarray,
    _latitude_rad: float,
    _omega: np.ndarray,
    hours
) -> np.ndarray:
    _gamma_s = np.copy(np.arcsin(np.sin(_declinaison)*np.sin(_latitude_rad) + np.cos(_declinaison)*np.cos(_latitude_rad)*np.cos(_omega)))
    # _gamma_s[_gamma_s == 0] = 0.0001 # to prevent division by zero
    _I = np.zeros((hours.shape[0],))
    _I[:] = 1367
    return(_gamma_s)

def greater_than_zero(var):
    var[var <= 0] = 0
    var[var > 0] = 1
    return var

# def get_Rb_horizontal(gamma_s: np.array, pyrano_origin: int, cos_theta: np.ndarray):
#     rb = cos_theta[pyrano_origin] / np.maximum(0.02, np.sin(gamma_s))
#     rb[rb > 1] = 1
#     rb[rb <= 0] = 0.0001 # ! to prevent division by zero. Check if this introduces any problem
#     return rb

def read_sim_info(folder):
    with open(os.path.join(folder, "sim_info.txt")) as f:
        s = 'month: {}\nyear: {}\nlongitude: {}\nlatitude: {}\norigins: {}\nfit: {}'
        month, year, longitude, latitude, origins, fit = parse(s, f.read())
        return int(month), int(year), float(longitude), float(latitude), json.loads(origins), json.loads(fit)


def get_Rb_horizontal(gamma_s: np.array, pyrano_origin: int, cos_theta: np.ndarray):
    rb = np.abs(cos_theta[pyrano_origin] / np.maximum(0.02, np.sin(gamma_s)))
    rb[rb < 0] = 0.002
    return rb

def get_Rcs(cos_theta: np.array, gamma_s: np.array, pyr):
    Rcs = cos_theta[pyr] / np.maximum(0.087, np.sin(gamma_s))
    Rcs[np.array(gamma_s) < 0] = 0
    Rcs[np.array(cos_theta[pyr]) < 0] = 0
    return Rcs

def get_Rb(gamma_s: np.array ,num_pyrano: int, cos_theta: np.ndarray):
    rb = np.abs((cos_theta[num_pyrano]/ np.maximum( 0.02, np.sin(gamma_s)) * greater_than_zero(gamma_s) * greater_than_zero(cos_theta[num_pyrano])))
    if (type(rb) == float):
        rb = np.array([rb] * len(gamma_s))
    rb[rb > 1] = 1
    rb[rb <= 0] = 0.002 # to avoid zero division error
    return rb

def get_Rb_from_angle(alpha, beta, gamma_s, sun_azimut):
    cos_theta = np.sin(gamma_s) * np.cos(beta) + np.sin(beta) * np.cos(gamma_s) * np.cos(sun_azimut - alpha)
    return cos_theta / np.maximum(0.02, np.sin(gamma_s))

def get_sun_azimut(_declinaison: np.ndarray,
                    hours: np.ndarray,
                    _gamma_s: np.ndarray,
                    _omega: np.ndarray,
                    _latitude_rad: np.ndarray) -> np.ndarray:
    _phi_s = np.zeros( (hours.shape[0], ) )

    cos_azimut_soleil =  (np.cos(_declinaison)* np.cos(_omega) * np.sin(_latitude_rad) - np.sin(_declinaison) * np.cos(_latitude_rad) )/np.cos(_gamma_s)
    sin_azimut_soleil = (np.cos(_declinaison) * np.sin(_omega) / np.cos(_gamma_s) )

    _phi_s[ sin_azimut_soleil > 0 ] = np.arccos(cos_azimut_soleil[sin_azimut_soleil > 0] )
    _phi_s[ np.logical_and(cos_azimut_soleil > 0, sin_azimut_soleil <= 0) ] = np.arcsin(sin_azimut_soleil[ np.logical_and(cos_azimut_soleil > 0, sin_azimut_soleil <= 0) ] )
    _phi_s[ np.logical_and(cos_azimut_soleil <= 0, sin_azimut_soleil <= 0) ] = np.pi +  np.arccos( -cos_azimut_soleil[ np.logical_and(cos_azimut_soleil <= 0, sin_azimut_soleil <= 0) ] )
    _phi_s = np.pi - _phi_s
    return(_phi_s)

def get_incidence_angle(shape: tuple[int, int],
                        _incli_pyrano: np.ndarray,
                        _declinaison: np.ndarray,
                        _latitude_rad: float,
                        _omega: np.ndarray)->np.ndarray:
        _cos_theta = np.zeros( (shape[0], shape[1]) )

        for k in range(shape[0]):
            inclinaison = _incli_pyrano[k,1] #en deg
            azimut_local = _incli_pyrano[k,2]
            inclinaison_rad = inclinaison*2*np.pi/360 #en rad
            azimut_local_rad = azimut_local*2*np.pi/360 #en rad
            _cos_theta[k] = np.sin(_declinaison)*np.sin(_latitude_rad)*np.cos(inclinaison_rad) - np.sin(_declinaison) * np.cos(_latitude_rad) * np.sin(inclinaison_rad)*np.cos(azimut_local_rad) + np.cos(_declinaison)*np.cos(_latitude_rad)*np.cos(inclinaison_rad)*np.cos(_omega) + np.cos(_declinaison)*np.sin(_latitude_rad)*np.sin(inclinaison_rad)* np.cos(azimut_local_rad)*np.cos(_omega)+np.cos(_declinaison)*np.sin(inclinaison_rad)*np.sin(azimut_local_rad)*np.sin(_omega)
        return(_cos_theta)

def get_horizon_elevation(horizons, num_pyrano, azimut)->np.array:
    """
    :param horizons: the array containing the elevation of the horizon at 360 degrees for every pyranometer. Use the get_horizon.py script to generate it.
    :param num_pyrano: the index of the pyranometer for which we want the horizon elevation.
    :param azimut: the array of the azimut of the sun during the time interval we are considering. (rad).
    :return: the elevation of the horizon at 360 degrees for the local azimut.
    """
    azimut = azimut * 360/(2*np.pi)
    azimut = np.floor(azimut ).astype(int) # ! we bring it to the sampling step used, ie 1 degree here. TO CHANGE IF IT CHANGES
    return(horizons[num_pyrano, azimut])

def running_mean(x, N):
    if N % 2 == 0:
        N += 1
    cumsum: np.array = np.cumsum(np.insert(x, 0, 0))
    averaged: np.array =  (cumsum[N:] - cumsum[:-N]) / float(N)
    copy = x.copy()
    print(len(copy[N - 1:len(copy)]))
    copy[N//2 - 1:len(x) - N//2 - 1] = averaged
    return copy


def get_Riso(incli_pyrano: np.array, horizons, num_pyrano: int):
    #Remarque de philippe, prendre par rapport au Nord, et l'inclinaison opposé de pi/2
    warnings.warn("get_Riso is deprecated", DeprecationWarning)
    inclinaison_f = np.pi/2 - incli_pyrano[num_pyrano][1]*2*np.pi/360 # rad
    azimut_local_f = np.pi - incli_pyrano[num_pyrano][2]*2*np.pi/360
    def theta_plus(phi): # TODO: this introduces a discontinuity. Try to remove it
        if np.cos(phi - azimut_local_f) < 0:
            term = np.cos(phi-azimut_local_f) * np.tan(inclinaison_f)
            if term == 0:
                return np.pi / 2
            return(np.arctan(-(term)**(-1)))
        else :
            return(np.pi/2)
    Z = lambda phi: np.minimum(np.pi/2 - get_horizon_elevation(horizons, num_pyrano, phi), theta_plus(phi))
    F = lambda phi: np.minimum(np.cos(phi - azimut_local_f)*(Z(phi) - np.sin(Z(phi)) * np.cos(Z(phi))), 0)
    res = (np.cos(inclinaison_f)/(2*np.pi) * integrate.quad(F,0, 2*np.pi, limit = 100 )[0] + np.sin(inclinaison_f)/(2*np.pi) * integrate.quad(
        F,0,2*np.pi, limit = 100
    )[0])
    return res if res > 0 else 0


def calc_Riso(alpha: float, beta: float, horizons: np.array):
    def theta(phi):
        c = np.cos(phi - alpha)
        if c > 0:
            return np.pi/2
        b = np.tan(beta)
        if c == 0 or b == 0:
            return 123453564
        return np.arctan(-(c * b)**(-1))
    H = lambda phi: horizons[int(np.rad2deg(phi))]
    Z = lambda phi: min(np.pi/2 - H(phi), theta(phi))
    R1 = lambda phi: np.sin(Z(phi)) ** 2
    R2 = lambda phi: np.cos(phi - alpha) * (Z(phi) - np.sin(Z(phi)) * np.cos(Z(phi)))
    Riso = np.cos(beta) / (np.pi * 2) * quad(R1, 0, np.pi * 2)[0] \
        + np.sin(beta) / (np.pi * 2) * quad(R2, 0, np.pi * 2)[0]
    return Riso

def get_time(date: str, unit: str="m"):
    units = ["y", "M", "d", "h", "m", "s"]
    assert unit in units, f"Unrecognized unit: {unit}"
    parsed_index = units.index(unit)
    if unit == "s":
        return float(parse("{}-{}-{}T{}:{}:{}", date)[parsed_index])
    return int(parse("{}-{}-{}T{}:{}:{}", date)[parsed_index])


def get_sun_above_horizon(length: int, gamma_s: np.array, horizon_elevations: np.array, num_pyrano: int):
        truth_table = np.zeros(length)
        truth_table[horizon_elevations[num_pyrano] < gamma_s ] = 1
        return(truth_table)

def get_relative_optical_airmass(model):
    a = 0.15
    b = 3.885
    c = 1.253
    return 1/(np.abs(np.sin(model._gamma_s)) + a*(np.abs(model._gamma_s)*model.sun_above_zero*360/(2*np.pi) + b)**(-c))

def solve_dhi(error_evaluator, epsilon, kappa, zeta, Rb, GHI_shadow, Riso, h, rho, fa, fb, I_0, m, pdt):
    f = lambda x: np.abs(np.mean(error_evaluator(epsilon, kappa, zeta, Rb[pdt], GHI_shadow[pdt], Riso, h[pdt], rho, fa, fb[pdt], pdt, m[pdt], x)))
    return fmin(f, x0 = 1, disp=False)[0]

def fill_nans(DHI, error_evaluator,  epsilon, kappa, zeta, Rb, GHI_shadow, Riso, h, rho, fa, fb, I_0, m):
    if (not np.isnan(DHI).any()):
        return DHI
    nan_indexes = []
    for i in range(len(DHI)):
        if (np.isnan(DHI[i])):
            nan_indexes.append(i)
    tmp = [solve_dhi(error_evaluator, epsilon, kappa, zeta, Rb, GHI_shadow, Riso, h, rho, fa, fb, I_0, m, i) for i in tqdm(nan_indexes)]
    j = 0
    for i in nan_indexes:
        DHI[i] = tmp[j]
        j += 1
    return DHI

def show_model(model, fit, pyr, err_lim = 100):
    opt_kd = model.fit(fit)
    pred, _, _, _ = model.project_gti(
                    alpha=np.deg2rad(model.incli[pyr][2]),
                    beta=np.deg2rad(model.incli[pyr][1]),
                    BHI=model.measures[0] * (1 - opt_kd),
                    DHI=model.measures[0] * opt_kd,
                    gamma_s=model.gamma_s,
                    alpha_s=model.alpha_s,
                    TOANI=model.TOANI,
                    elevation=model.elevation,
                    albedo=0.2,
                    use_riso=model.use_riso,
                    Riso=model.Riso[pyr] if model.use_riso else None,
                )
    model.measures[pyr][model.measures[pyr] == 0] = 0.01
    mask = model.measures[pyr] > err_lim
    model.measures[pyr][mask][model.measures[pyr][mask] == 0] = 0.001
    err = np.mean(np.abs(pred[mask] - model.measures[pyr][mask])/model.measures[pyr][mask])

    plt.plot(model.measures[pyr], label=f'{pyr}')
    plt.plot(pred, label=f"pred for {pyr}")
    plt.legend()
    plt.xlabel(f"{model.time[0]} to {model.time[-1]}")
    plt.ylabel("I (W/m2)")
    plt.title(f"GTI mes vs GTI pred (Real data, err={err * 100:.02f}%)")
    plt.show(block=True)
    plt.close()
