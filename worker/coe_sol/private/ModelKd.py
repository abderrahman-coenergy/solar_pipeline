import numpy as np
import dataclasses
import pandas as pd
from coe_sol.utils import *
from numba import prange
from datetime import datetime

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

@dataclasses.dataclass
class ModelKdSettings:
    latitude: float = 0
    longitude: float = 0
    elevation: float = 0
    albedo: float = 0.2
    use_riso: bool = False
    # horizons can be provided as:
    # - 1D array of length 360: same horizon for all pyranometers
    # - 2D array with shape (n_pyr, 360): one horizon per pyranometer
    # - dict with keys like 'fit-1', 'dest-1' returning 1D arrays
    horizons: np.array = field(default_factory=lambda: np.zeros(360))
    measures: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_fit: int = 0 # how many fit pyranometers are there
    n_predict: int = 0 # how many destination pyranometers are there

class ModelKd:
    # Constants
    tabF11 = np.array([-0.0083, 0.1299, 0.3297, 0.5682, 0.8730, 1.1326, 1.0602, 0.6777])
    tabF12 = np.array([0.5877, 0.6826, 0.4869, 0.1875, -0.3920, -1.2367, -1.5999, -0.3273])
    tabF13 = np.array([-0.0621, -0.1514, -0.2211, -0.2951, -0.3616, -0.4118, -0.3589, -0.2504])
    tabF21 = np.array([-0.060, -0.019, 0.055, 0.109, 0.226, 0.288, 0.264, 0.156])
    tabF22 = np.array([0.072, 0.066, -0.064, -0.152, -0.462, -0.823, -1.127, -1.377])
    tabF23 = np.array([-0.022, -0.029, -0.026, -0.014, 0.001, 0.056, 0.131, 0.251])
    bin_epsilon_0 = np.array([1, 1.065, 1.230, 1.5, 1.95, 2.8, 4.5, 6.2])
    bin_epsilon_1 = np.array([1.065, 1.230, 1.5, 1.95, 2.8, 4.5, 6.2, np.inf])

    def __init__(self, settings: ModelKdSettings):
        self.settings = settings
        self.time = np.array([np.datetime64(t) for t in format_hour_no_24(settings.measures['time'])])

        sun_info = sg2.sun_position(
            [[settings.longitude, settings.latitude, settings.elevation]],
            self.time,
            ['topoc.gamma_S0', 'topoc.alpha_S', 'topoc.toa_ni']
        )
        self.gamma_s = np.array(sun_info.topoc.gamma_S0[0])
        self.alpha_s = np.array(sun_info.topoc.alpha_S[0])
        self.TOANI = np.array(sun_info.topoc.toa_ni[0])

        if settings.use_riso:
            # Compute Riso for each pyranometer (fit and dest) using its own horizon when available
            self.Riso_fit = []
            for pyr in range(1, settings.n_fit + 1):
                az = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_azimuth'][0])
                tilt = np.deg2rad(settings.measures[f'pyrano-fit-{pyr}_tilt'][0])
                horizon = self._get_horizon_for_pyr(pyr, role='fit')
                self.Riso_fit.append(calc_Riso(az, tilt, horizon))
            self.Riso_fit = np.array(self.Riso_fit)

            # Destination pyranometers also need Riso for projection
            self.Riso_dest = []
            for pyr in range(1, settings.n_predict + 1):
                az = np.deg2rad(settings.measures[f'pyrano-dest-{pyr}_azimuth'][0])
                tilt = np.deg2rad(settings.measures[f'pyrano-dest-{pyr}_tilt'][0])
                horizon = self._get_horizon_for_pyr(pyr, role='dest')
                self.Riso_dest.append(calc_Riso(az, tilt, horizon))
            self.Riso_dest = np.array(self.Riso_dest)

        else:
            self.Riso_fit = None
            self.Riso_dest = None

    def _get_horizon_for_pyr(self, pyr: int, role: str = 'fit'):
        """Return the horizon array for a given pyranometer.

        role: 'fit' or 'dest'. pyr is 1-based index.
        Supported settings.horizons formats:
        - 1D array (length 360): returned for all pyrs
        - 2D array: if first dim == n_fit + n_predict, fits come first then dests
                 else if first dim matches n_fit or n_predict, will map accordingly
        - dict with keys 'fit-<i>' or 'dest-<i>'
        Falls back to the first row or the 1D array when ambiguous.
        """
        h = self.settings.horizons
        # dict mapping
        if isinstance(h, dict):
            key = f"{role}-{pyr}"
            if key in h:
                return np.asarray(h[key])
            # fallback to generic key if present
            if pyr in h:
                return np.asarray(h[pyr])

        h_arr = np.asarray(h)
        if h_arr.ndim == 1:
            return h_arr
        # 2D array
        nrows = h_arr.shape[0]
        n_fit = self.settings.n_fit
        n_pred = self.settings.n_predict
        # case: horizons provided for all pyrs (fit first, then dest)
        if nrows == (n_fit + n_pred):
            if role == 'fit':
                idx = pyr - 1
            else:
                idx = n_fit + pyr - 1
            return h_arr[idx]
        # case: horizons provided only for fit pyrs
        if role == 'fit' and nrows == n_fit:
            return h_arr[pyr - 1]
        # case: horizons provided only for dest pyrs
        if role == 'dest' and nrows == n_pred:
            return h_arr[pyr - 1]
        # fallback: return first horizon row
        return h_arr[0]

    def process(self, must_fit = True)->pd.DataFrame:
        '''
        Returns the DataFrame with the computed GTI values for all the destination pyranometers
        '''
        # Backwards-compatible: run a full pipeline (fit then project all dest pyrs)
        self.fit()
        for pyr in range(1, self.settings.n_predict + 1):
            proj = self.project_one(pyr)
            # proj is (GTI, DTI, BTI, RTI) -> store GTI
            self.settings.measures[f'pyrano-dest-{pyr}_value'] = proj[0]
        return self.settings.measures

    def project_one(self, pyr: int):
        """Project irradiance for a single destination pyranometer (1-based index).

        Returns the GTI, DTI, BTI and RTI time series (numpy array) for the requested pyr.
        """
        projection = self.project_gti(
            alpha = np.deg2rad(self.settings.measures[f'pyrano-dest-{pyr}_azimuth']),
            beta = np.deg2rad(self.settings.measures[f'pyrano-dest-{pyr}_tilt']),
            BHI = self.settings.measures['pyrano-origin'].to_numpy() * (1 - self.kd),
            DHI = self.settings.measures['pyrano-origin'].to_numpy() * self.kd,
            gamma_s = self.gamma_s,
            alpha_s = self.alpha_s,
            TOANI = self.TOANI,
            elevation = self.settings.elevation,
            albedo = self.settings.albedo,
            use_riso = self.settings.use_riso,
            Riso = (self.Riso_dest[pyr-1] if (self.settings.use_riso and self.Riso_dest is not None) else None),
            horizon = self._get_horizon_for_pyr(pyr, role='dest'),
        )
        # projection is (GTI, DTI, BTI, RTI)
        return projection

    def fit(self):
        """Fit the model using the origin and fit pyranometers.

        This computes and stores `self.kd`, `self.optimal_index`, and `self.errors`.
        Returns the optimal `kd`.
        """
        kd_list = np.linspace(0.0001, 1, num=100)
        variations = [
            [
                self.project_gti(
                    alpha = np.deg2rad(self.settings.measures[f'pyrano-fit-{pyr}_azimuth'][0]),
                    beta = np.deg2rad(self.settings.measures[f'pyrano-fit-{pyr}_tilt'][0]),
                    BHI = self.settings.measures['pyrano-origin'].to_numpy() * (1 - kd_list[i]),
                    DHI = self.settings.measures['pyrano-origin'].to_numpy() * kd_list[i],
                    gamma_s = self.gamma_s,
                    alpha_s = self.alpha_s,
                    TOANI = self.TOANI,
                    elevation = self.settings.elevation,
                    albedo = self.settings.albedo,
                    use_riso = self.settings.use_riso,
                    Riso = (self.Riso_fit[pyr-1] if (self.settings.use_riso and self.Riso_fit is not None) else None),
                    horizon = self._get_horizon_for_pyr(pyr, role='fit'),
                )[0] for pyr in range(1, self.settings.n_fit + 1)
            ] for i in prange(len(kd_list))
        ]

        errors, optimal_index = get_errors_kd(self.settings, variations, kd_list)
        self.kd = kd_list[optimal_index]
        self.optimal_index = optimal_index
        self.errors = errors
        print(f"Optimal kd: {self.kd}")
        return self.kd

    @staticmethod
    def project_gti(
                    alpha: np.array,
                    beta: np.array,
                    BHI: np.array,
                    DHI: np.array,
                    gamma_s: np.array,
                    alpha_s: np.array,
                    TOANI: np.array,
                    elevation: float,
                    albedo: float,
                    use_riso: bool,
                    Riso: np.array = None,
                    horizon: np.array = None):
        assert np.shape(DHI) == np.shape(BHI), f'DHI: {np.shape(DHI)} is not same shape as BHI: {np.shape(BHI)}'
        assert np.shape(DHI) == np.shape(gamma_s), f'DHI: {np.shape(DHI)} is not same shape as gamma_s: {np.shape(gamma_s)}'
        assert np.shape(DHI) == np.shape(alpha_s), f'DHI: {np.shape(DHI)} is not same shape as alpha_s: {np.shape(alpha_s)}'
        assert np.shape(DHI) == np.shape(TOANI), f'DHI: {np.shape(DHI)} is not same shape as TOANI: {np.shape(TOANI)}'

        DHI[DHI == 0] = 0.001
        kappa = 1.041
        theta_s = np.pi / 2 - gamma_s
        cos_theta_s = np.cos(theta_s)
        min_gamma_s = 1 * np.pi / 180
        if type(horizon) == type(None):
            daylight_bool = gamma_s > min_gamma_s
        else:
            daylight_bool = gamma_s > np.maximum(min_gamma_s, horizon)[alpha_s.astype("int32")]
        AM = ModelKd.__calc_airmass(gamma_s, elevation, True)

        delta = AM * DHI / TOANI

        # deducing epsilon
        BNI = np.zeros(np.shape(daylight_bool))
        BNI[daylight_bool] = BHI[daylight_bool] / np.sin(gamma_s[daylight_bool])
        A = 1 + BNI / DHI
        B = (kappa * theta_s ** 3)
        epsilon = (A + B) / (1 + B)

        # Horizon and fi
        f1, f2 = ModelKd.__calc_f1_f2(epsilon, delta, theta_s)
        if use_riso:
            Vd = Riso
        else:
            Vd = (1 + np.cos(beta)) / 2
        cos_thetaI = np.cos(beta) * cos_theta_s + np.sin(beta)                  \
                        * np.cos(gamma_s) * np.cos(alpha_s - alpha)
        cos_thetaI[cos_thetaI < 0] = 0
        Rb = np.zeros(len(daylight_bool))
        Rb[daylight_bool] = cos_thetaI [daylight_bool] / np.maximum(0.087, cos_theta_s[daylight_bool])

        # Computing the fluxes
        DHI_cs = f1 * DHI
        DHI_iso = DHI - DHI_cs
        DTI = DHI_cs * Rb + Vd * DHI_iso + f2 * np.sin(beta) * DHI

        BTI = np.zeros(len(daylight_bool))
        BTI[daylight_bool] = BHI[daylight_bool] * cos_thetaI[daylight_bool] / cos_theta_s[daylight_bool]

        RTI = albedo * (1 - Vd) * (BHI + DHI)

        GTI = BTI + DTI + RTI
        return  GTI, DTI, BTI, RTI

    @staticmethod
    def __calc_airmass(gamma_s: np.array, elevation: float, corr_refract=True):
        """Calculates the relative optical airmass.

        Args:
            gamma_s (np.array): sun elevation through time
            elevation (float): elevation of the measurement site
            corr_refract (bool, optional): Correct gamma_s by taking into account the diffraction of the light passing through the athmosphere. Defaults to True.

        Returns:Cla
            np.array: the relative airmass
        """
        corr_h = np.array([np.exp(-elevation / 8434.5)] * len(gamma_s))
        gamma_s_deg = gamma_s * 180 / np.pi
        if corr_refract:
            Dgamma_s_deg = 180 / np.pi * 0.061359                                       \
                            * (0.1594 + 1.1230 * gamma_s + 0.065656 * gamma_s ** 2)    \
                            / (1 + 28.9344 * gamma_s + 277.3971 * gamma_s ** 2)
        else:
            Dgamma_s_deg = np.zeros(np.shape(gamma_s_deg))
        gamma_s_deg_c = gamma_s_deg + Dgamma_s_deg
        K = ((gamma_s_deg_c+1.79)>0)
        m = 57.6 * np.ones(np.shape(gamma_s))
        m[K] = corr_h[K] / (np.sin(gamma_s_deg_c[K] * np.pi / 180) + 0.50572 * (gamma_s_deg_c[K]+ 6.07995) ** (-1.6364))
        return m


    @staticmethod
    def __calc_f1_f2(epsilon: np.ndarray, delta: np.ndarray, theta_s: np.ndarray, nbin=8):
        """Computes the f1 and f2 coefficients from the Perez Model.

        Args:
            epsilon (float): epsilon from the Perez Model
            delta (np.array): delta from the Perez Model
            theta_s (np.array): theta_s frm the Perez Model
            nbin (int, optional): number of epsilon classes in your eps_bin array. Defaults to 8.

        Returns:
            tuple[np.array]: the f1 and f2 coefficients through time
        """
        epsilon = epsilon.astype("float64")
        delta = delta.astype("float64")

        ok = np.logical_and(np.logical_not(np.isinf(epsilon)), np.logical_not(np.isnan(epsilon)))
        ok = np.logical_and(np.logical_not(np.isnan(delta)), ok)
        epsilon = epsilon[ok]
        delta = delta[ok]
        theta_s = theta_s[ok]
        kbin_epsilon = np.array([np.nan] * len(epsilon))
        for kb in range(nbin):
            b = np.logical_and(epsilon >= ModelKd.bin_epsilon_0[kb], epsilon < ModelKd.bin_epsilon_1[kb])
            kbin_epsilon[b] = kb

        kbin_epsilon = kbin_epsilon.astype("int64")
        kbin_epsilon[np.isnan(kbin_epsilon)] = 0
        f11 = ModelKd.tabF11[kbin_epsilon]
        f12 = ModelKd.tabF12[kbin_epsilon]
        f13 = ModelKd.tabF13[kbin_epsilon]
        f1 = np.array([np.nan] * len(ok))
        f1[ok] = f11 + f12 * delta + f13 * theta_s
        f21 = ModelKd.tabF21[kbin_epsilon]
        f22 = ModelKd.tabF22[kbin_epsilon]
        f23 = ModelKd.tabF23[kbin_epsilon]
        f2 = np.array([np.nan] * len(ok))
        f2[ok] = f21 + f22 * delta + f23 * theta_s
        return f1, f2