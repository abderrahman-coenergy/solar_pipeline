import dataclasses
import numpy as np
from coe_sol.private.ModelKd import ModelKd, ModelKdSettings
from coe_sol.utils import calc_Riso
import pandas as pd

@dataclasses.dataclass
class PyranoInfo:
    azimuth_deg: float
    inclination_deg: float
    horizon: np.array

@dataclasses.dataclass
class PyranoMeasure:
    timestamps: np.array
    values: np.array

class RealPyrano:
    def __init__(self, info: PyranoInfo, measures: PyranoMeasure):
        self.info = info
        self.measures = measures

class VirtualPyrano:
    def __init__(self, info: PyranoInfo):
        self.info = info

@dataclasses.dataclass
class SolarModelOptions:
    latitude: float = 0
    longitude: float = 0
    albedo: float = 0.2
    use_riso: bool = False
    elevation_meter: float = 0 # elevation of the building, the small diff between pyranos is negligible

class SolarModel:
    '''
    SolarModel class to fit and project solar irradiance using real and virtual pyranometers.
    Usage:
        1. Create an instance of SolarModel with desired options.
            solar = SolarModel(options)
        2. Set the origin pyranometer (real).
            solar.set_origin(origin_pyrano)
        3. Add fit pyranometers (real).
            solar.add_fit(fit_pyrano_1)
            solar.add_fit(fit_pyrano_2)
            ...
        4. Add target pyranometers (virtual).
            solar.add_target(target_pyrano_1)
            solar.add_target(target_pyrano_2)
            ...
        5. Fit the model parameters.
            solar.fit_parameters()
        6. Project irradiance for target pyranometers.
            result_df = solar.project()
    '''
    def __init__(self, options: SolarModelOptions):
        self._fit_pyr: list[RealPyrano] = []
        self._origin_pyr: RealPyrano | None  = None
        self._target_pyr: list[VirtualPyrano] = []
        self.options = options
        self.fitted = False
        self.model = None

    def set_origin(self, pyr: RealPyrano):
        '''Registers the origin pyranometer (real) for the model.'''
        self._origin_pyr = pyr

    def add_fit(self, pyr: RealPyrano):
        '''Adds a fit pyranometer (real) to the model.'''
        self._fit_pyr.append(pyr)

    def add_target(self, pyr: VirtualPyrano):
        '''Adds a target pyranometer (virtual) to the model. This pyranometer's measures will
        be projected by the model after fitting.'''
        self._target_pyr.append(pyr)

    def _assert_timestamps_match(self):
        '''
        Asserts that the timestamps of the origin pyranometer and all fit pyranometers match.
        Raises an error if they do not match. Timestamps are expected to be numpy arrays with
        elemets formatted as 'YYYY-MM-DD HH:mm:ss.sss' strings.
        '''
        if self._origin_pyr is None:
            raise ValueError("Origin pyranometer not set")

        def _to_iso_strings(arr):
            # normalize numpy datetime64 and other date-like values to ISO-like strings
            out = []
            for v in arr:
                if isinstance(v, np.datetime64):
                    # produce ISO-like with milliseconds
                    out.append(np.datetime_as_string(v, unit='ms'))
                else:
                    out.append(str(v))
            return np.array(out)

        origin_timestamps = _to_iso_strings(self._origin_pyr.measures.timestamps)
        for fit_pyr in self._fit_pyr:
            fit_timestamps = _to_iso_strings(fit_pyr.measures.timestamps)
            if not np.array_equal(origin_timestamps, fit_timestamps):
                raise ValueError("Timestamps of origin pyranometer and fit pyranometers do not match")

    def fit_parameters(self):
        '''This method uses the fit pyranometer and the origin pyranometer to estimate the parameters of the physical model,
        which will then be used to project the irradiance on the target pyranometers.'''
        if self._origin_pyr is None:
            raise ValueError("Origin pyranometer not set")
        if len(self._fit_pyr) == 0:
            raise ValueError("No fit pyranometers set")
        self._assert_timestamps_match()
        self._create_model()
        if self.model is None:
            raise RuntimeError("Model not created, _create_model() must be implemented in subclass")
        self.model.fit()
        self.fitted = True

    def _create_model(self):
        '''Creates the solar model instance using the registered pyranometers and options.'''
        if self._origin_pyr is None:
            raise ValueError("Origin pyranometer not set")
        if len(self._fit_pyr) == 0:
            raise ValueError("No fit pyranometers set")

        settings = ModelKdSettings()
        settings.latitude = self.options.latitude
        settings.longitude = self.options.longitude
        settings.elevation = self.options.elevation_meter
        settings.albedo = self.options.albedo
        settings.use_riso = self.options.use_riso
        settings.n_fit  = len(self._fit_pyr)
        settings.horizons = self.build_horizons_array()
        settings.n_predict = len(self._target_pyr)
        settings.measures = self.build_measure_df()

        self.model = ModelKd(settings)

    def build_horizons_array(self) -> np.array:
        '''Appends the horizon lines of fit pyranometers and target (dest) pyranometers into a 2D numpy array.
        Note that the origin pyranometer's horizon is not included in this array, as it is not used by the model.'''
        n_fit = len(self._fit_pyr)
        if n_fit > 0:
            horizons = np.zeros((n_fit, 360))
            # append fit pyranometers horizons
            for i, fit_pyr in enumerate(self._fit_pyr):
                horizons[i, :] = fit_pyr.info.horizon
            # append target pyranometers horizons
            for i, pyr_dest in enumerate(self._target_pyr):
                horizons = np.vstack([horizons, pyr_dest.info.horizon])
            return horizons
        else:
            return np.zeros(360)

    def build_measure_df(self) -> pd.DataFrame:
        ''''Builds the measures DataFrame required by ModelKd from the registered pyranometers.
        Format of the DataFrame:
        - time: timestamps from origin pyranometer
        - pyrano-origin: values from origin pyranometer
        - pyrano-fit-i_value: values from fit pyranometer i
        - pyrano-fit-i_azimuth: azimuth of fit pyranometer i
        - pyrano-fit-i_tilt: tilt of fit pyranometer i
        - pyrano-dest-i_azimuth: azimuth of target pyranometer i
        - pyrano-dest-i_tilt: tilt of target pyranometer i
        '''
        if self._origin_pyr is None:
            raise ValueError("Origin pyranometer not set")
        length = len(self._origin_pyr.measures.timestamps)
        data = {}
        # Normalize timestamps to ISO-like strings expected by ModelKd.format_hour_no_24
        def _to_iso_strings(arr):
            out = []
            for v in arr:
                if isinstance(v, np.datetime64):
                    out.append(np.datetime_as_string(v, unit='ms'))
                else:
                    out.append(str(v))
            return np.array(out)

        data['time'] = _to_iso_strings(self._origin_pyr.measures.timestamps)
        data['pyrano-origin'] = np.asarray(self._origin_pyr.measures.values)

        for i, fit_pyr in enumerate(self._fit_pyr, start=1):
            data[f'pyrano-fit-{i}_value'] = np.asarray(fit_pyr.measures.values)
            data[f'pyrano-fit-{i}_azimuth'] = np.full(length, fit_pyr.info.azimuth_deg)
            data[f'pyrano-fit-{i}_tilt'] = np.full(length, fit_pyr.info.inclination_deg)

        for i, pyr_dest in enumerate(self._target_pyr, start=1):
            data[f'pyrano-dest-{i}_azimuth'] = np.full(length, pyr_dest.info.azimuth_deg)
            data[f'pyrano-dest-{i}_tilt'] = np.full(length, pyr_dest.info.inclination_deg)

        return pd.DataFrame(data)

    def project(self) -> pd.DataFrame:
        '''Projects the irradiance for all target pyranometers using the fitted model.
        Returns a DataFrame with the projected values for each target pyranometer.
        Format of the DataFrame:
        - time: timestamps from origin pyranometer
        - pyrano-origin: values from origin pyranometer
        - pyrano-fit-i_value: values from fit pyranometer i
        - pyrano-fit-i_azimuth: azimuth of fit pyranometer i
        - pyrano-fit-i_tilt: tilt of fit pyranometer i
        - pyrano-dest-i_value: projected values for target pyranometer i
        - pyrano-dest-i_azimuth: azimuth of target pyranometer i
        - pyrano-dest-i_tilt: tilt of target pyranometer i

        Before using this method, user must set the origin, fit and target pyranometers,
        and call fit_parameters() to fit the model.
        '''
        if not self.fitted:
            raise RuntimeError("Model not fitted yet, call fit_parameters() first")
        if self.model is None:
            raise RuntimeError("Model not created, _create_model() must be implemented in subclass")
        result_df = self.model.process(must_fit=False)
        return result_df
