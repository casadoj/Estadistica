class bias_correction(object):
    """
    Apply bias correction to all scenario series assuming two possible methods:

    quantile_mapping or scaled_distribution_mapping

    Args:

    * obs_cube (vector):
        the observational data

    * mod_cube (vector):
        the model data at the reference period

    * sce_cubes (vector):
        the scenario data that shall be corrected"""
    
    def __init__(self, obs,mod,sce):
        self.obs=obs
        self.mod=mod
        self.sce=sce
        return
    def quantile_mapping (self):
        from statsmodels.distributions.empirical_distribution import ECDF
        
        obs_cube=self.obs.copy()
        mod_cube=self.mod.copy()
        sce_cubes=self.sce.copy()

        obs_data = obs_cube
        mod_data = mod_cube
        mod_ecdf = ECDF(mod_data)

        sce_data = sce_cubes
        p = mod_ecdf(sce_data).flatten() * 100
        corr = np.percentile(obs_data, p) - \
            np.percentile(mod_data, p)
        sce_cubes += corr
        sce_cubes[sce_cubes<0]=0
        return(sce_cubes)
    
    def quantile_deltamapping (self):
        from statsmodels.distributions.empirical_distribution import ECDF

        obs_cube=self.obs.copy()
        mod_cube=self.mod.copy()
        sce_cubes=self.sce.copy()

        obs_data = obs_cube
        mod_data = mod_cube
        mod_ecdf = ECDF(mod_data)

        sce_data = sce_cubes
        p = mod_ecdf(sce_data).flatten() * 100
        corr = np.percentile(obs_data, p) / np.percentile(mod_data, p)
        sce_cubes= sce_cubes*corr
        sce_cubes[sce_cubes<0]=0
        return(sce_cubes)
    
    
    def scaled_distribution_mapping(self,variable, *args, **kwargs):
        """
        Scaled Distribution Mapping

        apply scaled distribution mapping to all scenario cubes

        the method works differently for different meteorological parameters

        * air_temperature
            :meth:`absolute_sdm` using normal distribution
        * precipitation_amount, surface_downwelling_shortwave_flux_in_air
            :meth:`relative_sdm` using gamma distribution

        Args:

        * obs_cube:
            the observational data

        * mod_cube:
            the model data at the reference period

        * sce_cubes:
            the scenario data that shall be corrected
        * variable:
            meteorological variable: precipitation or temperature
        
        """
        def relative_sdm(obs_c,mod_c,sce_c, *args, **kwargs):
            """
            Apply relative scaled distribution mapping to all scenario cubes
            assuming a gamma distributed parameter (with lower limit zero)

            if one of obs, mod or sce data has less than min_samplesize valid
            values, the correction will NOT be performed but the original data
            is output

            Args:

            * obs_c (vector):
                the observational data

            * mod_c (vector):
                the model data at the reference period

            * sce_c (vector):
                the scenario data that shall be corrected

            Kwargs:

            * lower_limit (float):
                assume values below lower_limit to be zero (default: 0.1)

            * cdf_threshold (float):
                limit of the cdf-values (default: .99999999)

            * min_samplesize (int):
                minimal number of samples (e.g. wet days) for the gamma fit
                (default: 10)
            """
            from scipy.stats import gamma

            lower_limit = kwargs.get('lower_limit', 0.1)
            cdf_threshold = kwargs.get('cdf_threshold', .99999999)
            min_samplesize = kwargs.get('min_samplesize', 10)


            obs_data = obs_c.copy()
            mod_data = mod_c.copy()
            obs_raindays = obs_data[obs_data >= lower_limit]
            mod_raindays = mod_data[mod_data >= lower_limit]


            obs_frequency = 1. * len(obs_raindays) / len(obs_data)
            mod_frequency = 1. * len(mod_raindays) / len(mod_data)
            obs_gamma = gamma.fit(obs_raindays, floc=0)
            mod_gamma = gamma.fit(mod_raindays, floc=0)

            obs_cdf = gamma.cdf(np.sort(obs_raindays), *obs_gamma)
            mod_cdf = gamma.cdf(np.sort(mod_raindays), *mod_gamma)
            obs_cdf[obs_cdf > cdf_threshold] = cdf_threshold
            mod_cdf[mod_cdf > cdf_threshold] = cdf_threshold

            sce_data = sce_c.copy()
            sce_raindays = sce_data[sce_data >= lower_limit]


            sce_frequency = 1. * len(sce_raindays) / len(sce_data)
            sce_argsort = np.argsort(sce_data)
            sce_gamma = gamma.fit(sce_raindays, floc=0)

            expected_sce_raindays = min(
                np.round(
                    len(sce_data) * obs_frequency * sce_frequency
                    / mod_frequency),
                len(sce_data))

            sce_cdf = gamma.cdf(np.sort(sce_raindays), *sce_gamma)
            sce_cdf[sce_cdf > cdf_threshold] = cdf_threshold

            # interpolate cdf-values for obs and mod to the length of the
            # scenario
            obs_cdf_intpol = np.interp(
                np.linspace(1, len(obs_raindays), len(sce_raindays)),
                np.linspace(1, len(obs_raindays), len(obs_raindays)),
                obs_cdf
            )
            mod_cdf_intpol = np.interp(
                np.linspace(1, len(mod_raindays), len(sce_raindays)),
                np.linspace(1, len(mod_raindays), len(mod_raindays)),
                mod_cdf
            )

            # adapt the observation cdfs
            obs_inverse = 1. / (1 - obs_cdf_intpol)
            mod_inverse = 1. / (1 - mod_cdf_intpol)
            sce_inverse = 1. / (1 - sce_cdf)
            adapted_cdf = 1 - 1. / (obs_inverse * sce_inverse / mod_inverse)
            adapted_cdf[adapted_cdf < 0.] = 0.

            # correct by adapted observation cdf-values
            xvals = gamma.ppf(np.sort(adapted_cdf), *obs_gamma) *\
                gamma.ppf(sce_cdf, *sce_gamma) /\
                gamma.ppf(sce_cdf, *mod_gamma)

            # interpolate to the expected length of future raindays
            correction = np.zeros(len(sce_data))
            if len(sce_raindays) > expected_sce_raindays:
                xvals = np.interp(
                    np.linspace(1, len(sce_raindays), int(expected_sce_raindays)),
                    np.linspace(1, len(sce_raindays), len(sce_raindays)),
                    xvals
                )
            else:
                xvals = np.hstack(
                    (np.zeros(expected_sce_raindays -
                              len(sce_raindays)), xvals))

            correction[sce_argsort[-int(expected_sce_raindays):].flatten()] = xvals
            return correction
    
        def absolute_sdm(obs_c,mod_c,sce_c, *args, **kwargs):
            """
            apply absolute scaled distribution mapping to all scenario cubes
            assuming a normal distributed parameter

            Args:

            * obs_cube (:class:`iris.cube.Cube`):
                the observational data

            * mod_cube (:class:`iris.cube.Cube`):
                the model data at the reference period

            * sce_cubes (:class:`iris.cube.CubeList`):
                the scenario data that shall be corrected

            Kwargs:

            * cdf_threshold (float):
                limit of the cdf-values (default: .99999)
            """
            from scipy.stats import norm
            from scipy.signal import detrend

            cdf_threshold = kwargs.get('cdf_threshold', .99999)


            obs_data = obs_c.copy()
            mod_data = mod_c.copy()
            obs_len = len(obs_data)
            mod_len = len(mod_data)

            obs_mean = obs_data.mean()
            mod_mean = mod_data.mean()

            # detrend the data
            obs_detrended = detrend(obs_data)
            mod_detrended = detrend(mod_data)

            obs_norm = norm.fit(obs_detrended)
            mod_norm = norm.fit(mod_detrended)

            obs_cdf = norm.cdf(np.sort(obs_detrended), *obs_norm)
            mod_cdf = norm.cdf(np.sort(mod_detrended), *mod_norm)
            obs_cdf = np.maximum(
                np.minimum(obs_cdf, cdf_threshold), 1 - cdf_threshold)
            mod_cdf = np.maximum(
                np.minimum(mod_cdf, cdf_threshold), 1 - cdf_threshold)

            sce_data = sce_c.copy()

            sce_len = len(sce_data)
            sce_mean = sce_data.mean()

            sce_detrended = detrend(sce_data)
            sce_diff = sce_data - sce_detrended
            sce_argsort = np.argsort(sce_detrended)

            sce_norm = norm.fit(sce_detrended)
            sce_cdf = norm.cdf(np.sort(sce_detrended), *sce_norm)
            sce_cdf = np.maximum(
                np.minimum(sce_cdf, cdf_threshold), 1 - cdf_threshold)

            # interpolate cdf-values for obs and mod to the length of the
            # scenario
            obs_cdf_intpol = np.interp(
                np.linspace(1, obs_len, sce_len),
                np.linspace(1, obs_len, obs_len),
                obs_cdf
            )
            mod_cdf_intpol = np.interp(
                np.linspace(1, mod_len, sce_len),
                np.linspace(1, mod_len, mod_len),
                mod_cdf
            )

            # adapt the observation cdfs
            # split the tails of the cdfs around the center
            obs_cdf_shift = obs_cdf_intpol - .5
            mod_cdf_shift = mod_cdf_intpol - .5
            sce_cdf_shift = sce_cdf - .5
            obs_inverse = 1. / (.5 - np.abs(obs_cdf_shift))
            mod_inverse = 1. / (.5 - np.abs(mod_cdf_shift))
            sce_inverse = 1. / (.5 - np.abs(sce_cdf_shift))
            adapted_cdf = np.sign(obs_cdf_shift) * (
                1. - 1. / (obs_inverse * sce_inverse / mod_inverse))
            adapted_cdf[adapted_cdf < 0] += 1.
            adapted_cdf = np.maximum(
                np.minimum(adapted_cdf, cdf_threshold), 1 - cdf_threshold)

            xvals = norm.ppf(np.sort(adapted_cdf), *obs_norm) \
                + obs_norm[-1] / mod_norm[-1] \
                * (norm.ppf(sce_cdf, *sce_norm) - norm.ppf(sce_cdf, *mod_norm))
            xvals -= xvals.mean()
            xvals += obs_mean + (sce_mean - mod_mean)

            correction = np.zeros(sce_len)
            correction[sce_argsort] = xvals
            correction += sce_diff - sce_mean
            
            return correction
       
        implemented_parameters = {
            'temperature': absolute_sdm,
            'precipitation': relative_sdm,
            'surface_downwelling_shortwave_flux_in_air': relative_sdm,
        }
        try:
            corr=implemented_parameters[variable](
                self.obs,self.mod,self.sce, *args, **kwargs)
            return corr
        except KeyError:
            print(
                'SDM not implemented for {}'.format(variable))