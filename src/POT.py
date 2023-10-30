#!/usr/bin/env python
# coding: utf-8

# # Funciones POT (peaks-over-threshold)
# ***
# 
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __30/10/2023__ <br>
# 
# __Introducción__<br>
# Se desarrolla una clase llamada `POT` en la que, definida una serie y las distribuciones discreta y continua a ajustar, permite hacer un análisis de extremos mediate POT.
# 
# __Cosas que arreglar__ <br>
# En el método POT, la distribución discreta puede ser una Poisson, una binomial o una binomial negativa. Por ahora sólo está montado con la Poisson. Igualmente, la distribución continua puede ser una exponencial o una pareto generalizada; por ahora sólo está montado con la exponencial.
# 
# __Índice__ <br>
# 
# __[Funciones](#Funciones)__ <br>
# 
# __[Clase](#Clase)__ <br>



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from scipy.stats import rv_discrete, rv_continuous
from fitter import Fitter
from typing import Union, List


# diccionario para llamar a las funciones de distribución a partir de strings
DFdict = {'beta': stats.beta,
          'expon': stats.expon,
          # 'frechet_r': stats.frechet_r,
          'gamma': stats.gamma,
          'genextreme': stats.genextreme,
          'genpareto': stats.genpareto,
          'gumbel_r': stats.gumbel_r,
          'norm': stats.norm,
          'weibull_max': stats.weibull_max}


class POT:
    
    def __init__(self, series: pd.Series, distDis: rv_discrete, distCon: rv_continuous, threshold: float = None):
        """
        Parámetros:
        -----------
        series:     pd.Series
            Serie temporal; el índice debe ser 'datetime' o 'datetime.date'
        distDis:   rv_discrete
            Función de distribución discreta con la que modelar la ocurrencia anual de eventos
        distCon:   rv_continuous
            Función de distribución continua con la que modelar la magnitud de los eventos
        threshold: float
            Umbral a partir del cual se identifica un evento
        """
        
        self.series = series
        self.distDis = distDis
        self.distCon = distCon
        self.threshold = threshold

    def select_peaks(self, days: int = None, plot: bool = True) -> pd.Series:
        """Selección de eventos para el futuro ajuste de una distribución POT (peaks-over-threshold).

        En primer lugar se seleccionan todos los pasos temporales con valor por encima del umbral. Seguidamente se selecciona úncicamente el máximo de la cada subserie de excesos sobre el umbral, p.ej., el máximo de cada hidrograma de avenida.

        Parámetros:
        -----------
        plot:      boolean. Si se quiere mostrar un gráfico de línea con la serie temporal y los eventos seleccionados

        Salida:
        -------
        eventos:   pd.Series. Serie de eventos seleccionados; el índice sigue siendo 'datetime' o 'datetime.date'
        """

        # boolean series of exceedances over thresholds
        temp = self.series.copy() * 0
        temp[self.series >= self.threshold] = 1
        # the difference of the boolean timeseries indicates starts (1) and ends (-1)
        diff = temp.diff()
        if temp.iloc[0] == 1: # add a start if the time series exceeds the threshold in the 1st timestep
            diff.iloc[0] = 1
        if (temp.iloc[-1] == 1) & (diff.iloc[-1] == 0): # add an end if the time series exceeds the threshold in the last timestep
            diff.iloc[-1] = -1
        # identify start and end of the periods exceeding the threshold
        starts = temp[diff == 1].index
        ends = temp[diff == -1].index

        # find maxima in those periods
        peaks = pd.Series({self.series[start:end].idxmax(): self.series[start:end].max() for start, end in zip(starts, ends)})

        # remove peaks that are closer than the specified 'days'
        if days is not None:
            mask = peaks.index.to_series().diff().dt.total_seconds() < days * 3600 * 24
            peaks = peaks[~mask]

        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(self.series, 'steelblue', lw=.5, label='serie')
            plt.axhline(self.threshold, c='k', ls=':', lw=.75, label='threshold')
            plt.scatter(peaks.index, peaks, marker='+', s=30, lw=1, c='indianred',
                        label='eventos')
            plt.xlim(self.series.index[0], self.series.index[-1])
            plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.1, -0.22, 0.8, 0.1), frameon=False);

        self.peaks = peaks
        return peaks
    
    def fit_discrete(self, plot: bool = True) -> float:
        """Ajusta la distribución discreta que modela la ocurrencia anual de eventos extremos
        
        Input:
        ------
        plot: bool
            Whether to plot or not the fitted distribution
            
        Output:
        -------
        mu:    float
            Mean of the discrete distribution
        """
        
        # recurrencia (parámetro de la distribución Poisson)
        T = 365 # pasos temporales en la serie correspondiente al intervalo de estudio (1 año)
        mu = T * len(self.peaks) / len(self.series)
        print('mu = {0:.3f}'.format(mu))
        
        # momentos de la distribución de Poisson
        mean, var, skew = self.distDis.stats(mu, moments='mvs')
        print('E(m) = {0:.3f}\tvar(m) = {1:.3f}\tmu3 = {2:.3f}'.format(mean, var, skew))

        # comprobar que la distribución de Poisson es adecuada
        if mean == var:
            print('La distribución Poisson es adecuada para el nº anual de eventos')
        elif mean > var:
            print('Usar la distribución binomial')
        elif mean < var:
            print('Usar la distribución binomial negativa')
        
        if plot:
            epsilon = 1e-4
            kmin = int(self.distDis.ppf(epsilon, mu))
            kmax = int(self.distDis.ppf(1 - epsilon, mu))
            k = np.arange(kmin, kmax)

            fig, ax = plt.subplots(nrows=2, figsize=(4, 8.5), sharex=True)
            ax[0].scatter(k, self.distDis.pmf(k, mu), s=5, label='PMF')
            ax[0].set_ylabel('PMF')
            ax[1].scatter(k, self.distDis.cdf(k, mu), s=5, label='CDF')
            ax[1].set(xlabel='no. peaks', ylabel='CDF');

        self.mu = mu
        return mu
        
    def fit_continuous(self, plot: bool = True) -> List[float]:
        """Ajusta la distribución continua que modela la magnitud de un evento extremo
        
        Parámetros:
        -----------
        plot:      bool
            Si se quiere mostrar gráficos del ajuste
        
        Salidas:
        --------
        pars: List
            Parámetros ajustados de la distribución continua: localización y escala, si es una exponencial
        """
        
        # serie de excedencias en los eventos
        y = self.peaks - self.threshold

        # ajustar distribución empírica
        ecdfPOT = ECDF(y)
        ecdfs = ecdfPOT(y)

        # ajustar distribución exponencial
        pars = self.distCon.fit(y)
        ys = np.linspace(0, np.ceil(y.max()), 200)
        cdfs = self.distCon.cdf(ys, *pars)
        
        if plot:
            fig, ax = plt.subplots(nrows=2, figsize=(4, 8.5), sharex=True)
            # histograma
            sns.histplot(y, kde=False, stat='count', ax=ax[0])
            ax[0].set(xlabel=r'$serie - thr$ (m)', ylabel='nº eventos')
            # cdf empírica vs ajuste
            ax[1].plot(ys, cdfs, label='ajuste')
            ax[1].scatter(y, ecdfs, s=1, c='k', label='empírica')
            ax[1].set(xlim=(0, np.ceil(y.max())), ylim=(0), ylabel='CDF', xlabel=r'$serie - thr$')
            ax[1].legend(loc=4, frameon=False);
            
        self.parsCon = pars
        return pars
        
    def cdf(self, x: Union[List, np.ndarray]) -> np.ndarray:
        """Calcula la función de distribución acumulada (CDF, cumulative density function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        x:         array or list (n,)
            Valor/es de la variable para los que calcular su probabilidad de no excedencia acumulada

        Salida:
        -------
        cdf:       array (n,)
            Probabilidad de no excedencia de cada uno de los valores de 'x'
        """

        # asegurar que 'x' es un array
        x = np.array(x)
        # restar el umbral para convertir la variable en el exceso
        y = x - self.threshold
        
        # 'array' donde guardar la probabilidad acumulada
        cdf = np.zeros_like(y)

        # nº máximo de eventos anuales
        kmax = int(self.distDis.ppf(1 - 1e-5, self.mu))
        for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
            cdf += self.distDis.pmf(k, self.mu) * self.distCon.cdf(y, *self.parsCon)**k

        return cdf
    
    def pdf(self, x: Union[List, np.ndarray]) -> np.ndarray:
        """Calcula la función de densidad (PDF, probability distribution function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        x:         array or list (n,)
            Valor/es de la variable para los que calcular su probabilidad de no excedencia

        Salida:
        -------
        pdf:       array (n,)
            Probabilidad de cada uno de los valores de 'x'
        """

        # asegurar que 'x' es un array
        x = np.array(x)
        y = x - self.threshold
        # 'array' donde guardar la probabilidad acumulada
        pdf = np.zeros_like(y)

        # nº máximo de eventos anuales
        kmax = int(self.distDis.ppf(1 - 1e-5, self.mu))
        for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
            pdf += self.distDis.pmf(k, self.mu) * k * self.distCon.cdf(y, *self.parsCon)**(k - 1) * self.distCon.pdf(y, *self.parsCon)

        return pdf
    
    def ppf(self, q: Union[List, np.ndarray]) -> np.ndarray:
        """Calcula el cuantil asociado a una probabilidad de no excedencia de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        q:         array or list (n,)
            Valor/es de probabilidad de no excedencia

        Salida:
        -------
        x:       array (n,)
            Cuantiles asociados a cada probabilidad de 'q'
        """

        # asegurar que 'q' es un array
        q = np.array(q)

        # array de resultados
        x = np.zeros_like(q)

        # nº máximo de eventos anuales
        kmax = int(self.distDis.ppf(1 - 1e-5, self.mu))

        # calcular el cuantil para cada probabilidad de no excedencia
        # ecuación: PMFpoiss(kmax) * CDFexp(y)**kmax + ... + PMFpoiss(2) * CDFexp(y)**2 + PMFpoiss(1) * CDFexp(y) - q = 0
        for i, qx in enumerate(q):
            # coeficientes de la ecuación lineal: [PMFpoiss(kmax), ..., PMFpoiss(2), PMFpoiss(1), -q]
            p = np.array([self.distDis.pmf(k, self.mu) for k in range(0, kmax + 1)[::-1]])
            p[-1] -= qx
            # raíces de la ecuación: CDFexp(y)
            roots = np.roots(p)
            # encontrar la raíz real con valor positivo
            for root in roots:
                if (np.imag(root) == 0) & (np.real(root) > 0):
                    qy = np.real(root)
                    break

            # valor de exceso asociado a dicho cuantil
            y = self.distCon.ppf(qy, *self.parsCon)
            # convertir a la variable original
            x[i] = y + self.threshold

        return x