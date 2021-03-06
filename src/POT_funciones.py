#!/usr/bin/env python
# coding: utf-8

# # Funciones POT (peaks-over-threshold)
# ***
# 
# _Autor:_    __Jesús Casado__ <br> _Revisión:_ __06/03/2020__ <br>
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

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
import os
rutaBase = os.getcwd().replace('\\', '/') + '/'


# In[3]:


from statsmodels.distributions.empirical_distribution import ECDF
from scipy import stats
from fitter import Fitter


# In[4]:


# diccionario para llamar a las funciones de distribución a partir de 'string's
DFdict = {'beta': stats.beta, 'expon': stats.expon, 'frechet_r': stats.frechet_r,
          'gamma': stats.gamma, 'genextreme': stats.genextreme,
          'genpareto': stats.genpareto, 'gumbel_r': stats.gumbel_r,
          'norm': stats.norm, 'weibull_max': stats.weibull_max}


# ## Funciones

# In[5]:


def POTevents(serie, thr, plot=True):
    """Selección de eventos para el futuro ajuste de una distribución POT (peaks-over-threshold).
    
    En primer lugar se seleccionan todos los pasos temporales con valor por encima del umbral. Seguidamente se selecciona úncicamente el máximo de la cada subserie de excesos sobre el umbral, p.ej., el máximo de cada hidrograma de avenida.
    
    Parámetros:
    -----------
    serie:     pd.Series. Serie temporal; el índice debe ser 'datetime' o 'datetime.date'
    thr:       float. Umbral a partir del cual se identifica un evento
    plot:      boolean. Si se quiere mostrar un gráfico de línea con la serie temporal y los eventos seleccionados
    
    Salida:
    -------
    events:    pd.Series. Serie de eventos seleccionados; el índice sigue siendo 'datetime' o 'datetime.date'
    """
    
    # serie preliminar de eventos
    events0 = serie[serie > thr]

    # diferencia en días entre los eventos consecutivos de 'events0'
    At = events0.index[1:] - events0.index[:-1]
    At = pd.Series([t.days for t in At], index=events0.iloc[:-1].index)

    # serie de eventos independientes
    events = pd.Series()
    To, Tf = [], []
    i = 0
    while i < len(events0) - 2:
        to = events0.index[i]
        for j in range(i, len(events0) - 1):
            tf = events0.index[j]
            if At[tf] > 1:
                break
        events.loc[events0.loc[to:tf].idxmax()] = events0.loc[to:tf].max()
        i = events0.index.get_loc(tf) + 1
    
    if plot:
        plt.figure(figsize=(12, 4))
        plt.plot(serie, 'steelblue', lw=.5, label='serie')
        plt.hlines(thr, serie.index[0], serie.index[-1], ls=':', lw=.75, label='umbral')
        plt.scatter(events.index, events, marker='+', s=30, lw=1, c='indianred',
                    label='eventos')
        plt.xlim(serie.index[0], serie.index[-1])
        plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.1, -0.22, 0.8, 0.1));
        
    return events


# In[6]:


def POTcdf(x, thr, mu, loc, scale):
    """Calcula la función de distribución acumulada (CDF, cumulative density function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.
    
    Parámetros:
    -----------
    x:         array or list (n,). Valor/es de la variable para los que calcular su probabilidad de no excedencia
    thr:       float. Umbral que define los eventos considerados como extremos en la POT
    mu:        float. Parámetro de ocurrencia media de la distribución de Poisson
    loc:       float. Parámetro de localización de la distribución exponencial
    scale:     float. Parámetro de escala de la distribución exponencial
    
    Salida:
    -------
    cdf:       array (n,). Probabilidad de no excedencia de cada uno de los valores de 'x'
    """
    
    # asegurar que 'x' es un array
    x = np.array(x)
    # restar el umbral para convertir la variable en el exceso
    x -= thr
    # 'array' donde guardar la probabilidad acumulada
    cdf = np.zeros_like(x)
    
    # nº máximo de eventos anuales
    kmax = int(stats.poisson.ppf(1 - 1e-5, mu))
    for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
        cdf += stats.poisson.pmf(k, mu) * stats.expon.cdf(x, *parsExp)**k
        
    return cdf


# In[7]:


def POTpdf(x, thr, mu, loc, scale):
    """Calcula la función de densidad (PDF, probability distribution function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.
    
    Parámetros:
    -----------
    x:         array or list (n,). Valor/es de la variable para los que calcular su probabilidad de no excedencia
    thr:       float. Umbral que define los eventos considerados como extremos en la POT
    mu:        float. Parámetro de ocurrencia media de la distribución de Poisson
    loc:       float. Parámetro de localización de la distribución exponencial
    scale:     float. Parámetro de escala de la distribución exponencial
    
    Salida:
    -------
    pdf:       array (n,). Probabilidad de cada uno de los valores de 'x'
    """
    
    # asegurar que 'x' es un array
    x = np.array(x)
    x -= thr
    # 'array' donde guardar la probabilidad acumulada
    pdf = np.zeros_like(x)
    
    # nº máximo de eventos anuales
    kmax = int(stats.poisson.ppf(1 - 1e-5, mu))
    for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
        pdf += stats.poisson.pmf(k, mu) * k * stats.expon.cdf(x, *parsExp)**(k - 1) * stats.expon.pdf(x, *parsExp)
    
    return pdf


# In[8]:


def POTppf(q, thr, mu, loc, scale):
    """Calcula el cuantil asociado a una probabilidad de no excedencia de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.
    
    Parámetros:
    -----------
    q:         array or list (n,). Valor/es de probabilidad de no excedencia
    thr:       float. Umbral que define los eventos considerados como extremos en la POT
    mu:        float. Parámetro de ocurrencia media de la distribución de Poisson
    loc:       float. Parámetro de localización de la distribución exponencial
    scale:     float. Parámetro de escala de la distribución exponencial
    
    Salida:
    -------
    x:       array (n,). Cuantil asociado a cada probabilidad de 'q'
    """
    
    # asegurar que 'q' es un array
    q = np.array(q)
    
    # array de resultados
    x = np.zeros_like(q)
    
    # nº máximo de eventos anuales
    kmax = int(stats.poisson.ppf(1 - 1e-5, mu))
    
    # calcular el cuantil para cada probabilidad de no excedencia
    # ecuación: PMFpoiss(kmax) * CDFexp(y)**kmax + ... + PMFpoiss(2) * CDFexp(y)**2 + PMFpoiss(1) * CDFexp(y) - q = 0
    for i, qx in enumerate(q):
        # coeficientes de la ecuación lineal: [PMFpoiss(kmax), ..., PMFpoiss(2), PMFpoiss(1), -q]
        p = np.array([stats.poisson.pmf(k, mu) for k in range(0, kmax + 1)[::-1]])
        p[-1] -= qx
        # raíces de la ecuación: CDFexp(y)
        roots = np.roots(p)
        # encontrar la raíz real con valor positivo
        for root in roots:
            if (np.imag(root) == 0) & (np.real(root) > 0):
                qy = np.real(root)
                break
        
        # valor de exceso asociado a dicho cuantil
        y = stats.expon.ppf(qy, *parsExp)
        # convertir a la variable original
        x[i] = y + thr
        
    return x


# ## Clase

# In[9]:


class POT:
    
    def __init__(self, serie, distDis, distCon):
        """
        Parámetros:
        -----------
        serie:     pd.Series. Serie temporal; el índice debe ser 'datetime' o 'datetime.date'
        distDis:   callable. Función de distribución discreta con la que modelar la ocurrencia anual de eventos
        distCon:   callable. Función de distribución continua con la que modelar la magnitud de los eventos
        """
        
        self.serie = serie
        self.distDis = distDis
        self.distCon = distCon
    
    def select_events(self, thr, plot=True):
        """Selección de eventos para el futuro ajuste de una distribución POT (peaks-over-threshold).

        En primer lugar se seleccionan todos los pasos temporales con valor por encima del umbral. Seguidamente se selecciona úncicamente el máximo de la cada subserie de excesos sobre el umbral, p.ej., el máximo de cada hidrograma de avenida.

        Parámetros:
        -----------
        thr:       float. Umbral a partir del cual se identifica un evento
        plot:      boolean. Si se quiere mostrar un gráfico de línea con la serie temporal y los eventos seleccionados

        Salida:
        -------
        eventos:   pd.Series. Serie de eventos seleccionados; el índice sigue siendo 'datetime' o 'datetime.date'
        """

        # serie preliminar de eventos
        events0 = self.serie[self.serie > thr]

        # diferencia en días entre los eventos consecutivos de 'events0'
        At = events0.index[1:] - events0.index[:-1]
        At = pd.Series([t.days for t in At], index=events0.iloc[:-1].index)

        # serie de eventos independientes
        events = pd.Series()
        To, Tf = [], []
        i = 0
        while i < len(events0) - 2:
            to = events0.index[i]
            for j in range(i, len(events0) - 1):
                tf = events0.index[j]
                if At[tf] > 1:
                    break
            events.loc[events0.loc[to:tf].idxmax()] = events0.loc[to:tf].max()
            i = events0.index.get_loc(tf) + 1

        if plot:
            plt.figure(figsize=(12, 4))
            plt.plot(self.serie, 'steelblue', lw=.5, label='serie')
            plt.hlines(thr, self.serie.index[0], self.serie.index[-1], ls=':', lw=.75,
                       label='umbral')
            plt.scatter(events.index, events, marker='+', s=30, lw=1, c='indianred',
                        label='eventos')
            plt.xlim(self.serie.index[0], self.serie.index[-1])
            plt.legend(loc=8, ncol=3, bbox_to_anchor=(0.1, -0.22, 0.8, 0.1));
        
        self.eventos = events
        return events
    
    def fitDis(self, plot=True):
        """Ajusta la distribución discreta que modela la ocurrencia anual de eventos extremos
        """
        
        # recurrencia (parámetro de la distribución Poisson)
        T = 365 # pasos temporales en la serie correspondiente al intervalo de estudio (1 año)
        mu = T * len(self.eventos) / len(self.serie)
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

            fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
            ax[0].scatter(k, self.distDis.pmf(k, mu), s=5, label='pmf')
            ax[0].set_ylabel('pmf')
            ax[1].scatter(k, self.distDis.cdf(k, mu), s=5, label='cdf')
            ax[1].set_ylabel('cdf');

        self.mu = mu
        return mu
        
    def fitCon(self, thr, plot=True):
        """Ajusta la distribución continua que modela la magnitud de un evento extremo
        
        Parámetros:
        -----------
        thr:       float. Umbral a partir del cual se identifica un evento
        plot:      boolean. Si se quiere mostrar gráficos del ajuste
        
        Salidas:
        --------
        self.parsCon: list. Parámetros ajustados de la distribución continua: localización y escala, si es una exponencial
        """
        
        # serie de excedencias en los eventos
        y = self.eventos - thr

        # ajustar distribución empírica
        ecdfPOT = ECDF(y)
        ecdfs = ecdfPOT(y)

        # ajustar distribución exponencial
        pars = self.distCon.fit(y)
        ys = np.linspace(0, np.ceil(y.max()), 200)
        cdfs = self.distCon.cdf(ys, *pars)
        
        if plot:
            fig, ax = plt.subplots(ncols=2, figsize=(10, 4.5), sharex=True)
            # histograma
            sns.distplot(y, kde=False, ax=ax[0])
            ax[0].set(xlabel='$N-thr$ (m)', ylabel='nº eventos')
            # cdf empírica vs ajuste
            ax[1].plot(ys, cdfs, label='ajuste')
            ax[1].scatter(y, ecdfs, s=1, c='k', label='empírica')
            ax[1].set(xlim=(0, np.ceil(y.max())), ylim=(0), ylabel='CDF', xlabel='$serie-thr$')
            ax[1].legend(loc=4);
            
        self.parsCon = pars
        return pars
        
    def cdf(self, x, thr):
        """Calcula la función de distribución acumulada (CDF, cumulative density function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        thr:       float. Umbral que define los eventos considerados como extremos en la POT

        Salida:
        -------
        cdf:       array (n,). Probabilidad de no excedencia de cada uno de los valores de 'x'
        """

        # asegurar que 'x' es un array
        x = np.array(x)
        # restar el umbral para convertir la variable en el exceso
        y = x - thr
        
        # 'array' donde guardar la probabilidad acumulada
        cdf = np.zeros_like(y)

        # nº máximo de eventos anuales
        kmax = int(self.distDis.ppf(1 - 1e-5, self.mu))
        for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
            cdf += self.distDis.pmf(k, self.mu) * self.distCon.cdf(y, *self.parsCon)**k

        return cdf
    
    def pdf(self, x, thr):
        """Calcula la función de densidad (PDF, probability distribution function) de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        x:         array or list (n,). Valor/es de la variable para los que calcular su probabilidad de no excedencia
        thr:       float. Umbral que define los eventos considerados como extremos en la POT

        Salida:
        -------
        pdf:       array (n,). Probabilidad de cada uno de los valores de 'x'
        """

        # asegurar que 'x' es un array
        x = np.array(x)
        y = x - thr
        # 'array' donde guardar la probabilidad acumulada
        pdf = np.zeros_like(y)

        # nº máximo de eventos anuales
        kmax = int(self.distDis.ppf(1 - 1e-5, self.mu))
        for k in range(0, kmax + 1): # para cada 'k': nº de eventos anuales
            pdf += self.distDis.pmf(k, self.mu) * k * self.distCon.cdf(y, *self.parsCon)**(k - 1) * self.distCon.pdf(y, *self.parsCon)

        return pdf
    
    def ppf(self, q, thr):
        """Calcula el cuantil asociado a una probabilidad de no excedencia de una distribución POT (peaks-over-threshold) compuesta por una Poisson y una exponencial.

        Parámetros:
        -----------
        q:         array or list (n,). Valor/es de probabilidad de no excedencia
        thr:       float. Umbral que define los eventos considerados como extremos en la POT

        Salida:
        -------
        x:       array (n,). Cuantil asociado a cada probabilidad de 'q'
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
            x[i] = y + thr

        return x


# In[ ]:




