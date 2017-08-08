# -*- coding: utf-8 -*-
"""
A tool for interpolating across isochrones
"""
from __future__ import (absolute_import, unicode_literals, division,
                        print_function)

__all__ = ['IsochroneSet', 'PARSECIsochrones']

import copy
from abc import ABCMeta, abstractmethod, abstractproperty

import numpy as np

from astropy import units as u
from astropy.coordinates import Distance
from astropy.utils.compat.misc import override__dir__


class IsochroneSet(object):
    """
    One or more isochrones stored as arrays.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, isoc_table, color_name=None, mag_name=None):
        #should be overrridden to set isoc_table
        self.isoc_table = isoc_table

        self.color_name = color_name
        self.mag_name = mag_name
        self.zsun = None

    def __getattr__(self, key):
        if 'isoc_table' not in self.__dict__:
            raise AttributeError('No isoc_table present')

        if key in self.isoc_table.colnames:
            return self.isoc_table[key]
        else:
            raise AttributeError("'{0}' object has no attribute '{1}'".format(self.__class__.__name__, key))

    @override__dir__
    def __dir__(self):
        return self.isoc_table.colnames

    @abstractmethod
    def get_mag(self, bandname):
        """
        Given a band name, return the array of magnitudes for these isochrones.
        """
        raise NotImplementedError

    @abstractproperty
    def age(self):
        """
        the age of the points on the isochrones as an astropy quantity (with time units)
        """
        raise NotImplementedError

    @abstractproperty
    def zmet(self):
        """
        the metallicity of the points on the isochrones as an array
        """
        raise NotImplementedError

    @property
    def feh(self):
        if self.zsun is None:
            raise ValueError('zsun is not set, cannot determine [Fe/H]')
        return np.log10(self.zmet) - np.log10(self.zsun)

    def get_cmd(self):
        if self.color_name is None or self.mag_name is None:
            raise ValueError('color_name or mag_name not set')

        c1nm, c2nm = self.color_name.split('-')
        c1 = self.get_mag(c1nm)
        c2 = self.get_mag(c2nm)
        mag = self.get_mag(self.mag_name)

        return c1 - c2, mag

    def interpolate(self, key_colnm, intended, output_colnm, filt=None):
        """
        Interpolate over an isochrone to sample from a requested set of masses.

        Parameters
        ----------
        key_colnm : str or array
            The column to use as the "x" axis for interpolation, or an array for
            this purpose.
        intended : array
            The output "x" values to interpolate on to.
        output_colnm : str or array
            The isochrone column to interpolate on to, or an array for
            this purpose.
        filt : None or bool array
            A filter to apply to all of the arrays before computing.  Often this
            is the output of ``self.select_isochrone_*``.
        """
        if isinstance(key_colnm, basestring):
            x_col = getattr(self, key_colnm)
        else:
            x_col = key_colnm

        if isinstance(output_colnm, basestring):
            out_col = getattr(self, output_colnm)
        else:
            out_col = output_colnm

        sorti = np.argsort(x_col)
        if filt is None:
            msk = slice(None)
        else:
            msk = filt[sorti]

        intended = np.asarray(intended)
        return np.interp(intended, x_col[sorti][msk], out_col[sorti][msk])

    def select_isochrone_z(self, age, z):
        amsk = self.age == age
        if np.sum(amsk) < 1:
            raise ValueError('No isochrone elements have age '+str(age))
        zmsk = self.zmet == z
        if np.sum(zmsk) < 1:
            raise ValueError('No isochrone elements have z '+str(z))
        return amsk & zmsk

    def select_isochrone_feh(self, age, feh):
        amsk = self.age == age
        if np.sum(amsk) < 1:
            raise ValueError('No isochrone elements have age '+str(age))
        zmsk = self.feh == feh
        if np.sum(zmsk) < 1:
            raise ValueError('No isochrone elements have feh '+str(feh))
        return amsk & zmsk

    def plot_isocs(self, colorby=None, distance=None, **kwargs):
        from matplotlib import pylab as plt

        color, mag = self.get_cmd()
        if distance is not None:
            mag = mag + Distance(distance).distmod.value
        if colorby is None:
            plt.scatter(color, mag, **kwargs)
        else:
            plt.scatter(color, mag, c=getattr(self, colorby), **kwargs)
            plt.colorbar().set_label(colorby)

        plt.xlabel(self.color_name)
        plt.ylabel(self.mag_name)
        plt.ylim(*plt.ylim()[::-1])

    def unique(self, colname):
        """
        Find the *unique* elements of the isochrone table for the provided
        ``colname``.
        """
        return np.unique(getattr(self, colname))

    def subsample(self, msk):
        new = copy.copy(self)
        new.isoc_table = self.isoc_table[msk]
        return new


class PARSECIsochrones(IsochroneSet):
    """
    Isochrones from http://stev.oapd.inaf.it/cgi-bin/cmd and Bressan+ 12
    """
    #from http://stev.oapd.inaf.it/~lgirardi/cmd_2.7/help.html
    STAGE_NUM_TO_NAME = {0: 'PMS',
                         1: 'MS',
                         2: 'SGB',
                         3: 'RGB',
                         4: 'CHeB1',
                         5: 'CHeB2',
                         6: 'CHeB3',
                         7: 'EAGB',
                         8: 'TPAGB'}

    def __init__(self, fns, color_name=None, mag_name=None):
        from astropy.io import ascii

        lines = []
        headerline = None
        for fn in fns:
            headerchecked = lastcomment = None
            with open(fn) as f:
                for l in f:
                    if l.startswith('#'):
                        lastcomment = l
                        continue
                    if headerchecked:
                        lines.append(l)
                    else:
                        if 'log(age/yr)' in l:
                            thisheader = l
                        else:
                            # the header is commented, and we are on a data line
                            # so take the final comment line instead
                            thisheader = lastcomment[1:]
                            lines.append(l)

                        thisheader = '\t'.join(thisheader.strip().split())
                        if headerline is None:
                            headerline = thisheader
                        elif thisheader != headerline:
                            raise ValueError('Header for file "{0}" does not '
                                             'match previous files'.format(fn))
                        headerchecked = True

        lines.insert(0, headerline)
        isoc_table = ascii.read(lines, guess=False, delimiter='\s')

        super(PARSECIsochrones, self).__init__(isoc_table, color_name, mag_name)
        self.zsun = .0152

    def get_mag(self, magname):
        return self.isoc_table[magname]

    @property
    def age(self):
        if 'log(age/yr)' in self.isoc_table.colnames:
            return 10**(self.isoc_table['log(age/yr)']-6) * u.Myr
        elif 'Age' in self.isoc_table.colnames:
            return self.isoc_table['Age']/1e6 *u.Myr
        else:
            raise ValueError('Could not find an age-related column in these isochrones')


    @property
    def logage(self):
        """
        log of age in yr
        """
        return np.log10(self.age/u.yr)

    @property
    def zmet(self):
        return self.isoc_table['Z']

    def stage_names(self):
        maxstrlen = max([len(s) for s in self.STAGE_NUM_TO_NAME.values()])
        namearr = np.zeros(np.max(self.STAGE_NUM_TO_NAME.keys())+1, dtype='S'+str(maxstrlen))
        for k, v in self.STAGE_NUM_TO_NAME.items():
            namearr[k] = v
        return namearr[self.isoc_table['stage']]
    def sample_IMF(self, fieldnamesorvals, nperisoc, filt=None):
        """
        samples each unique age, z pairs and generates ``nperisoc`` samples
        of the requested ``fields``
        """
        fields = []
        for f in fieldnamesorvals:
            if isinstance(f, basestring):
                fields.append(getattr(self, f))
            else:
                fields.append(f)

        uage = self.unique('age')
        uz = self.unique('zmet')
        int_IMF = self.int_IMF

        resultdct = {}
        flatfields = [[] for _ in fields]
        flatzs = []
        flatages = []
        for z in uz:
            for age in uage:
                msk = self.select_isochrone_z(age, z)
                if filt is not None:
                    msk = filt & msk

                if np.sum(msk) > 0:
                    miniimf_msk = np.min(int_IMF[msk])
                    maxiimf_msk = np.max(int_IMF[msk])
                    int_IMF_to_interp = np.linspace(miniimf_msk, maxiimf_msk, nperisoc)
                    resultdct[(age, z)] = resi = []
                    for i, f in enumerate(fields):
                        resi.append(self.interpolate('int_IMF', int_IMF_to_interp, f, msk))
                        flatfields[i].extend(resi[-1])
                    flatzs.extend([z]*len(resi[-1]))
                    flatages.extend([age.to(u.Gyr).value]*len(resi[-1]))

        return resultdct, flatfields, flatzs, flatages*u.Gyr
