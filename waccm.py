from numpy import *
import scipy.constants as co
from scipy.interpolate import interp1d

COL_OFFSET = 4
class WACCM(dict):
    """ Class to get waccm densitites. """
    SPECIES = dict((x, i + COL_OFFSET) 
                   for i, x in
                   enumerate(['O' ,'O3', 'N', 'NO', 'NO2', 'NO3', 
                              'N2O', 'N2O5', 'H', 'OH', 'H2', 'HO2', 
                              'H2O2', 'HNO3', 'CO']))

    def __init__(self, filename, felectrons=None):
        super(WACCM, self).__init__()

        self.data = loadtxt(filename, comments='!', skiprows=9)
        self.z = self.data[::-1, 0] * co.kilo
        for sname, col in self.SPECIES.items():
            self[sname] = interp1d(self.z, self.data[::-1, col] * co.centi**-3)
            
        if felectrons is not None:
            self.edata = loadtxt(felectrons)
            self['E'] = interp1d(self.edata[:, 1],
                                  self.edata[:, 0] * co.centi**-3)

    def __call__(self, altitude, units=1):
        return dict((k, v(altitude) / units) for k, v in self.items())
