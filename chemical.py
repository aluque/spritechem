import sys

import numpy as np
import scipy.constants as co
from scipy.interpolate import interp1d
from scipy.integrate import ode
import string
from waccm import WACCM

import chemise as ch

H = 80 * co.kilo
NAIR = co.value('Loschmidt constant (273.15 K, 101.325 kPa)') * np.exp(-H / 7.2e3) 
N_N2 = 0.78 * NAIR
N_O2 = 0.22 * NAIR
TEMPERATURE = 200 # in K
Td = 1.0e-21

def main(field, model, tend, latex=False):
    # init reaction set
    rs = DryAir(associative_detachment_model=model)
    rs.print_summary()

    waccm = WACCM("waccm_fg_l38.dat")
    W = waccm(H)
    
    if latex:
        write_latex(rs)
        print("LaTeX file written")
        sys.exit(0)

        
    n0 = rs.zero_densities(1)

    # approx. density at 80 km
    rs.set_species(n0, 'e', 10e4 * co.centi**-3)
    rs.set_species(n0, 'N2', N_N2)
    rs.set_species(n0, 'O2', N_O2)
    for s in ['O3', 'H2', 'CO', 'O', 'NO', 'NO2']:
        try:
            rs.set_species(n0, s, W[s])
        except KeyError:
            pass

    # See e.g. Emmert 2012. below ~80 km ppm is about the same as at ground
    # level
    # rs.set_species(n0, 'CO2', 370e-6 * NAIR)
    
    # Boltzmann population of vibrational levels
    hw = 0.28 * co.eV
    kT = co.k * 200    
    for v in range(1, 3):
        rs.set_species(n0, f'N2(v{v})', N_N2 * np.exp(-v * hw / kT))
        
    en = np.array([0.0])
    T = np.array([TEMPERATURE])
    
    time = np.linspace(0, tend, 1 + int(100 * tend / 1e-3))
    n = rs.zero_densities(len(time))


    # closure to compute derivatives
    def f(t, n):
        return rs.fderivs(n[:, np.newaxis], en, T)

    # closure to compute jacobian
    def jac(t, n):
        return rs.fjacobian(n, en, T)

    for en1 in field:
        r = ode(f)
        r.set_integrator('vode', method='bdf', nsteps=2000,
                         rtol=np.full_like(n0, 1e-8))
        
        n[:, 0] = np.squeeze(n0)
        r.set_initial_value(n0, time[0])

        path = f"output/{model}/{int(en1):03.0f}Td"
        print(f"computing E/n = {en1} [{path}]")
        en[0] = en1

        # init QtPlaskin
        rs.qtp_init_single(path, ["Reduced field"])    
        rs.qtp_append_single(0.0, path, n[:, 0], en, T)
    
        # time-marching
        for i, it in enumerate(time[1:]):
            if ((i + 1) % 500 == 0):
                percent = 100 * it / time[-1]
                print(f"...at time t = {1e6 * it:.1f} us ({percent:.1f}%)")
                
            
            n[:, i + 1] = np.squeeze(r.integrate(it))
            rs.qtp_append_single(it, path, n[:, i + 1], en, T)
            


class DryAir(ch.ReactionSet):
    def __init__(self, associative_detachment_model="only-vib", extend=False):
        super(DryAir, self).__init__()
        
        self.compose({'M': {'N2': 1.0, 'O2': 1.0}})

        self.add("e + N2 -> 2 * e + N2+",
                 ch.LogLogInterpolate0("swarm/k025.dat", extend=extend),
                 ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")
    
        self.add("e + O2 -> 2 * e + O2+", 
                 ch.LogLogInterpolate0("swarm/k042.dat", extend=extend),
                 ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")

        self.add("e + O2 + O2 -> O2- + O2",
                 ch.LogLogInterpolate0("swarm/k026.dat",
                                       prefactor=(1 / co.centi**-3),
                                       extend=extend),
                 ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")

        self.add("e + O2 -> O + O-",
                 ch.LogLogInterpolate0("swarm/k027.dat", extend=extend),
                 ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")

        self.add("M + O2- -> e + O2 + M",
                 PancheshnyiFitEN(1.24e-11 * co.centi**3, 179, 8.8),
                 ref="Pancheshnyi2013/JPhD")
               
        self.add("O2 + O- -> O2- + O",
                 PancheshnyiFitEN(6.96e-11 * co.centi**3, 198, 5.6),
                 ref="Pancheshnyi2013/JPhD")

        #
        #                 - associative detachment -
        # ======================================================================
        if associative_detachment_model == "rm78":
            # This is the old rate from Pancheshnyi
            self.add("N2 + O- -> e + N2O",
                     PancheshnyiFitEN(1.16e-12 * co.centi**3, 48.9, 11))

        elif associative_detachment_model == "current":
            # From Viggiano et al. N2v represents N2(v > 0)
            # self.add("N2v + O- -> e + N2O",
            #          ch.Constant(3.2e-12 * co.centi**3))
            self.add("N2 + O- -> e + N2O", ModArrhenius(3.98e-17, 5097, -1.36),
                     ref="Schuman2023/PhysChem")
            self.add("N2(v1) + O- -> e + N2O", ModArrhenius(9.04e-18, 674, -0.85),
                     ref="Schuman2023/PhysChem")
            self.add("N2(v2) + O- -> e + N2O", ModArrhenius(2.74e-17, 186, -1.10),
                     ref="Schuman2023/PhysChem")
            
            
        # ======================================================================
        #
        
        self.add("O2 + O- + M -> O3- + M",
                 PancheshnyiFitEN2(1.1e-30 * co.centi**6, 65), ref="Pancheshnyi2013/JPhD")

        # Add available vibrational levels
        for v in range(1, 3):
            self.add(f"e + N2 -> e + N2(v{v})",
                     ch.Interpolate0(f"swarm/k{3+v:03d}.dat",
                                     extend=extend),
                     ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")

        # the n2(v1res) part of v=1 excitation
        self.add(f"e + N2 -> e + N2(v1)",
                 ch.Interpolate0(f"swarm/k003.dat",
                                 extend=extend),
                 ref="Lawton1978/JChPh, Phelps1985/PhRvA, Hagelaar2005/PSST")
            
        # Positive ions
        self.add("N2+ + N2 + M -> N4+ + M",
                 TemperaturePower(5e-29 * co.centi**6, 2),
                 ref="Aleksandrov1999/PSST")
        
        self.add("N4+ + O2 -> 2 * N2 + O2+",
                 ch.Constant(2.5e-10 * co.centi**3),
                 ref="Aleksandrov1999/PSST")


        self.add("O2+ + O2 + M -> O4+ + M",
                 TemperaturePower(2.4e-30 * co.centi**6, 3),
                 ref="Aleksandrov1999/PSST")


        # Recombination
        self.add("e + O4+ -> O2 + O2",
                 ch.Interpolate0("swarm/rec_electron.dat", zero_value=0.0,
                                 extend=extend),
                 ref="Kossyi1992/PSST")

        # Add bulk recombination reactions
        pos = [s for s in self.species if '+' in s] 
        neg = [s for s in self.species if '-' in s] 
        self.add_pattern("{pos} + {neg} -> ",
                         {'pos': pos, 'neg': neg},
                         ch.Constant(1e-7 * co.centi**3),
                         generic="A+ + B- -> ", 
                         ref="Kossyi1992/PSST")
        
        # Set of reactions proposed by Nick note change in sign in the
        # exponent because in TemperaturePower I use (T0/T) while he sets (T/T0)

        # self.add("O- + O3 -> O3- + O",
        #          TemperaturePower(1.1e-9 * co.centi**3, -0.14))
        # self.add("O- + O3 -> O2- + O2",
        #          TemperaturePower(1.3e-9 * co.centi**3, -0.14))
        # self.add("O- + CO2 + M -> CO3- + M",
        #          TemperaturePower(3e-28 * co.centi**6, 1.5))
        # self.add("O- + H2 -> H2O + e",
        #          TemperaturePower(6e-10 * co.centi**3, 0.25))
        # self.add("O- + H2 -> OH- + H",
        #          TemperatureExp(8e-11 * co.centi**3, 200 * co.h * co.c / co.k))
        # self.add("O- + CO -> CO2 + e",
        #          TemperaturePower(6e-10 * co.centi**3, 0.20))
        # self.add("O- + O -> O2 + e",
        #          TemperaturePower(1.5e-10 * co.centi**3, 1/2))
        # self.add("O- + O -> O2 + e",
        #          ModArrhenius(1.5e-10 * co.centi**3, 0.0, -1/2),
        #          mgas=15.999 * co.gram / co.Avogadro, K0=4.5 * co.centi**2,
        #          ref="Ard2013/JChPh")
        # self.add("O- + NO -> NO2 + e",
        #          TemperaturePower(2.9e-10 * co.centi**3, 0.7))
        # self.add("O- + NO2 -> NO2- + O",
        #          ch.Constant(2.9e-10 * co.centi**3))
        
        # This one is already accounted for:
        # self.add("O- + O2 + M -> O3- + M",
        #          TemperaturePower(1e-30 * co.centi**6, -1.5))
        self.initialize()


    def init_neutrals(self, n):
        self.set_species(n, 'N2', N_N2)
        self.set_species(n, 'O2', N_O2)

        
        
class TemperaturePower(ch.Rate):
    def __init__(self, k0, power, T0=300):
        self.k0 = k0
        self.power = power
        self.T0 = T0

    def __call__(self, EN, T):
        return np.full_like(EN, self.k0 * (self.T0 / T)** self.power)

    def latex(self):
        return (r"$\num{%g} \times (\num{%g} / T)^{\num{%g}}$"
                % (self.k0, self.T0, self.power))

class TemperatureExp(ch.Rate):
    def __init__(self, k0, T0):
        self.k0 = k0
        self.T0 = T0

    def __call__(self, EN, T):
        return np.full_like(EN, self.k0 * np.exp(self.T0 / T))

    def latex(self):
        return (r"$\num{%g} \times \exp(\num{%g} / T)$"
                % (self.k0, self.T0))
    

class ETemperaturePower(ch.LogLogInterpolate0):
    def __init__(self, k0, power, *args, T0=300, **kwargs):
        self.k0 = k0
        self.power = power
        self.T0 = T0

        super(ETemperaturePower, self).__init__(*args, **kwargs)
        

    def __call__(self, EN, T):
        Te = 2 * np.exp(self.s(log(EN))) / (3 * co.k)
        return np.full_like(EN, self.k0 * (self.T0 / Te)** self.power)

    def latex(self):
        return (r"$\num{%g} \times (\num{%g} / T_e)^{\num{%g}}$"
                % (self.k0, self.T0, self.power))


class PancheshnyiFitEN(ch.Rate):
    def __init__(self, k0, a, b):
        self.k0 = k0
        self.a = a
        self.b = b

    def __call__(self, EN, T):
        return self.k0 * np.exp(-(self.a / (self.b + EN))**2)

    def latex(self):
        self.k_0 = self.k0
        params = ["$%s = \\num{%g}$" % (s, getattr(self, s))
                  for s in ('k_0', 'a', 'b')]

        row1 = "$k_0e^{-\\left(\\frac{a}{b + E/n}\\right)^2}$"
        row2 = "%s" % ", ".join(params)
        return "\\makecell{%s\\\\%s}" % (row1, row2)

class PancheshnyiFitEN2(ch.Rate):
    def __init__(self, k0, a):
        self.k0 = k0
        self.a = a

    def latex(self):
        self.k_0 = self.k0
        params = ["$%s = \\num{%g}$" % (s, getattr(self, s))
                  for s in ('k_0', 'a')]
        row1 = "$k_0e^{-\\left(\\frac{E/n}{a}\\right)^2}$"
        row2 = "%s" % ", ".join(params)
        return "\\makecell{%s\\\\%s}" % (row1, row2)

    def __call__(self, EN, T):
        return self.k0 * np.exp(-(EN / self.a)**2)


class Gallimberti(ch.Rate):
    def __init__(self, kforward, n0, deltag, g):
        self.k0 = kforward * n0
        self.deltag = deltag
        self.g = g

    def latex(self):
        params = ["$%s = \\num{%.3g}$" % (s, v)
                  for s, v in (('k_0', self.k0),
                               ('\Delta G', self.deltag))]

        row1 = "$k_0 e^{\\left(\\frac{\Delta G}{kT_i}\\right)}$"
        row2 = "%s" % ", ".join(params)

        return "\\makecell{%s\\\\%s}" % (row1, row2)

    # return ("$k_0 e^{\\left(\\frac{\Delta G}{kT_i}\\right)}; T_i=T + \\frac{1}{g}\\frac{E}{n}$ [%s]"
        #         % ", ".join(params))
        

    def __call__(self, EN, T):
        Ti = T + EN / self.g
        
        return self.k0 * np.exp(self.deltag / (co.k * Ti))


class ModArrhenius(ch.Rate):
    def __init__(self, k0, T0, d,
                 Tgas=200, mgas=28.02 * co.gram / co.Avogadro,
                 K0=4.5 * co.centi**2):
        self.k0 = k0
        self.T0 = T0
        self.d = d
        self.mgas = mgas
        self.K0 = K0

    def latex(self):
        return ("$\\num{%.2g} \\times (T_\\text{eff}/300)^{%.2f} \\exp(-%d/T_\\text{eff})$ "
                % (self.k0, self.d, self.T0))

    def __call__(self, EN, T):
        vd = NAIR * self.K0 * EN * Td
        Teff = T + self.mgas * vd**2 / (3 * co.k)
        k = self.k0 * (Teff / 300)**self.d * np.exp(-self.T0 / Teff)
        return k
        
        
def write_latex(chem, pdf=False, name="../paper/chemistry.tex"):
    with open("small_template.tex", "r") as ftempl:
        latex_template = string.Template(ftempl.read())

    latex_tbl = chem.latex()
    latex_tbl = latex_tbl.replace("$f(x_0)$", "Bolsig+")
    
    with open(name, 'w') as flatex:
        flatex.write(latex_template.safe_substitute(
            reaction_table=latex_tbl,
            nreactions=str(chem.nreactions),
            nspecies=str(chem.nspecies)))
        
    if pdf:
        subprocess.call("latexmk -pdf %s" % self.name, shell=True)

            
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m",
                        help="Dissociative attachment model",
                        choices=["rm78", "current"],
                        default="current")

    parser.add_argument("--field", "-e", action="store", nargs="+",
                        type=float,
                        help="Reduced field in Td")

    parser.add_argument("--tend", "-t", action="store",
                        type=float,
                        help="Final time in seconds")

    parser.add_argument("--latex", "-L", action="store_true",
                        default=False,
                        help="Produce latex output?")
    
    args = parser.parse_args()

    main(args.field, args.model, args.tend, latex=args.latex)
