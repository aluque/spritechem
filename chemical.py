import numpy as np
import scipy.constants as co
from scipy.interpolate import interp1d
from scipy.integrate import ode


import chemise as ch

H = 80 * co.kilo
NAIR = co.value('Loschmidt constant (273.15 K, 101.325 kPa)') * np.exp(-H / 7.2e3) 
N_N2 = 0.78 * NAIR
N_O2 = 0.22 * NAIR
TEMPERATURE = 200 # in K

def main(field, model, tend):
    # init reaction set
    rs = DryAir(associative_detachment_model=model)
    rs.print_summary()

    n0 = rs.zero_densities(1)
    # approx. density at 80 km
    rs.set_species(n0, 'e', 10 * co.centi**-3)
    rs.set_species(n0, 'N2', N_N2)
    rs.set_species(n0, 'O2', N_O2)
    
    # Boltzmann population of vibrational levels
    hw = 0.28 * co.eV
    kT = co.k * 200    
    for v in range(1, 7):
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
                         rtol=np.full_like(n0, 1e-5))
        
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
            if ((i + 1) % 20 == 0):
                percent = 100 * it / time[-1]
                print(f"...at time t = {1e6 * it:.1f} us ({percent:.1f}%)")
                
            
            n[:, i + 1] = np.squeeze(r.integrate(it))
            rs.qtp_append_single(it, path, n[:, i + 1], en, T)
            


class DryAir(ch.ReactionSet):
    def __init__(self, associative_detachment_model="only-vib", extend=False):
        super(DryAir, self).__init__()
        
        self.compose({'M': {'N2': 1.0, 'O2': 1.0},
                      'N2v': {'N2(v1)': 1.0,
                              'N2(v2)': 1.0,
                              'N2(v3)': 1.0,
                              'N2(v4)': 1.0,
                              'N2(v5)': 1.0,
                              'N2(v6)': 1.0}})

        self.add("e + N2 -> 2 * e + N2+",
                 ch.LogLogInterpolate0("swarm/k025.dat", extend=extend))

        self.add("e + O2 -> 2 * e + O2+", 
                 ch.LogLogInterpolate0("swarm/k042.dat", extend=extend))

        self.add("e + O2 + O2 -> O2- + O2",
                 ch.LogLogInterpolate0("swarm/k026.dat",
                                       prefactor=(1 / co.centi**-3),
                                       extend=extend))

        self.add("e + O2 -> O + O-",
               ch.LogLogInterpolate0("swarm/k027.dat", extend=extend))

        self.add("M + O2- -> e + O2 + M",
               PancheshnyiFitEN(1.24e-11 * co.centi**3, 179, 8.8))
               
        self.add("O2 + O- -> O2- + O",
               PancheshnyiFitEN(6.96e-11 * co.centi**3, 198, 5.6))

        #
        #                 - associative detachment -
        # ======================================================================
        if associative_detachment_model == "pancheshnyi":
            # This is the old rate from Pancheshnyi
            self.add("N2 + O- -> e + N2O",
                     PancheshnyiFitEN(1.16e-12 * co.centi**3, 48.9, 11))

        elif associative_detachment_model == "only-vib":        
            # From Viggiano et al. N2v represents N2(v > 0)
            self.add("N2v + O- -> e + N2O",
                     ch.Constant(3.2e-12 * co.centi**3))
        # ======================================================================
        #
        
        self.add("O2 + O- + M -> O3- + M",
               PancheshnyiFitEN2(1.1e-30 * co.centi**6, 65))

        # Let us simplify and assume that all + ions are converted to O4+
        # within 1 us
        self.add("N2+ -> O4+", ch.Constant(1e9))
        self.add("O2+ -> O4+", ch.Constant(1e9))
        
        self.add("e + O4+ -> ",
               ch.Interpolate0("swarm/rec_electron.dat", zero_value=0.0,
                               extend=extend))
        
        self.add("O- + O4+ + M -> ",
               ch.Interpolate0("swarm/rec_ion.dat",
                               zero_value=0.0, extend=extend))
        
        self.add("O2- + O4+ + M -> ",
               ch.Interpolate0("swarm/rec_ion.dat",
                               zero_value=0.0, extend=extend))
        
        self.add("O3- + O4+ + M -> ",
               ch.Interpolate0("swarm/rec_ion.dat",
                               zero_value=0.0, extend=extend))

        # Add available vibrational levels
        for v in range(1, 7):
            self.add(f"e + N2 -> e + N2(v{v})",
                     ch.Interpolate0(f"swarm/k{3+v:03d}.dat",
                                     extend=extend))
        # the n2(v1res) part of v=1 excitation
        self.add(f"e + N2 -> e + N2(v1)",
                 ch.Interpolate0(f"swarm/k003.dat",
                                 extend=extend))
            
        
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
        return full_like(EN, self.k0 * (self.T0 / T)** self.power)

    def latex(self):
        return (r"$\num{%g} \times (\num{%g} / T)^{\num{%g}}$"
                % (self.k0, self.T0, self.power))


class ETemperaturePower(ch.LogLogInterpolate0):
    def __init__(self, k0, power, *args, T0=300, **kwargs):
        self.k0 = k0
        self.power = power
        self.T0 = T0

        super(ETemperaturePower, self).__init__(*args, **kwargs)
        

    def __call__(self, EN, T):
        Te = 2 * np.exp(self.s(log(EN))) / (3 * co.k)
        return full_like(EN, self.k0 * (self.T0 / Te)** self.power)

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

        return ("$k_0e^{-\\left(\\frac{a}{b + E/n}\\right)^2}$ [%s]"
                % ", ".join(params))


class PancheshnyiFitEN2(ch.Rate):
    def __init__(self, k0, a):
        self.k0 = k0
        self.a = a

    def latex(self):
        self.k_0 = self.k0
        params = ["$%s = \\num{%g}$" % (s, getattr(self, s))
                  for s in ('k_0', 'a')]

        return ("$k_0e^{-\\left(\\frac{E/n}{a}\\right)^2}$ [%s]"
                % ", ".join(params))

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

        return ("$k_0 e^{\\left(\\frac{\Delta G}{kT_i}\\right)}$ [%s]"
                % ", ".join(params))
        # return ("$k_0 e^{\\left(\\frac{\Delta G}{kT_i}\\right)}; T_i=T + \\frac{1}{g}\\frac{E}{n}$ [%s]"
        #         % ", ".join(params))
        

    def __call__(self, EN, T):
        Ti = T + EN / self.g
        
        return self.k0 * np.exp(self.deltag / (co.k * Ti))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m",
                        help="Dissociative attachment model",
                        choices=["only-vib", "pancheshnyi"],
                        default="only-vib")

    parser.add_argument("--field", "-e", action="store", nargs="+",
                        type=float,
                        help="Reduced field in Td")

    parser.add_argument("--tend", "-t", action="store",
                        type=float,
                        help="Final time in seconde")
    
    args = parser.parse_args()

    main(args.field, args.model, args.tend)
