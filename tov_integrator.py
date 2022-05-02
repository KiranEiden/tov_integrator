#!/usr/bin/env python3

import os
import sys
import argparse
import warnings
import numpy as np

from rk4 import System, rk4

description = "Script for integrating the Tolman-Oppenheimer-Volkoff equations for a given EoS table."

eos_table_help = """Path to equation of state table file. Should be simple whitespace delimited
        ASCII file with the first five columns being temperature (MeV), baryonic number density
        (1/fm^3), charge fraction (dimensionless), pressure (MeV / fm^3), and energy density
        (MeV / fm^3) respectively."""
central_pressure_range_help = """Range of central pressures to initialize with to generate the
        mass-radius relation. Should be in units of the maximum pressure in the EoS table."""
num_points_help = """The number of central pressures to use, with bounds set by the
        --central_pressure_range or -cpr argument."""
linear_help = """Use linear spacing for the central pressures instead of logarithmic."""
outfile_help = """The name of the output file. Do not supply the path to the output directory; use
        the -d or --outdir argument for that."""
outdir_help = """The path to the output directory. If this argument is not supplied, the output files
        will be located in the same directory as the eos_table."""
profiles_help = """If supplied, will output a profile of pressure, mass, and energy density as a
        function of radius for each choice of central pressure."""
nmtov_help = """If supplied will not calculate the TOV limit, which requires refining the search
        near peak mass."""
mtov_prec_help = """Precision to determine the central pressure associated with the TOV limit to, as
        a fraction of the central pressure estimate."""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('eos_table', help=eos_table_help)
parser.add_argument('-cpr', '--central_pressure_range', nargs=2, type=float, default=(1e-2, 1.),
        help=central_pressure_range_help)
parser.add_argument('-n', '--num_points', type=int, default=100, help=num_points_help)
parser.add_argument('-l', '--linear', action='store_true', help=linear_help)
parser.add_argument('-f', '--outfile', default="mr_rel.txt", help=outfile_help)
parser.add_argument('-d', '--outdir', help=outdir_help)
parser.add_argument('-p', '--profiles', action='store_true', help=profiles_help)
parser.add_argument('--noMtov', action='store_true', help=nmtov_help)
parser.add_argument('--Mtov_prec', type=int, default=1e-6, help=mtov_prec_help)

args = parser.parse_args(sys.argv[1:])

# Constants
G  = 6.67430e-8 # cm**3 / g / s**2
kB = 1.380649e-16 # erg/K
c  = 2.99792458e10 # cm/s

# Unit conversions
MeV_erg = 1.6021766339999998e-06
fm3_cm3 = 1e39
cm_km = 1e-5
g_Msun = 1 / (1.99e33)

# Load EoS table and convert to CGS
x = np.loadtxt(args.eos_table).T
T, nb, Yq, P, ε = x[0:5]
idx = np.argsort(P)

T  = T[idx] / kB * MeV_erg
nb = nb[idx] * fm3_cm3
Yq = Yq[idx]
P  = P[idx] * MeV_erg * fm3_cm3
ε  = ε[idx] * MeV_erg * fm3_cm3

# Generate central pressure sequence
for val in args.central_pressure_range:
    assert val > 0, "Values specifying central pressure bounds must be positive"
    assert val <= 1, "Values specifying central pressure bounds must be <= 1"

if args.linear:
    P_max = P[-1]
    d1 = args.central_pressure_range[0]
    d2 = args.central_pressure_range[1]
    P_c = np.linspace(P_max*d1, P_max*d2, num=args.num_points)
else:
    log_P_max = np.log10(P[-1])
    d1 = np.log10(args.central_pressure_range[0])
    d2 = np.log10(args.central_pressure_range[1])
    P_c = np.logspace(log_P_max+d1, log_P_max+d2, num=args.num_points)
    
# Determine output directory
if args.outdir is None:
    args.outdir = os.path.dirname(args.eos_table)

# Setup for integrator            
def Pprime(r_i, P_i, M_i):
    """ Derivative of P with respect to r. """
    
    ε_i = np.interp(P_i, P, ε)
    return -G * (M_i + 4*np.pi*r_i**3 * P_i / c**2) * (ε_i + P_i) / (r_i * (c**2*r_i - 2*G*M_i))
    
def Mprime(r_i, P_i, M_i):
    """ Derivative of M with respect to r. """
    
    ε_i = np.interp(P_i, P, ε)
    return 4*np.pi * r_i**2 * ε_i / c**2
    
def completed(ivar, dvars, epsilon=P[0]):
    """ Returns 1 if the desired endpoint has been reached, -1 if the step should be halved, and 0 otherwise. """
    
    if dvars[0] < 0.0:
        return -1
    if dvars[0] < epsilon:
        return 1
    return 0

def integrate_TOV(step, P0):
    """Use 4th-order Runge-Kutta to solve the TOV equation for a particular EoS and central pressure."""
    
    r0 = step
    M0 = 4 * np.pi * step**3 * np.interp(P0, P, ε) / c**2
    
    system = System(Pprime, Mprime, init_conds=(r0, P0, M0))
    res = np.array(list(rk4(system, step, completed)))
    return res
    
def calculate_Mtov(step, P_c, M):
    """Determine the TOV limit by refining the search near the peak mass."""

    print("\nCalculating TOV limit...")

    new_P_c = []
    new_results = []
    
    i_peak = np.argmax(M)
    M_tov = M[i_peak]
    
    if (i_peak == 0) or (i_peak == len(M)-1):
        print("TOV limit cannot be determined with this central pressure range.")
        return new_P_c, new_results
        
    thresh = args.Mtov_prec
    P0, P1, P2 = P_c[i_peak-1:i_peak+2]
    
    while (P2 - P0)/P1 > thresh:
        
        P_1_2 = 0.5 * (P0 + P1)
        new_res = integrate_TOV(step, P_1_2)
        new_M = new_res[-1][2]
        
        if new_M > M_tov:
            
            M_tov = new_M
            P2 = P1; P1 = P_1_2
            new_P_c.append(P_1_2)
            new_results.append(new_res)
            continue
            
        P_3_2 = 0.5 * (P1 + P2)
        new_res = integrate_TOV(step, P_3_2)
        new_M = new_res[-1][2]
        
        if new_M > M_tov:
            
            M_tov = new_M
            P0 = P1; P1 = P_3_2
            new_P_c.append(P_3_2)
            new_results.append(new_res)
            continue
            
        P0 = P_1_2
        P2 = P_3_2
    
    print(f"TOV limit: {M_tov*g_Msun:.3f}")
    return new_P_c, new_results

def progress_bar(frac, size=50):
    
    n = round(size*frac)
    bar = '[' + '⊙'*n + ' '*(size-n) + ']'
    if frac < 1.0:
        end = '\r'
    else:
        end = '\n'
    print(bar, f'{round(100*frac)}%', end=end)

# Do integration for each central pressure
print("Calculating M(R)...")

dr = 1e2
results = [None]*len(P_c)
progress_bar(0)
for i, P_ci in enumerate(P_c):
    results[i] = integrate_TOV(dr, P_ci)
    progress_bar((i+1)/len(P_c))

# Calculate TOV limit
if not args.noMtov:
    
    M = np.array([res[-1][2] for res in results])
    new_P_c, new_results = calculate_Mtov(dr, P_c, M)
    P_c = np.append(P_c, new_P_c)
    results += new_results
    idx = np.argsort(P_c)
    
    P_c = P_c[idx]
    results = [results[i] for i in idx]
    
# Get radii and masses for mass-radius relation
R = np.array([res[-1][0] for res in results])
M = np.array([res[-1][2] for res in results])

# Output data
fname = os.path.join(args.outdir, args.outfile)

with open(fname, 'w') as file:
    print("# Radius", "Mass", "Central_Pressure", file=file)
    print("# km", "M_sun", "MeV/fm^3", file=file)
    for i in range(len(P_c)):
        print(R[i]*cm_km, M[i]*g_Msun, P_c[i]/MeV_erg/fm3_cm3, file=file)

if args.profiles:

    for i in range(len(P_c)):

        fname = os.path.join(args.outdir, f"prof_{i}.txt")
        with open(fname, 'w') as file:
            print("# Radius Pressure Mass Energy_Density", file=file)
            print("# cm MeV/fm^3 M_sun MeV/fm^3", file=file)
            for row in results[i]:
                print(row[0]*cm_km,  row[1]/MeV_erg/fm3_cm3,
                      row[2]*g_Msun, np.interp(row[1], P, ε)/MeV_erg/fm3_cm3,
                      file=file)
