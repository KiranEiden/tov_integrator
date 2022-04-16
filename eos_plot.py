#!/usr/bin/env python3

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

description = "Script for plotting the output of tov_integrator.py."

files_help = """List files to read data from."""
vars_help = """Variables to plot from each of the files. The variable to go on the x-axis should
        come first, followed by the y-axis."""
prof_help = """Read the files in as profile data instead of mass-radius relation data."""
log_help = """Whether the x and y axes respectively should be plotted on a logarithmic scale."""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('files', nargs='+', help=files_help)
parser.add_argument('-v', '--vars', nargs=2, default=['R', 'M'], help=vars_help)
parser.add_argument('-p', '--profiles', action='store_true', help=prof_help)
parser.add_argument('--log', nargs=2, type=bool, default=[False, False], help=log_help)

args = parser.parse_args(sys.argv[1:])

labels = \
{
    "R"  : r"$R~[\mathrm{km}]$",
    "M"  : r"$M~[\mathrm{M_\odot}]$",
    "P_c": r"$P_c~[\mathrm{MeV \cdot fm^{-3}}]$",
    "P"  : r"$P~[\mathrm{MeV \cdot fm^{-3}}]$",
    "eps": r"$\varepsilon~[\mathrm{MeV \cdot fm^{-3}}]$"
}

def load_mr(fname):
    
    R, M, P_c = np.loadtxt(fname).T
    return dict(R=R, M=M, P_c=P_c)
    
def load_prof(fname):
    
    R, P, M, eps = np.loadtxt(fname).T
    return dict(R=R, P=P, M=M, eps=eps)
    
if args.profiles:
    data = map(load_prof, args.files)
else:
    data = map(load_mr, args.files)
    
v1, v2 = args.vars

for d in data:
    plt.plot(d[v1], d[v2])
    
plt.xlabel(labels[v1])
plt.ylabel(labels[v2])

if args.log[0]:
    plt.xscale("log")
if args.log[1]:
    plt.yscale("log")

plt.show()
