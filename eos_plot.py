#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import configparser as cfgp
import matplotlib.pyplot as plt

description = """Script for plotting the output of tov_integrator.py."""

files_help = """A list of either data files or INI-style configuration files with
        information on how to plot a given curve. Each section of a configuration file should
        specify a data file ('file' key), label ('label' key), or any matplotlib keyword argument
        that can be supplied as a string, float, or int (as key-value pairs). To plot a group of
        files with one section, one can supply a glob pattern as the value associated with the
        'file' key to collect the files. To only plot a subset of files from that group (but print
        a summary that includes all of them), add the 'pfile' key with a glob pattern for
        capturing the list of filenames. Can specify the position of the label (from 'left',
        'right', 'peak', 'underpeak', and 'None') as well using the 'labelpos' key. Setting 'legend'
        equal to True will add the label to the legend. String values can be bare (unquoted)
        or single-quoted in these files.
        """
vars_help = """Variables to plot from each of the files. The variable to go on the x-axis should
        come first, followed by the y-axis."""
prof_help = """Read the files in as profile data instead of mass-radius relation data."""
log_help = """Whether the x and y axes respectively should be plotted on a logarithmic scale."""
xlim_help = """Bounds on the x-axis. If not supplied, will use what matplotlib defaults to."""
ylim_help = """Bounds on the y-axis. If not supplied, will use what matplotlib defaults to."""
sum_help = """Print a tabular summary of neutron star properties calculated for each EoS. Will only
        do so for mass-radius data. Units are M_⊙ (solar masses) for masses, km for radii, and
        1e20 erg / g for compactnesses."""
latex_help = """If a summary is printed, print it in LaTeX format."""
outfile_help = """If supplied, save plot to file instead of showing it."""
constraints_help = """Plot constraints from observational data. The following options are available:
        'PSR J0348+0432' (Antoniadis et al. 2013), 'MSP J0740+6620' (Miller et al. 2021),
        'PSR J0030+0451' (Miller et al. 2019), 'Vela X-1' (Falanga et al. 2015), 'GW170817 S19'
        (Shibata et al. 2019 upper mass limit). Omit arguments to plot all of them."""
constraint_col_help = """Colors to use for the constraints. Will cycle through colors in order."""

parser = argparse.ArgumentParser(description=description)
parser.add_argument('files', nargs='+', help=files_help)
parser.add_argument('-v', '--vars', nargs=2, default=['R', 'M'], help=vars_help)
parser.add_argument('-p', '--profiles', action='store_true', help=prof_help)
parser.add_argument('--log', nargs=2, type=bool, default=[False, False], help=log_help)
parser.add_argument('-x', '--xlim', nargs=2, type=float, default=[None, None], help=xlim_help)
parser.add_argument('-y', '--ylim', nargs=2, type=float, default=[None, None], help=ylim_help)
parser.add_argument('-s', '--summary', action='store_true', help=sum_help)
parser.add_argument('--latex', action='store_true', help=latex_help)
parser.add_argument('-o', '--outfile', default='', help=outfile_help)
parser.add_argument('-c', '--constraints', nargs='*', default=None, help=constraints_help)
parser.add_argument('--constraint_colors', nargs='+', default=['lightcyan', 'blanchedalmond',
        'mistyrose', 'gainsboro', 'black'], help=constraints_help)

args = parser.parse_args(sys.argv[1:])
args.legend = False

labels = \
{
    "R"  : r"$R~[\mathrm{km}]$",
    "M"  : r"$M~[\mathrm{M_\odot}]$",
    "P_c": r"$P_c~[\mathrm{MeV \cdot fm^{-3}}]$",
    "P"  : r"$P~[\mathrm{MeV \cdot fm^{-3}}]$",
    "eps": r"$\varepsilon~[\mathrm{MeV \cdot fm^{-3}}]$"
}

constraints = \
{
    'Vela X-1'      : (2.12, 0.16, 0.16, None, None, None),
    'PSR J0348+0432': (2.01, 0.04, 0.04, None, None, None),
    'MSP J0740+6620': (2.08, 0.07, 0.07, 13.7, 2.6, 1.5),
    'PSR J0030+0451': (1.44, 0.15, 0.14, 13.02, 1.24, 1.06),
    'GW170817 S19' : ('M', 2.3)
}

#############
# Load Data #
#############

def load_mr(fnames):
    """Load list of mass-radius relation data files."""
    
    R = [None]*len(fnames)
    M = [None]*len(fnames)
    P_c = [None]*len(fnames)
        
    for i, fname in enumerate(fnames):
        R[i], M[i], P_c[i] = np.loadtxt(fname).T
    return dict(R=R, M=M, P_c=P_c)
    
def load_prof(fnames):
    """Load list of pressure, mass and energy density profiles."""
    
    R = [None]*len(fnames)
    P = [None]*len(fnames)
    M = [None]*len(fnames)
    eps = [None]*len(fnames)
    
    for i, fname in enumerate(fnames):
        R[i], P[i], M[i], eps[i] = np.loadtxt(fname).T
    return dict(R=R, P=P, M=M, eps=eps)
    
def _try_convert_str(s):
    """Try to convert string to numeric type, return the original string if unsuccessful."""
    
    try:
        return int(s)
    except ValueError:
        pass
        
    try:
        return float(s)
    except ValueError:
        pass
        
    if s == 'True':
        return True
    if s == 'False':
        return False
        
    return s.lstrip("'").rstrip("'")
    
def load_cfg(fname):
    """Load plot configuration file."""
    
    dicts = []
    cfg = cfgp.ConfigParser()
    cfg.read(fname)
    
    for s in cfg.sections():
        
        s = cfg[s]
        d = dict()
        
        pat = os.path.join(os.path.dirname(fname), s['file'])
        d['file'] = glob.glob(pat)
        
        if 'pfile' in s:
            pat = os.path.join(os.path.dirname(fname), s['pfile'])
            d['pfile'] = set(glob.glob(pat))
        else:
            d['pfile'] = set()
            
        processed = {'file', 'pfile'}
        
        for k in s:
            if k not in processed:
                d[k] = _try_convert_str(s[k])
            
        dicts.append(d)
        
    return dicts

conf = []

for fname in args.files:
    if os.path.splitext(fname)[1] == '.ini':
        conf += load_cfg(fname)
    else:
        conf.append(dict(file=[fname]))

if args.profiles:
    data = [load_prof(c['file']) for c in conf]
else:
    data = [load_mr(c['file']) for c in conf]
    
#################
# Generate Plot #
#################

def _plot_label(l, pos, d1, d2, c):
    
    if len(pos.split()) == 2:
        
        x, y = map(float, pos.split())
        plt.text(x, y, l, fontsize=16, color=c['color'],
                horizontalalignment='right', zorder=10)
    
    elif pos == 'left':
        
        idx = np.argmin(d1)
        plt.text(d1[idx]*0.993, d2[idx]*0.985, l, fontsize=16, color=c['color'],
                horizontalalignment='right', zorder=10)
                
    elif pos == 'right':
        
        idx = np.argmax(d1)
        plt.text(d1[idx]*1.007, d2[idx], l, fontsize=16, color=c['color'],
                horizontalalignment='left', zorder=10)
                
    elif pos == 'peak':
        
        idx = np.argmax(d2)
        plt.text(d1[idx], d2[idx]*1.007, l, fontsize=16, color=c['color'],
                horizontalalignment='center', zorder=10)
                
    elif pos == 'underpeak':
        
        idx = np.argmax(d2)
        plt.text(d1[idx], d2[idx]*0.965, l, fontsize=16, color=c['color'],
                horizontalalignment='center', zorder=10)
                
    elif pos == 'None':
        
        return
                
    else:
        
        raise ValueError(f"Invalid label position '{pos}'.")
        
def _plot_constraints():
    
    assert (args.vars[0] == 'R') and (args.vars[1] == 'M'), 'Can only plot constraints for mass-radius plots'
    
    if len(args.constraints) == 0:
        args.constraints = list(constraints.keys())
        
    for i in range(len(args.constraints)):
        
        color = args.constraint_colors[i % len(args.constraint_colors)]
        
        if len(constraints[args.constraints[i]]) == 2:
            
            var, val = constraints[args.constraints[i]]
            if var == 'R':
                plt.axvline(val, linestyle='-.', color=color, alpha=0.7, zorder=0)
            elif var == 'M':
                plt.axhline(val, linestyle='-.', color=color, alpha=0.7, zorder=0)
            continue
        
        M, dMp, dMm, R, dRp, dRm = constraints[args.constraints[i]]
        
        if not any((x is None for x in (R, dRp, dRm))):
            x1, x2 = R+dRp, R-dRp
        else:
            x1, x2 = plt.xlim()
            
        if not any((y is None for y in (M, dMp, dMm))):
            y1, y2 = M+dMp, M-dMp
        else:
            y1, y2 = plt.ylim()
            
        plt.fill([x1, x1, x2, x2], [y1, y2, y2, y1], color=color, alpha=0.7, zorder=0)
    
v1, v2 = args.vars

for c, d in zip(conf, data):
    
    f = c.pop('file')
    l = c.pop('label', '')
    lp = c.pop('labelpos', 'left')
    pf = c.pop('pfile', set())
    pl = c.pop('plabel', l)
    leg = c.pop('legend', False)
    
    c.setdefault('linewidth', 2)
    c.setdefault('color', 'black')
    
    if leg:
        c['label'] = pl
        args.legend = True
    
    if pf:
        irange = [i for i in range(len(f)) if f[i] in pf]
    else:
        irange = range(len(f))
    
    for i in irange:
        
        d1 = d[v1][i]; d2 = d[v2][i]
        plt.plot(d1, d2, **c, zorder=5)
        
        if l:
            _plot_label(pl, lp, d1, d2, c)
                    
    c['file'] = f
    c['label'] = l
    c['labelpos'] = lp
    c['pfile'] = pf
    c['plabel'] = pl
    c['legend'] = leg
    
plt.xlabel(labels[v1], fontsize=15)
plt.ylabel(labels[v2], fontsize=15)

plt.gca().tick_params(axis='x', labelsize=12)
plt.gca().tick_params(axis='y', labelsize=12)

if args.log[0]:
    plt.xscale("log")
if args.log[1]:
    plt.yscale("log")
    
if not any(x is None for x in args.xlim):
    plt.xlim(*args.xlim)
    
if not any(y is None for y in args.ylim):
    plt.ylim(*args.ylim)
    
if args.constraints is not None:
    _plot_constraints()
    
if args.legend:
    plt.legend(fontsize=12)
    
plt.gcf().set_size_inches((15, 10))

if args.outfile:
    plt.gcf().savefig(args.outfile, bbox_inches='tight')
else:
    plt.show()
    
#################
# Print Summary #
#################
    
def _print_row(entries, header=False, divider_len=95):
    """Print row in table in terminal readable format."""
    
    if header:
        print('-'*divider_len)
    
    for i in range(len(entries)):
        entries[i] = entries[i].center(13)
        entries[i] = '|' + entries[i] + '|'
    
    print(*entries)    
    print('-'*divider_len)
    
def _print_row_latex(entries, header=False, footer=False):
    """Print row in table in LaTeX format."""
    
    if header:
        print(r"\begin{tabular}{cccccc}")
        print('\t' + r"\hline"*2)
        
    entries[0] = r"\mathsf{" + entries[0] + "}"
    
    for i in range(len(entries)):
        entries[i] = " $" + entries[i] + "$ "
    
    print('\t' + '&'.join(entries) + r"\\")
    
    if header:
        print('\t' + r"\hline")
        
    if footer:
        print('\t' + r"\hline")
        print(r"\end{tabular}")
    
def _reduce_prop_mat(mat):
    """Reduce matrix of neutron star properties for an EoS to min, max and mean for each."""
    
    return np.array((mat.min(axis=0), mat.max(axis=0), mat.mean(axis=0))).T
    
if args.summary:
    
    G = 6.67430e-8
    Msun_g = 1.98847e33
    km_cm = 1e5
    
    labels = (c['label'] for c in conf)
    labels = map(str, labels)
    idx = np.argsort(list(labels))
    
    if args.latex:
        cols = ['Label', r'M_{\mathrm{TOV}}~[\mathrm{M_\odot}]', r'R_{\mathrm{TOV}}~[\mathrm{km}]',
                r'R_{1.4}~[\mathrm{km}]', r'C_{\mathrm{TOV}}~[\mathrm{erg / g}]',
                r'C_{1.4}~[\mathrm{erg / g}]']
        _print_row_latex(cols, True)
    else:
        cols = ['Label', 'M_{TOV}', 'R_{TOV}', 'R_{1.4}', 'C_{TOV}',
                'C_{1.4}']
        _print_row(cols, True)
    
    for i in idx:
        
        Ms = list(map(np.array, data[i]['M']))
        Rs = list(map(np.array, data[i]['R']))
        ns_prop = np.zeros((len(Ms), 5), dtype=np.float64)
        
        for j in range(len(Ms)):
            
            M = Ms[j]
            R = Rs[j]
            
            i_max = np.argmax(M)
            M_max = M[i_max]
            R_max = R[i_max]
            C_max = G * (M_max*Msun_g) / (R_max*km_cm)
            
            i_1_4 = (M > 1.4).argmax()
            R_1_4 = np.interp(1.4, M[i_1_4-1:i_1_4+1], R[i_1_4-1:i_1_4+1])
            C_1_4 = G * (1.4*Msun_g) / (R_1_4*km_cm)
            
            ns_prop[j, :] = M_max, R_max, R_1_4, C_max/1e20, C_1_4/1e20
        
        if len(ns_prop) == 1:
            ostr = [str(conf[i]['label'])]
            ostr += '{:.2f} {:.1f} {:.1f} {:.1f} {:.1f}'.format(*ns_prop[0]).split()
        else:
            stat = _reduce_prop_mat(ns_prop)
            # Leave out the mean for now
            ostr = \
            [
                str(conf[i]['label']),
                '{:.2f}-{:.2f}'.format(*stat[0]),
                '{:.1f}-{:.1f}'.format(*stat[1]),
                '{:.1f}-{:.1f}'.format(*stat[2]),
                '{:.1f}-{:.1f}'.format(*stat[3]),
                '{:.1f}-{:.1f}'.format(*stat[4])
            ]
        
        if args.latex:
            _print_row_latex(ostr, footer=(i == idx[-1]))
        else:    
            _print_row(ostr)
