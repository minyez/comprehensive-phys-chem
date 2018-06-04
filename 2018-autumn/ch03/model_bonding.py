#!/usr/bin/env python
# coding=utf-8

# ====================================================
#     File Name : model_bonding.py
# Creation Date : 18-04-2018
#    Created By : Min-Ye Zhang
#       Contact : stevezhang@pku.edu.cn
# ====================================================

from __future__ import print_function
import numpy as np
from scipy.optimize import fsolve
import scipy.constants as scc
import matplotlib.pyplot as plt
import sys

def pos_sin_minus_lin(t, lin_coeff):
    return np.sin(t)-np.multiply(lin_coeff,t)

def neg_sin_minus_lin(t, lin_coeff):
    return -np.sin(t)-np.multiply(lin_coeff,t)

def find_root(lin_coeff):
    '''
    find the root of $\pm\sin(t)=lin_coeff*t$ by scipy.fsolve
    '''
    t0_pos_valid = []
    t0_pos_invalid = []
    t0_neg_valid = []
    t0_neg_invalid = []
   
    # iterate over reasonable starting point
    for i in range(int(np.ceil(1/lin_coeff/pi))):
        t0 = fsolve(pos_sin_minus_lin, x0=pi*(i+1), args=(lin_coeff), xtol=1E-3)
        if t0 < 1/lin_coeff and t0 != 0.0:
            # check if t0 is in the II/IV quadrant
            if np.ceil(t0*2/pi)%2 == 0:
                t0_pos_valid.append(t0[0].round(3))
            else:
                t0_pos_invalid.append(t0[0].round(3))
        t0 = fsolve(neg_sin_minus_lin, x0=pi*(i+1), args=(lin_coeff), xtol=1E-3)
        if t0 < 1/lin_coeff and t0 != 0.0:
            # check if t0 is in the II/IV quadrant
            if np.ceil(t0*2/pi)%2 == 0:
                t0_neg_valid.append(t0[0].round(3))
            else:
                t0_neg_invalid.append(t0[0].round(3))

    return t0_pos_valid, t0_pos_invalid, t0_neg_valid, t0_neg_invalid

def __set_ax_tick(subplot_ax, linewidth=4, labelsize=12):
    
    for axis in ['top','bottom','left','right']:
        subplot_ax.spines[axis].set_linewidth(linewidth)

    subplot_ax.tick_params(axis='both', which='major', length=linewidth*2, \
                           width=linewidth/2, direction='in', labelsize=labelsize)
    subplot_ax.tick_params(axis='both', which='minor', length=linewidth, \
                           width=linewidth/2, direction='in')

# ====================================================
# Main program
# ====================================================
def c3p2_bond_model():
    '''
    This program solves and shows the bound state solution of the model 
    hamiltonian defined in Prolem 2 of Chapter 3, C.P.C. textbook.
    '''

    # Define physical constants
    pi   = scc.pi
    hbar = scc.hbar
    e = scc.e
    bohr_in_AA = scc.physical_constants['atomic unit of length'][0] * 1E10
    # set m to the reduced mass of a hydrogen molecule
    m    = 1.008*scc.m_u/2.
    
    # ====================================================
    # Define question-specific variables
    # the starting point a1 in \AA
    a1_in_AA = bohr_in_AA
    
    # V0_in_red_unit is V0 in the reduced unit of \hbar^2/2m\AA^2
    if len(sys.argv) == 2:
        V0_in_red_unit = float(sys.argv[1]) * e /hbar**2/1.0E20*2*m
        # set a to the Bohr radius as default
        a_in_AA = bohr_in_AA
    elif len(sys.argv) == 3:
        V0_in_red_unit = float(sys.argv[1])
        a_in_AA = float(sys.argv[2]) * bohr_in_AA
    else:
        a_in_AA = bohr_in_AA
        V0_in_red_unit = ((pi/2 + 0.1)/a_in_AA)**2
    
    a2_in_AA = a1_in_AA + a_in_AA
    a = a_in_AA * 1.0E-10
    V0 = V0_in_red_unit * hbar**2/2./m / 1.0E-20
    V0_in_eV = V0/e
    
    lin_coeff = 1/np.sqrt(V0_in_red_unit)/a_in_AA
    
    # ====================================================
    # find and print the roots
    t0_pos_valid, t0_pos_invalid, t0_neg_valid, t0_neg_invalid = find_root(lin_coeff)
    
    t_solve_array = []
    t_solve_array.extend(t0_pos_valid)
    t_solve_array.extend(t0_neg_valid)
    t_solve_array = np.sort(np.array(t_solve_array))
    
    k1_sol = np.divide(t_solve_array, a)
    k1_sol_in_invAA = np.multiply(k1_sol, 1E-10)
    E_plus_V0_sol = np.divide(np.multiply(np.power(k1_sol, 2), np.power(hbar,2)), 2*m)
    E_sol = np.subtract(E_plus_V0_sol, V0)
    E_sol_in_eV = np.divide(E_sol, e)
    K2_sol = np.sqrt(-E_sol*2*m/np.power(hbar,2))
    K2_sol_in_invAA = np.multiply(K2_sol, 1E-10)
    
    # normalization factors
    # 1 = A^2[a/2 - \sin(2*k1a)/4k1 + sin^2k1a/2K2 * \exp(-2K2a)]
    norm_A_in_sqrtAA_inv_sq = np.subtract(a_in_AA/2, np.divide(np.sin(2 * t_solve_array),4*k1_sol_in_invAA))  + \
                              np.divide(np.power(np.sin(t_solve_array),2),2*K2_sol_in_invAA)
    norm_A_in_sqrtAA = 1/np.sqrt(norm_A_in_sqrtAA_inv_sq)
    # B = A\sin(k1a)\exp(K2a2)
    norm_B_in_sqrtAA = np.multiply(norm_A_in_sqrtAA, np.multiply(np.sin(t_solve_array), np.exp(K2_sol_in_invAA*a2_in_AA)))
    print("Solution E (eV) [with a=%5.3f (A) V_0=%8.5f (eV)]:" % (a_in_AA, V0_in_eV))
    for i in range(len(E_sol_in_eV)):
        print("    E_%d: %8.5f" % (i, E_sol_in_eV[i]))
    
    # ====================================================
    # plot parameters
    tgrid = 1000
    xgrid_per_AA = 200
    eps = 10.0/tgrid
    tlimit_in_twopi = np.ceil(t_solve_array.max()/2/pi) 
    tlimit = tlimit_in_twopi * pi * 2
    x_extend_from_a1 = 3
    xlimit = a1_in_AA + x_extend_from_a1*a_in_AA
    
    # ====================================================
    # Visualization
    fig, axs = plt.subplots(1, 2,figsize=(8,8))
    
    t = np.linspace(0, tlimit, tgrid)
    sint_p = np.sin(t)  
    sint_m = -np.sin(t)
    lin_t = np.multiply(t, lin_coeff)
    
    # == Plotting the root finding process ==
    axs[0].plot(t, sint_p, '-b', label='$\sin(t)$')
    axs[0].plot(t, sint_m, '--b', label='$-\sin(t)$')
    axs[0].plot(t, lin_t, 'black', label='$\hbar t/a\sqrt{2mV_0}$')
    
    axs[0].plot(t0_pos_valid  ,  np.sin(t0_pos_valid)  , 'og', markersize=10)
    axs[0].plot(t0_neg_valid  , -np.sin(t0_neg_valid)  , 'og', markersize=10, label=r'Solution $t$')
    axs[0].plot(t0_pos_invalid,  np.sin(t0_pos_invalid), 'xr', markersize=10)
    axs[0].plot(t0_neg_invalid, -np.sin(t0_neg_invalid), 'xr', markersize=10)
    
    axs[0].legend(loc='lower right', fancybox=True, shadow=True, fontsize=14)
    axs[0].axhline(0, c='black', linewidth=2)
    # set y limit
    axs[0].set_xlim([0, tlimit])
    axs[0].set_xlabel(r'$t=k_1a$', fontsize=14)
    axs[0].set_ylim([-0.2, 1.1])
    axs[0].set_ylabel(r'$f$', fontsize=14)
    axs[0].set_title(r"$a=%5.3f$ $\mathrm{\AA}$, $V_0=%8.5f$ eV" % (a_in_AA, V0_in_eV), fontsize=14)
    # shade the II/IV quadrant where the roots lie
    axs[0].fill_between(t, -0.3, 1.2, where=np.ceil(2*t/pi)%2==1, facecolor='grey')
    
    # == plotting the solution wave function ==
    # set the x grid
    x_I   = np.linspace(0, a1_in_AA, int(a1_in_AA*xgrid_per_AA))
    x_II  = np.linspace(a1_in_AA, a2_in_AA, int(a_in_AA*xgrid_per_AA))
    x_III = np.linspace(a2_in_AA, xlimit, int((xlimit-a2_in_AA)*xgrid_per_AA))
    x     = np.append(x_I,x_II)
    x     = np.append(x,x_III)
    
    for i in range(len(t_solve_array)):
        # the solution \psi =
        #   - I  (0<x<a_1)  :   0
        #   - II (a_1<x<a_2):   A\sin(k1(x-a1))
        #   - III(x>a_2)    :   B\exp(-K2x)
        psi_I = np.zeros(len(x_I))
        psi_II = norm_A_in_sqrtAA[i] * np.sin(k1_sol_in_invAA[i]*(x_II - a1_in_AA))
        psi_III = norm_B_in_sqrtAA[i] * np.exp(-K2_sol_in_invAA[i]*x_III)
        psi = np.append(psi_I, psi_II)
        psi = np.append(psi, psi_III)
        axs[1].plot(x, psi, label='$\psi_%d, E=%6.3f$ eV' % (i, E_sol_in_eV[i]))
    
    yrange = norm_A_in_sqrtAA.max()*1.1
    axs[1].set_ylim([-yrange, yrange])
    axs[1].set_xlim([0, xlimit])
    axs[1].axhline(0, c='black', lw=2)
    axs[1].axvline(a1_in_AA, 0.25, c='black', linestyle='--', lw=2)
    axs[1].axvline(a2_in_AA, 0.25, 0.5, c='black', linestyle='--', lw=2)
    axs[1].axhline(-yrange/2, a1_in_AA/xlimit, a2_in_AA/xlimit, linestyle='--', c='black', lw=2)
    axs[1].set_xlabel(r'$x$ ($\mathrm{\AA}$)', fontsize=14)
    axs[1].set_ylabel(r'Normalized $\psi$', fontsize=14)
    axs[1].set_title(r"$a_1=%5.3f$ $\mathrm{\AA}$, $a_2=%5.3f$ $\mathrm{\AA}$" % (a1_in_AA, a2_in_AA), fontsize=14)
    axs[1].legend(loc='lower right', fancybox=True, shadow=True, fontsize=14)
    
    for ax in axs:
        __set_ax_tick(ax, linewidth=4, labelsize=14)
    
    plt.show()
    #fig.savefig('model_bonding_1_eV.eps',dpi=300)


if __name__ == "__main__":
    c3p2_bond_model()

    
