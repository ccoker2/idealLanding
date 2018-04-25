#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spacraft Landing Code
Christian Coker
Last update: 7 March 2018
ASE 4543: Spacecraft Design 2
"""
# Standard library imports
import math
# 3rd party imports
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=3)

class centralBody:

    def __init__(self, g0, R, mu):
        self.g0 = g0   # surface gravity (m/s^2)
        self.R  = R    # mean radius     (m)
        self.mu = mu   # gravitational parameter (m^3/s^2)

    def gravity(self, z):
        return self.g0*(self.R/(self.R+z))**2


class lander:
    def __init__(self, initialMass, maxThrust, maxIsp):
        self.mi = initialMass
        self.Tx = maxThrust
        self.Ix = maxIsp


class maneuver:

    def __init__(self, lander, centralBody):

        self.centralBody = centralBody
        self.lander = lander

    def retroBurn(self, state, tst, isp):
        vel = state[0]
        gam = state[1]
        dst = state[2]
        alt = state[3]
        mas = state[4]
        R = self.centralBody.R
        grv = self.centralBody.gravity(alt)
        # avoid division by zero
        if (isp<1):
            mdt=0
        else:
            mdt = tst/(isp*earth.g0)
        # ODE state vector
        dVel = -tst/mas - grv * math.sin(gam)
        dGam = -(grv/vel - vel/(alt+R)) * math.cos(gam)
        dDst = R/(alt+R) * vel * math.cos(gam)
        dAlt = vel * math.sin(gam)
        dMas = -mdt
        return np.array([dVel, dGam, dDst, dAlt, dMas])


class mission:

    def __init__(self, centralBody, lander):

        self.lander = lander
        self.centralBody = centralBody

    def startCircular(self, alt0):
        r0 = alt0 + self.centralBody.R
        vel0 = math.sqrt(self.centralBody.mu/r0)
        gam0 = 0
        dst0 = 0
        # alt0 = r0 - self.centralBody.R
        mas0 = self.lander.mi
        self.T0   = math.pi**2 * math.sqrt( r0**3 / self.centralBody.mu)
        self.ic = [vel0, gam0, dst0, alt0, mas0]
        print("\n")
        print("Initial Velocity:          ", int(vel0))
        print("Initial Flight Path Angle: ", int(gam0))
        print("Initial Distance Traveled: ", int(dst0))
        print("Initial Altitude:          ", int(alt0))
        print("Initial Mass:              ", int(mas0))
        print("Initial Orbital Period     ", int(self.T0))
        return None

    def land(self, timespan, tstFrac):  # land the spacecraft!
        t0 = timespan[0]
        tf = timespan[1]
        n = np.shape(tstFrac)[0]
        Dt = (tf-t0)/n
        # Isp fractions, per the Dr. K approximation
        ispFrac = (1-(np.ones((n,1))-tstFrac) / 5)
        isp = ispFrac*self.lander.Tx
        tst = tstFrac*self.lander.Ix
        # initialize the set of state vectors as n-by-5 matrix
        states = np.zeros((n,6))
        states[0,0:5] = self.ic
        states[0,5] = t0
        obt = maneuver(self.lander, self.centralBody)
        # Euler's method of integration
        i = 1
        while (states[i-1,3]>0) and states[i-1,0]>=0 and (i<n-1):

            # apply retro burn maneuver (or coast) to orbit state instance i-1
            states[i,0:5]= (states[i-1,0:5]
                           + obt.retroBurn(states[i-1,0:5],
                                           tst[i-1],
                                           isp[i-1])*Dt)
            # store the time
            states[i,5] = states[i-1,5] + Dt
            i += 1

        states = states[0:i,:]
        return states, i


if __name__ == "__main__":

    #===========================================================================
    #  DEFINE centralBody instances
    #===========================================================================

    earth = centralBody(9.81, 6371000, 4.0e+14)
    enceladus = centralBody(0.113, 252000, 7.2e+09)

    #===========================================================================
    #  DEFINE SPACECRAFT
    #===========================================================================

    # frenchFryPhantom = lander(10000, 34624, 274) #initialMass (kg), maxThrust (N), maxIsp (s)
    # frenchFryPhantom = lander(1000, 300, 235)
    frenchFryPhantom = lander(1000, 306.927, 269.4)

    #===========================================================================
    #  LANDING
    #===========================================================================
    # rad0 = 3.9135e+05
    alt0 = 10000 #m

    # define mission
    FFPlanding = mission(enceladus,frenchFryPhantom)

    # define starting orbit
    FFPlanding.startCircular(alt0)

    #===========================================================================
    #  SET UP TIME DOMAIN
    #===========================================================================

    period = FFPlanding.T0
    tf = period
    t0 = 0
    Dt = 0.1                # stepsize, s
    n =  int((tf-t0)/Dt)  # number of steps
    timespan = [t0,tf]

    #===========================================================================
    #  THRUST HISTORY
    #===========================================================================

    # add de-orbit burn to thrust history
    firstBurnLength = 15  # seconds
    j = int(firstBurnLength/Dt)
    tstFrac = np.zeros((n,1))
    tstFrac[1:j] = 0.6
    del j
    #===========================================================================
    #  CALCULATE DE-ORBIT MANEUVER
    #===========================================================================

    data, i = FFPlanding.land(timespan, tstFrac)

    #===========================================================================
    #  CALCULATE LANDING BURN
    #===========================================================================
    v1 = data[-1,0]  #final velocity after de-orbit
    m1 = data[-1,4]  #final mass after de-orbit (near-trivial difference from initial mass)
    vf = 0           #final velocity desired
    maxThrust = FFPlanding.lander.Tx
    # finalBurnLength = (v1-vf)/(maxThrust/m1 - FFPlanding.centralBody.g0)
    # print(finalBurnLength)
    finalBurnLength = 2800
    # print(finalBurnLength)
    # finalBurnLength = 2500
    j = finalBurnLength/Dt
    k = int(i-j)
    #add suicide burn to thrust history
    tstFrac[k:-1] = 1
    del data
    del i
    data, i = FFPlanding.land(timespan,tstFrac)


    #===========================================================================
    #  CONVERT RESULTS TO CARTESIAN COORDINATES
    #===========================================================================

    d = data[:,2]
    r = data[:,3] + enceladus.R
    theta = d/r

    xTraj = r*np.sin(theta)/1000
    yTraj = r*np.cos(theta)/1000

    xSurf = enceladus.R * np.sin(np.linspace(0,2*math.pi, 1000))/1000
    ySurf = enceladus.R * np.cos(np.linspace(0,2*math.pi, 1000))/1000


    #===========================================================================
    #  PARSE THE DATA
    #===========================================================================

    vel = data[:,0]
    gam = data[:,1]*180/math.pi
    dst = data[:,2]
    alt = data[:,3]
    mas = data[:,4]
    tim = data[:,5]

    print("\n")
    print("Final Velocity:          ", float(vel[-1]))
    print("Final Flight Path Angle: ", float(gam[-1]))
    print("Final Distance Traveled: ", float(dst[-1]))
    print("Final Altitude:          ", float(alt[-1]))
    print("Final Mass:              ", float(mas[-1]))
    print("Final Time:              ", float(tim[-1]))


    #===========================================================================
    #===========================================================================
    #  PLOTTING
    #===========================================================================
    #===========================================================================

    box = dict(facecolor='white', pad=5, alpha=0)
    labelx = -0.145  # axes coords
    labely = 0.5


    #===========================================================================
    #  TRAJECTORY PLOT
    #===========================================================================

    fig, (traj) = plt.subplots(1, 1)
    fig.subplots_adjust(wspace=0.6,
                        hspace=0.3,
                        top=0.9,
                        right=0.8,
                        bottom=0.1,
                        left=0.16)
    traj.plot(xSurf, ySurf)
    traj.plot(xTraj, yTraj)
    traj.set_xlim([-500, 500])
    traj.set_ylim([-500, 500])
    # traj.set_title('Trajectory')
    traj.set_xlabel('$X, km$', bbox=box)
    temp = traj.set_ylabel('$Y, km$', bbox=box)
    temp.set_rotation(0)
    traj.yaxis.set_label_coords(labelx, labely)
    plt.show()


    #===========================================================================
    #  VELOCITY PLOT
    #===========================================================================

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(wspace=0.6,
                        hspace=0.3,
                        top=0.9,
                        right=0.93,
                        bottom=0.1,
                        left=0.16)
    ax1.plot(tim,vel)
    temp = ax1.set_ylabel('$V, \\frac{m}{s}$', bbox=box)
    temp.set_rotation(0)
    ax1.yaxis.set_label_coords(labelx, labely)

    #===========================================================================
    #  FLIGHT PATH ANGLE PLOT
    #===========================================================================

    ax2.plot(tim, gam)
    ax2.set_xlabel('$t, s$')
    temp = ax2.set_ylabel('$\gamma, ^\circ$', bbox=box)
    temp.set_rotation(0)
    ax2.yaxis.set_label_coords(labelx, labely)
    plt.show()

    fig, (ax3, ax4) = plt.subplots(2, 1)
    fig.subplots_adjust(wspace=0.6,
                        hspace=0.3,
                        top=0.9,
                        right=0.93,
                        bottom=0.1,
                        left=0.16)
    ax3.plot(tim, dst)
    temp = ax3.set_ylabel('$X, km$', bbox=box)
    temp.set_rotation(0)
    ax3.yaxis.set_label_coords(labelx, labely)

    #===========================================================================
    #  ALTITUDE PLOT
    #===========================================================================

    ax4.plot(tim, alt)
    ax4.set_xlabel('$t, s$')
    temp = ax4.set_ylabel('$Z, m$', bbox=box)
    temp.set_rotation(0)
    ax4.yaxis.set_label_coords(labelx, labely)
    plt.show()

    #===========================================================================
    #  MASS PLOT
    #===========================================================================

    fig, (ax5) = plt.subplots(1, 1)
    fig.subplots_adjust(wspace=0.6,
                        hspace=0.3,
                        top=0.9,
                        right=0.93,
                        bottom=0.1,
                        left=0.16)
    ax5.plot(tim, mas)
    temp = ax5.set_ylabel('$M, kg$', bbox=box)
    temp.set_rotation(0)
    ax5.yaxis.set_label_coords(labelx, labely)
    plt.show()

    # #===========================================================================
    # #  THRUST HISTORY PLOT
    # #===========================================================================
    #
    # fig, (thrust) = plt.subplots(1, 1)
    # fig.subplots_adjust(wspace=0.6,
    #                     hspace=0.3,
    #                     top=0.9,
    #                     right=0.8,
    #                     bottom=0.1,
    #                     left=0.16)
    # thrust.plot( np.reshape(100*tstFrac[0:i], (1,-1)), np.reshape(tim, (1,-1)) )
    # thrust.plot(xTraj, yTraj)
    # thrust.set_xlim([0, tim[-1]])
    # thrust.set_ylim([0, 100])
    # thrust.set_title('Thrust History')
    # thrust.set_xlabel('$t, s$', bbox=box)
    # temp = thrust.set_ylabel('$\% Max Thrust$', bbox=box)
    # temp.set_rotation(0)
    # thrust.yaxis.set_label_coords(labelx, labely)
    # plt.show()
