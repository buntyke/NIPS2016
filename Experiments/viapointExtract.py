#!/usr/bin/env python

# viapoint_extract.py: python class implementation to extract viapoints
# Author: Nishanth Koganti
# Date: 2016/10/27

# import modules
import argparse
import matplotlib
import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt

# matplotlib default settings
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size':   10}

matplotlib.rc('font', **font)

class ViaPoint:
    """class implementation for via-point extraction."""
    def __init__(self, data, nDims = 1, nViaPoints=6, errThresh=0.05, viaThresh=0.01):
        """initialization function with variable setting."""
        # setting the viapoint parameters
        self.params = {}
        self.nDims = nDims
        self.tDat = data[:, 0]
        self.data = data[:, 1:]
        self.errThresh = errThresh
        self.viaThresh = viaThresh
        self.nViaPoints = nViaPoints
        self.nPoints = self.tDat.shape[0]


    def compute(self):
        """function to compute via-points which is main interface to class."""
        # initialization
        out = np.zeros(self.data.shape)

        # loop over the dimensions
        for j in range(self.nDims):
            # via-Point algo doesn't work for small variations in angle
            if (self.data[:, j]).std() > self.errThresh:
                # smoothing trajectory
                out[:, j], param = self.viapoints(self.data[:, j], self.nViaPoints)
                self.params[j] = param
            else:
                # if angle doesn't change much then via-point not applied
                out[:, j] = self.data[:, j].copy()

        # output data
        out = np.concatenate((np.transpose(np.atleast_2d(self.tDat)),out), axis=1)
        return out, self.params

    def generate(self, params):
        """function to interpolate for given viapoints."""
        # initialization
        out = self.data.copy()

        # loop over the joints
        for j in params.keys():
            out[:, j] = self.interpolate(params[j])

        out = np.concatenate((np.transpose(np.atleast_2d(self.tDat)),out), axis=1)
        return out

    # function to interpolate between via-points
    def interpolate(self, vpoints):
        """function to interpolate between given viapoints."""
        # number of points for interpolation
        nPoints = self.tDat.shape[0]

        # create output array
        oDat = np.zeros(nPoints)

        # looping over pairs of via points
        for via in range(vpoints.shape[0]-1):
            # initial conditions
            t0 = vpoints[via, 0]
            y0 = vpoints[via, 1]
            yd0 = vpoints[via, 2]
            ind0 = np.int(vpoints[via, 3])

            # final conditions
            t1 = vpoints[via+1, 0]
            y1 = vpoints[via+1, 1]
            yd1 = vpoints[via+1, 2]
            ind1 = np.int(vpoints[via+1, 3])

            for ind in range(ind0, ind1+1):
                t = self.tDat[ind]
                oDat[ind] = y0+yd0*(t-t0)+((20.0*y1-20.0*y0-(8.0*yd1+12.0*yd0)*(t1-t0))/(2.0*(t1-t0)**3.0))*(t-t0)**3.0+(((30.0*y0-30.0*y1)+(14.0*yd1+16.0*yd0)*(t1-t0))/(2.0*(t1-t0)**4.0))*(t-t0)**4.0+((12.0*y1-12.0*y0-(6.0*yd1+6.0*yd0)*(t1-t0))/(2.0*(t1-t0)**5.0))*(t-t0)**5.0

        return oDat

    # function for extracting via points
    def viapoints(self, yDat, nVia):
        """function to extract viapoints for given trajectory."""
        # initialization
        nPoints = yDat.shape[0]

        # sampling period
        ts = self.tDat[-1]/nPoints

        # setting first and last via points
        vpoints = np.zeros((2, 4))
        vpoints[0, :] = np.asarray([self.tDat[0], yDat[0], 0.0, 0])
        vpoints[-1, :] = np.asarray([self.tDat[-1], yDat[-1], 0.0, nPoints-1])

        # loop over the via points
        oDat = np.zeros(nPoints)
        for runs in range(nVia-2):
            oDat = self.interpolate(vpoints)

            errors = np.sqrt((yDat-oDat)**2)

            mI = errors.argmax()
            vpoint = np.asarray([self.tDat[mI], yDat[mI],
                                 (yDat[mI+2]-yDat[mI-2])/(4*ts), mI])
            vpoints = np.concatenate((vpoints, np.atleast_2d(vpoint)), axis=0)
            vpoints = vpoints[np.argsort(vpoints[:, 0]), :]

            meanError = errors.mean()
            if meanError < self.viaThresh:
                break

        oDat = self.interpolate(vpoints)

        return oDat, vpoints

    def plot(self, params):
        """funtion to plot the computed results."""
        # plot initialization
        plt.figure()
        nPlots = len(params.keys())

        # looping over viapoints
        for i,j in enumerate(params.keys()):
            # via points extraction
            param = params[j]

            raw = self.data[:,j]
            smooth = self.interpolate(param)

            # plot figures
            plt.subplot(nPlots*100+11+i)

            plt.plot(self.tDat, raw, '-k', linewidth=2, label='raw')
            plt.plot(self.tDat, smooth, '-b', linewidth=2, label='smooth')
            plt.plot(param[:, 0], param[:, 1], 'ob', markersize=15, label='viapoints')

            plt.xlabel('Time', fontsize=12, fontweight='bold')
            plt.ylabel('Traj', fontsize=12, fontweight='bold')
            plt.title('Index %d' % (j+1), fontsize=15, fontweight='bold')
            if i == 0:
                plt.legend()

        plt.tight_layout()
        plt.show()
