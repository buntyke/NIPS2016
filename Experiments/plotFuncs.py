#!/usr/bin/env python

# plotFuncs.py: plot functions for data inspection
# Author: Nishanth Koganti
# Date: 2016/10/27

import GPy
import pydmps
import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from GPy.plotting.matplot_dep.controllers.imshow_controller import ImshowController
from GPy.inference.latent_function_inference import InferenceMethodList, VarDTC, VarDTC_minibatch

def plotTraj(Dataset, plotType = 0, jointIndex = np.arange(7), labels = ['Train','Test'], colors=['b','r']):
    """function to plot multiple joint tracks."""
    timeData = []
    leftData = []
    rightData = []

    LEFT_ANGLE_OFFSET = 1
    RIGHT_ANGLE_OFFSET = 8

    # loop over first plotNum files
    for data in Dataset.values():
        timeData.append(data[:, 0])
        leftData.append(data[:, LEFT_ANGLE_OFFSET+jointIndex])
        rightData.append(data[:, RIGHT_ANGLE_OFFSET+jointIndex])

    jointData = [leftData, rightData]

    # number of joints to plot
    xlabel = 'Time(sec)'
    arms = ['Left', 'Right']
    nJoints = jointIndex.size
    if plotType == 0:
        ylabels = 7*['Joint Angle (rad)']
    else:
        ylabels = 3*['Position (m)']+4*['Angle (rad)']

    # plot all the joint data
    for ind in range(2):
        fig = plt.figure(figsize=(10, 2*nJoints))
        for i, jI in enumerate(jointIndex):
            plt.subplot(nJoints, 1, i+1)

            # plot all the tracks
            for n in range(len(Dataset.values())):
                timeDat = timeData[n]
                nSamples = jointData[ind][n].shape[0]
                plt.plot(timeDat, jointData[ind][n][:, i], label=labels[n], color=colors[n], linewidth=2)

            plt.xlabel(xlabel, fontsize=12, fontweight='bold')
            plt.ylabel(ylabels[i], fontsize=12, fontweight='bold')

            if plotType == 0:
                plt.title('%s Joint %d' % (arms[ind], jI+1), fontsize=15, fontweight='bold')
            else:
                plt.title('%s Pose %d' % (arms[ind], jI+1), fontsize=15, fontweight='bold')

            # plot legend only for 1st sub plot
            if i == 0:
                plt.legend()

        # adjust subplots for legend
        fig.subplots_adjust(top=0.96, right=0.8)
        plt.tight_layout()

    # show all the plots
    plt.show()

def plotTraj2(Dataset, points = None, colors={'Train':'b','Test':'r'}):
    """function to plot multiple joint tracks."""
    timeData = {}
    latentData = {}

    # loop over first plotNum files
    for key,data in Dataset.iteritems():
        timeData[key] = data[:, 0]
        latentData[key] = data[:, 1:]

    # number of latent dims to plot
    xlabel = 'Time(sec)'
    nDim = latentData[key].shape[1]
    ylabels = nDim*['Latent Position']

    # plot all the latent data
    fig = plt.figure(figsize=(10, 2*nDim))
    for i in range(nDim):
        plt.subplot(nDim, 1, i+1)

        # plot all the tracks
        for n,key in enumerate(Dataset.keys()):
            timeDat = timeData[key]
            nSamples = latentData[key].shape[0]
            plt.plot(timeDat, latentData[key][:, i], label=key, color=colors[key], linewidth=2)

        if points:
            plt.plot(points[i][:, 0], points[i][:, 1], 'ob', markersize=15, label='viapoints')

        plt.xlabel(xlabel, fontsize=12, fontweight='bold')
        plt.ylabel(ylabels[i], fontsize=12, fontweight='bold')
        plt.title('Dim %d' % (i+1), fontsize=15, fontweight='bold')

        # plot legend only for 1st sub plot
        if i == 0:
            plt.legend()

    # adjust subplots for legend
    plt.tight_layout()

    # show all the plots
    plt.show()

def plotLatent(model, trainInput, testInput, nPoints=400, wThresh=0.05):
    sTest = 200
    sTrain = 150
    resolution = 50

    testMarker = 's'
    trainMarker = 'o'

    testLabels = [(1,0,0)]*nPoints
    trainLabels = [(0,0,1)]*nPoints

    # get active dimensions
    scales = model.kern.input_sensitivity(summarize=False)
    scales = scales/scales.max()
    activeDims = np.where(scales > wThresh)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # get latent space plot parameters
    plotIndices = [0,1]
    qDim = model.X.mean.shape[1]
    input1, input2 = plotIndices

    # loop over test data
    saveDim = activeDims[0].shape[0]
    testData = np.zeros((testInput.shape[0], saveDim))
    trainData = np.zeros((trainInput.shape[0], saveDim))

    for n in range(trainInput.shape[0]):
        # infer latent position
        xTrain, _ = model.infer_newX(np.atleast_2d(trainInput[n,:]), optimize=True)

        # update parameter
        trainData[n,:] = xTrain.mean[0,activeDims[0]]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    for n in range(testInput.shape[0]):
        # infer latent position
        xTest, _ = model.infer_newX(np.atleast_2d(testInput[n,:]), optimize=True)

        # update parameter
        testData[n,:] = xTest.mean[0,activeDims[0]]
        sys.stdout.write('.')
    sys.stdout.write('\n')

    # compute plot limits
    xmin, ymin = trainData[:, [input1, input2]].min(0)
    xmax, ymax = trainData[:, [input1, input2]].max(0)
    x_r, y_r = xmax-xmin, ymax-ymin
    xmin -= .1*x_r
    xmax += .1*x_r
    ymin -= .1*y_r
    ymax += .1*y_r

    # plot the variance for the model
    def plotFunction(x):
        Xtest_full = np.zeros((x.shape[0], qDim))
        Xtest_full[:, [input1, input2]] = x
        _, var = model.predict(np.atleast_2d(Xtest_full))
        var = var[:, :1]
        return -np.log(var)

    x, y = np.mgrid[xmin:xmax:1j*resolution, ymin:ymax:1j*resolution]
    gridData = np.hstack((x.flatten()[:, None], y.flatten()[:, None]))
    gridVariance = (plotFunction(gridData)).reshape((resolution, resolution))

    varianceHandle = plt.imshow(gridVariance.T, interpolation='bilinear', origin='lower', cmap=cm.gray,
                                extent=(xmin, xmax, ymin, ymax))

    # test and training plotting
    testHandle = ax.scatter(testData[:, input1], testData[:, input2], marker=testMarker, s=sTest, c=testLabels,
                            linewidth=.2, edgecolor='k', alpha=1.)
    trainHandle = ax.scatter(trainData[:, input1], trainData[:, input2], marker=trainMarker, s=sTrain, c=trainLabels,
                            linewidth=.2, edgecolor='k', alpha=1.)

    ax.grid(b=False)
    ax.set_aspect('auto')
    ax.tick_params(axis='both', labelsize=20)
    ax.set_xlabel('Latent Dimension %i' % (input1+1), fontsize=25, fontweight='bold')
    ax.set_ylabel('Latent Dimension %i' % (input2+1), fontsize=25, fontweight='bold')

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    properties = {'weight':'bold','size':25}
    plt.legend([trainHandle, testHandle], ['Train', 'Test'], prop=properties)

    fig.canvas.draw()
    fig.tight_layout()
    fig.canvas.draw()
    plt.show()

    return trainData, testData
