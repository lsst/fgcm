from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time
import scipy.optimize

import matplotlib.pyplot as plt


from fgcmUtilities import gaussFunction
from fgcmUtilities import histoGauss

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmGray(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        # need fgcmPars because it tracks good exposures
        #  also this is where the gray info is stored
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        # and record configuration variables...
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.minStarPerExp = fgcmConfig.minStarPerExp
        self.minCCDPerExp = fgcmConfig.minCCDPerExp
        self.maxCCDGrayErr = fgcmConfig.maxCCDGrayErr
        self.sigFgcmMaxErr = fgcmConfig.sigFgcmMaxErr
        self.sigFgcmMaxEGray = fgcmConfig.sigFgcmMaxEGray
        self.ccdGrayMaxStarErr = fgcmConfig.ccdGrayMaxStarErr
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.illegalValue = fgcmConfig.illegalValue
        self.expGrayInitialCut = fgcmConfig.expGrayInitialCut
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber

        self._prepareGrayArrays()

    def _prepareGrayArrays(self):
        """
        """

        # we have expGray for Selection
        self.expGrayForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expNGoodStarForInitialSelectionHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')

        # and the exp/ccd gray for the zeropoints

        self.ccdGrayHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayRMSHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdGrayErrHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.ccdNGoodObsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodStarsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')
        self.ccdNGoodTilingsHandle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')

        self.expGrayHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayRMSHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expGrayErrHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')
        self.expNGoodStarsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i4')
        self.expNGoodCCDsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='i2')
        self.expNGoodTilingsHandle = snmm.createArray(self.fgcmPars.nExp,dtype='f8')

        self.sigFgcm = np.zeros(self.fgcmPars.nBands,dtype='f8')

    def computeExpGrayForInitialSelection(self,noPlots=False):
        """
        """
        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeExpGrayForInitialSelection")

        # Note this computes ExpGray for all exposures, good and bad


        # useful numbers
        expGrayForInitialSelection = snmm.getArray(self.expGrayForInitialSelectionHandle)
        expGrayRMSForInitialSelection = snmm.getArray(self.expGrayRMSForInitialSelectionHandle)
        expNGoodStarForInitialSelection = snmm.getArray(self.expNGoodStarForInitialSelectionHandle)

        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        # first, we need to compute E_gray == <mstd> - mstd for each observation

        # compute all the EGray values

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # only use good observations of good stars...

        # for the required bands
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        # select observations of these stars...
        a,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex])

        # and first, we only use the required bands
        _,reqBandUse = esutil.numpy_util.match(self.fgcmStars.bandRequiredIndex,
                                               obsBandIndex[b])

        # now group per exposure and sum...

        expGrayForInitialSelection[:] = 0.0
        expGrayRMSForInitialSelection[:] = 0.0
        expNGoodStarForInitialSelection[:] = 0

        np.add.at(expGrayForInitialSelection,
                  obsExpIndex[b],
                  EGray[b])
        np.add.at(expGrayRMSForInitialSelection,
                  obsExpIndex[b],
                  EGray[b]**2.)
        np.add.at(expNGoodStarForInitialSelection,
                  obsExpIndex[b],
                  1)

        # loop over the extra bands...
        #  we only want to use previously determined "good" stars
        for extraBandIndex in self.fgcmStars.bandExtraIndex:
            extraBandUse, = np.where((obsBandIndex[b] == extraBandIndex) &
                                     (objNGoodObs[obsObjIDIndex[b],extraBandIndex] >=
                                      self.fgcmStars.minPerBand))

            np.add.at(expGrayForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      EGray[b[extraBandUse]])
            np.add.at(expGrayRMSForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      EGray[b[extraBandUse]]**2.)
            np.add.at(expNGoodStarForInitialSelection,
                      obsExpIndex[b[extraBandUse]],
                      1)


        gd,=np.where(expNGoodStarForInitialSelection > 0)
        expGrayForInitialSelection[gd] /= expNGoodStarForInitialSelection[gd]
        expGrayRMSForInitialSelection[gd] = np.sqrt((expGrayRMSForInitialSelection[gd]/expNGoodStarForInitialSelection[gd]) -
                                             (expGrayForInitialSelection[gd])**2.)


        if (noPlots):
            return

        expUse,=np.where((self.fgcmPars.expFlag == 0) &
                         (expNGoodStarForInitialSelection > self.minStarPerExp) &
                         (expGrayForInitialSelection > self.expGrayInitialCut))

        for i in xrange(self.fgcmPars.nBands):
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            fig=plt.figure(1)
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, expGrayForInitialSelection[expUse[inBand]])

            ax.tick_params(axis='both',which='major',labelsize=14)
            ax.locator_params(axis='x',nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[i]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.5f$' % (coeff[1]) + '\n' + \
                r'$\sigma = %.4f$' % (coeff[2])

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$\mathrm{EXP}^{\mathrm{gray}} (\mathrm{initial})$',fontsize=16)
            ax.set_ylabel(r'# of Exposures',fontsize=14)

            fig.savefig('%s/%s_expgray_initial_%s.png' % (self.plotPath,
                                                          self.outfileBaseWithCycle,
                                                          self.fgcmPars.bands[i]))


    def computeCCDAndExpGray(self,noPlots=False):
        """
        """

        if (not self.fgcmStars.allMagStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")

        # Note: this computes the gray values for all exposures, good and bad

        # values to set
        ccdGray = snmm.getArray(self.ccdGrayHandle)
        ccdGrayRMS = snmm.getArray(self.ccdGrayRMSHandle)
        ccdGrayErr = snmm.getArray(self.ccdGrayErrHandle)
        ccdNGoodObs = snmm.getArray(self.ccdNGoodObsHandle)
        ccdNGoodStars = snmm.getArray(self.ccdNGoodStarsHandle)
        ccdNGoodTilings = snmm.getArray(self.ccdNGoodTilingsHandle)

        expGray = snmm.getArray(self.expGrayHandle)
        expGrayRMS = snmm.getArray(self.expGrayRMSHandle)
        expGrayErr = snmm.getArray(self.expGrayErrHandle)
        expNGoodCCDs = snmm.getArray(self.expNGoodCCDsHandle)
        expNGoodStars = snmm.getArray(self.expNGoodStarsHandle)
        expNGoodTilings = snmm.getArray(self.expNGoodTilingsHandle)

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)


        # first, we need to compute E_gray == <mstd> - mstd for each observation

        EGray = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGray[obsIndex] = (objMagStdMean[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]] -
                           obsMagStd[obsIndex])

        # and need the error for Egray: sum in quadrature of individual and avg errs
        EGrayErr2 = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        EGrayErr2[obsIndex] = (objMagStdMeanErr[obsObjIDIndex[obsIndex],obsBandIndex[obsIndex]]**2. +
                               obsMagErr[obsIndex]**2.)

        goodObs,=np.where(EGrayErr2[obsIndex] < self.ccdGrayMaxStarErr)

        # only use good observations of good stars...
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)

        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        # select observations of these stars...
        _,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex[goodObs]])
        b = goodObs[b]

        # first, we only use the required bands
        _,reqBandUse = esutil.numpy_util.match(self.fgcmStars.bandRequiredIndex,
                                               obsBandIndex[b])

        # group by CCD and sum

        ## ccdGray = Sum(EGray/EGrayErr^2) / Sum(1./EGrayErr^2)
        ## ccdGrayRMS = Sqrt((Sum(EGray^2/EGrayErr^2) / Sum(1./EGrayErr^2)) - ccdGray^2)
        ## ccdGrayErr = Sqrt(1./Sum(1./EGrayErr^2))


        ccdGray[:,:] = 0.0
        ccdGrayRMS[:,:] = 0.0
        ccdGrayErr[:,:] = 0.0
        ccdNGoodObs[:,:] = 0
        ccdNGoodStars[:,:] = 0
        ccdNGoodTilings[:,:] = 0.0


        # temporary variable here
        ccdGrayWt = np.zeros_like(ccdGray)

        np.add.at(ccdGrayWt,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  1./EGrayErr2[b[reqBandUse]])
        np.add.at(ccdGray,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  EGray[b[reqBandUse]]/EGrayErr2[b[reqBandUse]])
        np.add.at(ccdGrayRMS,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  EGray[b[reqBandUse]]**2./EGrayErr2[b[reqBandUse]])
        np.add.at(ccdNGoodStars,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  1)
        np.add.at(ccdNGoodObs,
                  (obsExpIndex[b[reqBandUse]],obsCCDIndex[b[reqBandUse]]),
                  objNGoodObs[obsObjIDIndex[b[reqBandUse]],
                              obsBandIndex[b[reqBandUse]]])

        # loop over the extra bands
        #  we only want to use previously determined "good" stars
        for extraBandIndex in self.fgcmStars.bandExtraIndex:
            extraBandUse, = np.where((obsBandIndex[b] == extraBandIndex) &
                                     (objNGoodObs[obsObjIDIndex[b],extraBandIndex] >=
                                      self.fgcmStars.minPerBand))

            np.add.at(ccdGrayWt,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      1./EGrayErr2[b[extraBandUse]])
            np.add.at(ccdGray,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      EGray[b[extraBandUse]]/EGrayErr2[b[extraBandUse]])
            np.add.at(ccdGrayRMS,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      EGray[b[extraBandUse]]**2./EGrayErr2[b[extraBandUse]])
            np.add.at(ccdNGoodStars,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      1)
            np.add.at(ccdNGoodObs,
                      (obsExpIndex[b[extraBandUse]],obsCCDIndex[b[extraBandUse]]),
                      objNGoodObs[obsObjIDIndex[b[extraBandUse]],
                                  obsBandIndex[b[extraBandUse]]])


        # need at least 3 or else computation can blow up
        gd = np.where(ccdNGoodStars > 2)
        ccdGray[gd] /= ccdGrayWt[gd]
        ccdGrayRMS[gd] = np.sqrt((ccdGrayRMS[gd]/ccdGrayWt[gd]) - (ccdGray[gd]**2.))
        ccdGrayErr[gd] = np.sqrt(1./ccdGrayWt[gd])

        # check for infinities
        bad=np.where(~np.isfinite(ccdGrayRMS))
        ccdGrayRMS[bad] = 0.0

        # and the ccdNGoodTilings...
        ccdNGoodTilings[gd] = (ccdNGoodObs[gd].astype(np.float64) /
                               ccdNGoodStars[gd].astype(np.float64))

        # and now compute sigFGCM since we have the numbers ready to go

        for bandIndex in xrange(self.fgcmStars.nBands):
            # if we are an extraBand we need an extra check
            if (bandIndex in self.fgcmStars.bandRequiredIndex):
                sigUse,=np.where((np.abs(EGray[b]) < self.sigFgcmMaxEGray) &
                                 (EGrayErr2[b] > 0.0) &
                                 (EGrayErr2[b] < self.sigFgcmMaxErr**2.) &
                                 (EGray[b] != 0.0) &
                                 (obsBandIndex[b] == bandIndex))
            else:
                sigUse,=np.where((np.abs(EGray[b]) < self.sigFgcmMaxEGray) &
                                 (EGrayErr2[b] > 0.0) &
                                 (EGrayErr2[b] < self.sigFgcmMaxErr**2.) &
                                 (EGray[b] != 0.0) &
                                 (obsBandIndex[b] == bandIndex) &
                                 (objNGoodObs[obsObjIDIndex[b],bandIndex] >=
                                  self.fgcmStars.minPerBand))

            #hist = esutil.stat.histogram(EGray[b[sigUse]],binsize=0.0002,more=True)
            #hCenter=hist['center']
            #hHist = hist['hist'].astype('f8')
            #hHist = hHist / hHist.max()

            #p0=[np.sum(hHist),0.0,0.01]

            #coeff,varMatrix = scipy.optimize.curve_fit(gaussFunction, hCenter, hHist, p0=p0)
            #self.sigFgcm[bandIndex] = np.sqrt(coeff[2]**2. - np.median(EGrayErr2[b[sigUse]]))

            fig = plt.figure(1)
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, EGray[b[sigUse]])

            self.sigFgcm[bandIndex] = np.sqrt(coeff[2]**2. -
                                              np.median(EGrayErr2[b[sigUse]]))

            if (not np.isfinite(self.sigFgcm[bandIndex])):
                print("Failed to compute sigFgcm (%s).  Setting to 0.05?" %
                      (self.fgcmPars.bands[bandIndex]))

            print("sigFgcm (%s) = %.4f" % (self.fgcmPars.bands[bandIndex],
                                            self.sigFgcm[bandIndex]))

            if (noPlots):
                continue

            ax.tick_params(axis='both',which='major',labelsize=14)
            ax.locator_params(axis='x',nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.5f$' % (coeff[1]) + '\n' + \
                r'$\sigma_\mathrm{tot} = %.4f$' % (coeff[2]) + '\n' + \
                r'$\sigma_\mathrm{FGCM} = %.4f$' % (self.sigFgcm[bandIndex])

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$E^{\mathrm{gray}}$',fontsize=16)

            fig.savefig('%s/%s_sigfgcm_%s.png' % (self.plotPath,
                                                  self.outfileBaseWithCycle,
                                                  self.fgcmPars.bands[bandIndex]))
            ## FIXME: add plots



        # group CCD by Exposure and Sum

        goodCCD = np.where((ccdNGoodStars >= self.minStarPerCCD) &
                           (ccdGrayErr > 0.0) &
                           (ccdGrayErr < self.maxCCDGrayErr))

        # note: goodCCD[0] refers to the expIndex, goodCCD[1] to the CCDIndex

        expGray[:] = 0.0
        expGrayRMS[:] = 0.0
        expGrayErr[:] = 0.0
        expNGoodStars[:] = 0
        expNGoodCCDs[:] = 0
        expNGoodTilings[:] = 0.0

        # temporary
        expGrayWt = np.zeros_like(expGray)

        np.add.at(expGrayWt,
                  goodCCD[0],
                  1./ccdGrayErr[goodCCD]**2.)
        np.add.at(expGray,
                  goodCCD[0],
                  ccdGray[goodCCD]/ccdGrayErr[goodCCD]**2.)
        np.add.at(expGrayRMS,
                  goodCCD[0],
                  ccdGray[goodCCD]**2./ccdGrayErr[goodCCD]**2.)
        np.add.at(expNGoodCCDs,
                  goodCCD[0],
                  1)
        np.add.at(expNGoodTilings,
                  goodCCD[0],
                  ccdNGoodTilings[goodCCD])
        np.add.at(expNGoodStars,
                  goodCCD[0],
                  ccdNGoodStars[goodCCD])

        # need at least 3 or else computation can blow up
        ## FIXME: put illegal value as filler!

        gd, = np.where(expNGoodCCDs > 2)
        expGray[gd] /= expGrayWt[gd]
        expGrayRMS[gd] = np.sqrt((expGrayRMS[gd]/expGrayWt[gd]) - (expGray[gd]**2.))
        expGrayErr[gd] = np.sqrt(1./expGrayWt[gd])
        expNGoodTilings[gd] /= expNGoodCCDs[gd]

        self.fgcmPars.compExpGray[:] = expGray
        self.fgcmPars.compVarGray[gd] = expGrayRMS[gd]**2.
        self.fgcmPars.compNGoodStarPerExp = expNGoodStars

        ##  per band we plot the expGray for photometric exposures...

        if (noPlots):
            return

        expUse,=np.where((self.fgcmPars.expFlag == 0) &
                         (expNGoodStars > self.minStarPerExp))

        for i in xrange(self.fgcmPars.nBands):
            inBand, = np.where(self.fgcmPars.expBandIndex[expUse] == i)

            if (inBand.size == 0) :
                continue

            fig=plt.figure(1)
            fig.clf()

            ax=fig.add_subplot(111)

            coeff = histoGauss(ax, expGray[expUse[inBand]])

            ax.tick_params(axis='both',which='major',labelsize=14)
            ax.locator_params(axis='x',nbins=5)

            text=r'$(%s)$' % (self.fgcmPars.bands[i]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber) + '\n' + \
                r'$\mu = %.5f$' % (coeff[1]) + '\n' + \
                r'$\sigma = %.4f$' % (coeff[2])

            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=16)
            ax.set_xlabel(r'$\mathrm{EXP}^{\mathrm{gray}}$',fontsize=16)
            ax.set_ylabel(r'# of Exposures',fontsize=14)

            fig.savefig('%s/%s_expgray_%s.png' % (self.plotPath,
                                                  self.outfileBaseWithCycle,
                                                  self.fgcmPars.bands[i]))
