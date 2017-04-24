from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import time

from fgcmUtilities import _pickle_method
from fgcmUtilities import resourceUsage

import types
import copy_reg
import multiprocessing
from multiprocessing import Pool

from sharedNumpyMemManager import SharedNumpyMemManager as snmm


#from fgcmLUT import FgcmLUTSHM

copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmChisqOld(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmChisq')

        # does this need to be shm'd?
        self.fgcmPars = fgcmPars

        # this is shm'd
        self.fgcmLUT = fgcmLUT

        # also shm'd
        self.fgcmStars = fgcmStars

        # need to configure
        self.nCore = fgcmConfig.nCore
        self.ccdStartIndex = fgcmConfig.ccdStartIndex

        #self.fitChisqs = []
        self.resetFitChisqList()

        # not sure what we need the config for

        #resourceUsage('End of chisq init')

    def resetFitChisqList(self):
        self.fitChisqs = []

    def __call__(self,fitParams,fitterUnits=False,computeDerivatives=False,computeSEDSlopes=False,debug=False,allExposures=False):
        """
        """

        # computeDerivatives: do we want to compute the derivatives?
        # computeSEDSlope: compute SED Slope and recompute mean mags?
        # fitterUnits: units of the fitter or "true" units?

        self.computeDerivatives = computeDerivatives
        self.computeSEDSlopes = computeSEDSlopes
        self.fitterUnits = fitterUnits
        self.allExposures = allExposures

        self.fgcmLog.log('DEBUG','FgcmChisq: computeDerivatives = %d' %
                         (int(computeDerivatives)))
        self.fgcmLog.log('DEBUG','FgcmChisq: computeSEDSlopes = %d' %
                         (int(computeSEDSlopes)))
        self.fgcmLog.log('DEBUG','FgcmChisq: fitterUnits = %d' %
                         (int(fitterUnits)))
        self.fgcmLog.log('DEBUG','FgcmChisq: allExposures = %d' %
                         (int(allExposures)))


        if (self.allExposures and (self.computeDerivatives or
                                   self.computeSEDSlopes)):
            raise ValueError("Cannot set allExposures and computeDerivatives or computeSEDSlopes")

        # for things that need to be changed, we need to create an array *here*
        # I think.  And copy it back out.  Sigh.

        #resourceUsage('Start of call')

        # this is the function that will be called by the fitter, I believe.

        # unpack the parameters and convert units if necessary. These are not
        # currently shared memory, since they should be small enough to not be
        # a problem.  But we can revisit.

        self.fgcmPars.reloadParArray(fitParams,fitterUnits=self.fitterUnits)
        self.fgcmPars.parsToExposures()


        # and reset numbers if necessary
        if (not self.allExposures):
            snmm.getArray(self.fgcmStars.objMagStdMeanHandle)[:] = 99.0
            snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)[:] = 99.0

        # and select good stars!  These are the ones to map.
        #  note that these are the only stars we will care about.
        goodStars,=np.where(snmm.getArray(self.fgcmStars.objFlagHandle) == 0)

        self.fgcmLog.log('INFO','Found %d good stars for chisq' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # testing
        #goodStars=goodStars[0:10000]

        # prepare the return arrays...
        # how many do we have?

        self.nSums = 2   # chisq, nobs
        if (self.computeDerivatives):
            self.nSums += self.fgcmPars.nFitPars  # one for each parameter

        startTime = time.time()

        self.debug=debug
        if (self.debug):
            self.totalHandleDict = {}
            self.totalHandleDict[0] = snmm.createArray(self.nSums,dtype='f4')

            for goodStar in goodStars:
                self._worker(goodStar)

            partialSums = snmm.getArray(self.totalHandleDict[0])[:]
        else:

            self.fgcmLog.log('INFO','Running chisq on %d cores' % (self.nCore))
            # make a dummy process to discover starting child number
            proc = multiprocessing.Process()
            workerIndex = proc._identity[0]+1
            proc = None

            self.totalHandleDict = {}
            for thisCore in xrange(self.nCore):
                self.totalHandleDict[workerIndex + thisCore] = snmm.createArray(self.nSums,dtype='f4')

            # will want to make a pool

            pool = Pool(processes=self.nCore)
            #resourceUsage('premap')
            pool.map(self._worker,goodStars)
            pool.close()
            pool.join()

            # and return the derivatives + chisq
            partialSums = np.zeros(self.nSums,dtype='f8')
            for thisCore in xrange(self.nCore):
                partialSums[:] += snmm.getArray(self.totalHandleDict[workerIndex + thisCore])[:]

        # only in regular mode do we actually compute chisq
        if (not self.allExposures):
            # FIXME: dof should be actual number of fit parameters, shouldn't it.

            fitDOF = partialSums[-1] - float(self.fgcmPars.nFitPars)
            if (fitDOF <= 0):
                raise ValueError("Number of parameters fitted is more than number of constraints! (%d > %d)" % (self.fgcmPars.nFitPars,partialSums[-1]))

            fitChisq = partialSums[-2] / fitDOF
            if (self.computeDerivatives):
                dChisqdP = partialSums[0:self.fgcmPars.nFitPars] / fitDOF

            # want to append this...
            self.fitChisqs.append(fitChisq)

            self.fgcmLog.log('INFO','Chisq/dof = %.2f' % (fitChisq))
        else:
            # just set it to the last value for reference
            fitChisq = self.fitChisqs[-1]

        # free shared arrays
        for key in self.totalHandleDict.keys():
            snmm.freeArray(self.totalHandleDict[key])

        self.fgcmLog.log('INFO','Chisq computation took %.2f seconds.' %
                         (time.time() - startTime))

        #resourceUsage('end')

        #print(fitChisq)

        # and flag that we've computed magStd
        self.fgcmStars.magStdComputed = True
        if (self.allExposures):
            self.fgcmStars.allMagStdComputed = True

        if (self.computeDerivatives):
            return fitChisq, dChisqdP
        else:
            return fitChisq

    def _worker(self,objIndex):
        """
        """

        # make local pointers to useful arrays...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        objNobs = snmm.getArray(self.fgcmStars.objNobsHandle)


        thisObsIndex = obsIndex[objObsIndex[objIndex]:objObsIndex[objIndex]+objNobs[objIndex]]
        thisObsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)[thisObsIndex]

        # cut to good exposures, if we desire it...
        ## MAYBE: Check if this can be done more efficiently.
        if (not self.allExposures) :
            gd,=np.where(self.fgcmPars.expFlag[thisObsExpIndex] == 0)
            thisObsIndex=thisObsIndex[gd]
            thisObsExpIndex = thisObsExpIndex[gd]

        thisObsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)[thisObsIndex]
        thisObsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle)[thisObsIndex] - self.ccdStartIndex

        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagADUErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)

        # need to know which are fit bands!
        _,thisObsFitUse = esutil.numpy_util.match(self.fgcmPars.fitBandIndex,thisObsBandIndex)

        thisObjGoodBand, = np.where(objNGoodObs[objIndex,:] >= 1)

        thisSecZenith = snmm.getArray(self.fgcmStars.obsSecZenithHandle)[thisObsIndex]


        # get I0obs values...
        lutIndices = self.fgcmLUT.getIndices(thisObsBandIndex,
                                             self.fgcmPars.expPWV[thisObsExpIndex],
                                             self.fgcmPars.expO3[thisObsExpIndex],
                                             np.log(self.fgcmPars.expTau[thisObsExpIndex]),
                                             self.fgcmPars.expAlpha[thisObsExpIndex],
                                             thisSecZenith,
                                             thisObsCCDIndex,
                                             self.fgcmPars.expPmb[thisObsExpIndex])

        # and I10obs values...
        thisI0 = self.fgcmLUT.computeI0(thisObsBandIndex,
                                        self.fgcmPars.expPWV[thisObsExpIndex],
                                        self.fgcmPars.expO3[thisObsExpIndex],
                                        np.log(self.fgcmPars.expTau[thisObsExpIndex]),
                                        self.fgcmPars.expAlpha[thisObsExpIndex],
                                        thisSecZenith,
                                        thisObsCCDIndex,
                                        self.fgcmPars.expPmb[thisObsExpIndex],
                                        lutIndices)
        thisI10 = self.fgcmLUT.computeI1(lutIndices) / thisI0

        thisQESys = self.fgcmPars.expQESys[thisObsExpIndex]

        # compute thisMagObs
        thisMagObs = obsMagADU[thisObsIndex] + 2.5*np.log10(thisI0) + thisQESys

        thisMagErr2 = obsMagADUErr[thisObsIndex]**2.


        if (self.computeSEDSlopes):
            # use magObs to compute mean mags...
            # compute in all bands here.

            wtSum = np.zeros(self.fgcmStars.nBands)
            np.add.at(wtSum,
                      thisObsBandIndex,
                      1./thisMagErr2)
            objMagStdMean[objIndex,thisObjGoodBand] = 0.0
            np.add.at(objMagStdMean[objIndex,:],
                      thisObsBandIndex,
                      thisMagObs/thisMagErr2)
                      #obsMagStd[thisObsIndex]/thisMagErr2)
            objMagStdMean[objIndex,thisObjGoodBand] /= wtSum[thisObjGoodBand]
            objMagStdMeanErr[objIndex,thisObjGoodBand] = np.sqrt(1./wtSum[thisObjGoodBand])

            self.fgcmStars.computeObjectSEDSlope(objIndex)

        # compute magStd (and record)
        thisDeltaStd = 2.5 * np.log10((1.0 +
                                       objSEDSlope[objIndex,thisObsBandIndex] *
                                       thisI10) /
                                      (1.0 + objSEDSlope[objIndex,thisObsBandIndex] *
                                       self.fgcmLUT.I10Std[thisObsBandIndex]))

        obsMagStd[thisObsIndex] = thisMagObs + thisDeltaStd

        if (self.allExposures) :
            # kick out, don't compute (corrupted) means
            return None

        # compute mean objMagStdMean

        # new faster version...

        wtSum = np.zeros(self.fgcmStars.nBands)
        np.add.at(wtSum,
                  thisObsBandIndex,
                  1./thisMagErr2)
        objMagStdMean[objIndex,thisObjGoodBand] = 0.0
        np.add.at(objMagStdMean[objIndex,:],
                  thisObsBandIndex,
                  obsMagStd[thisObsIndex]/thisMagErr2)
        objMagStdMean[objIndex,thisObjGoodBand] /= wtSum[thisObjGoodBand]
        objMagStdMeanErr[objIndex,thisObjGoodBand] = np.sqrt(1./wtSum[thisObjGoodBand])

        # compute deltaMag
        deltaMag = (obsMagStd[thisObsIndex] -
                    objMagStdMean[objIndex,thisObsBandIndex])
        deltaMagErr2 = (thisMagErr2 +
                        objMagStdMeanErr[objIndex,thisObsBandIndex]**2.)
        deltaMagWeighted = deltaMag/deltaMagErr2

        # finally, compute the chisq.  Also need a return array!
        partialChisq = np.sum(deltaMag[thisObsFitUse]**2./deltaMagErr2[thisObsFitUse])

        partialArray = np.zeros(self.nSums,dtype='f4')

        # last one is the chisq
        partialArray[-2] = partialChisq
        partialArray[-1] = thisObsIndex[thisObsFitUse].size

        # and compute the derivatives if desired...
        if (self.computeDerivatives):

            unitDict=self.fgcmPars.getUnitDict(fitterUnits=self.fitterUnits)
            # do I need to loop over all parameters?

            # i,i',i": loop over observations (in a given band)
            # j: loop over objects

            # first, we need dL(i,j|p) = d/dp(2.5*log10(LUT(i,j|p)))
            #                          = 1.086*(LUT'(i,j|p)/LUT(i,j|p))
            (dLdPWV,dLdO3,dLdTau,dLdAlpha) = (
                self.fgcmLUT.computeLogDerivatives(lutIndices, thisI0, self.fgcmPars.expTau[thisObsExpIndex]))

            # we have objMagStdMeanErr[objIndex,:] = \Sum_{i"} 1/\sigma^2_{i"j}
            #   note that this is summed over all observations of an object in a band
            #   so that this is already done

            # we need magdLdp = \Sum_{i'} (1/\sigma^2_{i'j}) dL(i',j|p)
            #   note that this is summed over all observations in a filter that
            #   touch a given parameter

            # set up arrays
            magdLdPWVIntercept = np.zeros((self.fgcmPars.nCampaignNights,
                                           self.fgcmPars.nFitBands))
            magdLdPWVSlope = np.zeros_like(magdLdPWVIntercept)
            magdLdPWVOffset = np.zeros_like(magdLdPWVIntercept)
            magdLdTauIntercept = np.zeros_like(magdLdPWVIntercept)
            magdLdTauSlope = np.zeros_like(magdLdPWVIntercept)
            magdLdTauOffset = np.zeros_like(magdLdPWVIntercept)
            magdLdAlpha = np.zeros_like(magdLdPWVIntercept)
            magdLdO3 = np.zeros_like(magdLdPWVIntercept)

            magdLdPWVScale = np.zeros(self.fgcmPars.nFitBands,dtype='f4')
            magdLdTauScale = np.zeros_like(magdLdPWVScale)

            magdLdWashIntercept = np.zeros((self.fgcmPars.nWashIntervals,
                                            self.fgcmPars.nFitBands))
            magdLdWashSlope = np.zeros_like(magdLdWashIntercept)

            # precompute object err2...
            thisObjMagStdMeanErr2 = objMagStdMeanErr[objIndex,:]**2.

            ##########
            ## O3
            ##########

            tempNightIndex = self.fgcmPars.expNightIndex[thisObsExpIndex[thisObsFitUse]]
            uNightIndex = np.unique(tempNightIndex)

            np.add.at(magdLdO3,
                      (tempNightIndex,thisObsBandIndex[thisObsFitUse]),
                      dLdO3[thisObsFitUse] / thisMagErr2[thisObsFitUse])
            np.multiply.at(magdLdO3,
                           (tempNightIndex,thisObsBandIndex[thisObsFitUse]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse]])

            np.add.at(partialArray[self.fgcmPars.parO3Loc:
                                       (self.fgcmPars.parO3Loc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex,
                      deltaMagWeighted[thisObsFitUse] * (
                    (dLdO3[thisObsFitUse] -
                     magdLdO3[tempNightIndex,thisObsBandIndex[thisObsFitUse]])))

            partialArray[self.fgcmPars.parO3Loc +
                         uNightIndex] *= (2.0 / unitDict['o3Unit'])

            ###########
            ## Alpha
            ###########

            np.add.at(magdLdAlpha,
                      (tempNightIndex,thisObsBandIndex[thisObsFitUse]),
                      dLdAlpha[thisObsFitUse] / thisMagErr2[thisObsFitUse])
            np.multiply.at(magdLdAlpha,
                           (tempNightIndex,thisObsBandIndex[thisObsFitUse]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse]])

            np.add.at(partialArray[self.fgcmPars.parAlphaLoc:
                                       (self.fgcmPars.parAlphaLoc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex,
                      deltaMagWeighted[thisObsFitUse] * (
                    (dLdAlpha[thisObsFitUse] -
                     magdLdAlpha[tempNightIndex,thisObsBandIndex[thisObsFitUse]])))

            partialArray[self.fgcmPars.parAlphaLoc +
                         uNightIndex] *= (2.0 / unitDict['alphaUnit'])

            ###########
            ## PWV External
            ###########

            if (self.fgcmPars.hasExternalPWV):
                hasExt,=np.where(self.fgcmPars.externalPWVFlag[thisObsExpIndex[thisObsFitUse]])

                # PWV Nightly Offset
                np.add.at(magdLdPWVOffset,
                          (tempNightIndex[hasExt],thisObsBandIndex[thisObsFitUse[hasExt]]),
                          dLdPWV[thisObsFitUse[hasExt]] / thisMagErr2[thisObsFitUse[hasExt]])
                np.multiply.at(magdLdPWVOffset,
                               (tempNightIndex[hasExt],thisObsBandIndex[thisObsFitUse[hasExt]]),
                               thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[hasExt]]])

                np.add.at(partialArray[self.fgcmPars.parExternalPWVOffsetLoc:
                                           (self.fgcmPars.parExternalPWVOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          tempNightIndex[hasExt],
                          deltaMagWeighted[thisObsFitUse[hasExt]] * (
                        (dLdPWV[thisObsFitUse[hasExt]] -
                         magdLdPWVOffset[tempNightIndex[hasExt],
                                         thisObsBandIndex[thisObsFitUse[hasExt]]])))
                partialArray[self.fgcmPars.parExternalPWVOffsetLoc +
                             np.unique(tempNightIndex[hasExt])] *= (2.0 / unitDict['pwvUnit'])

                # PWV Global Scale
                np.add.at(magdLdPWVScale,
                          thisObsBandIndex[thisObsFitUse[hasExt]],
                          self.fgcmPars.expPWV[thisObsExpIndex[thisObsFitUse[hasExt]]] *
                          dLdPWV[thisObsFitUse[hasExt]] / thisMagErr2[thisObsFitUse[hasExt]])
                np.multiply.at(magdLdPWVScale,
                               thisObsBandIndex[thisObsFitUse[hasExt]],
                               thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[hasExt]]])
                partialArray[self.fgcmPars.parExternalPWVScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeighted[thisObsFitUse[hasExt]] * (
                            self.fgcmPars.expPWV[thisObsExpIndex[thisObsFitUse[hasExt]]] *
                            dLdPWV[thisObsFitUse[hasExt]] -
                            magdLdPWVScale[thisObsBandIndex[thisObsFitUse[hasExt]]])) /
                    unitDict['pwvUnit'])

            ###########
            ## PWV No External
            ###########

            noExt,=np.where(~self.fgcmPars.externalPWVFlag[thisObsExpIndex[thisObsFitUse]])

            # PWV Nightly Intercept

            np.add.at(magdLdPWVIntercept,
                      (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                      dLdPWV[thisObsFitUse[noExt]] / thisMagErr2[thisObsFitUse[noExt]])
            np.multiply.at(magdLdPWVIntercept,
                           (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[noExt]]])

            np.add.at(partialArray[self.fgcmPars.parPWVInterceptLoc:
                                       (self.fgcmPars.parPWVInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex[noExt],
                      deltaMagWeighted[thisObsFitUse[noExt]] * (
                    (dLdPWV[thisObsFitUse[noExt]] -
                     magdLdPWVIntercept[tempNightIndex[noExt],
                                        thisObsBandIndex[thisObsFitUse[noExt]]])))
            partialArray[self.fgcmPars.parPWVInterceptLoc +
                         uNightIndex] *= (2.0 / unitDict['pwvUnit'])

            # PWV Nightly Slope

            np.add.at(magdLdPWVSlope,
                      (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                      self.fgcmPars.expDeltaUT[thisObsExpIndex[thisObsFitUse[noExt]]] *
                      dLdPWV[thisObsFitUse[noExt]] / thisMagErr2[thisObsFitUse[noExt]])
            np.multiply.at(magdLdPWVSlope,
                           (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[noExt]]])

            np.add.at(partialArray[self.fgcmPars.parPWVSlopeLoc:
                                       (self.fgcmPars.parPWVSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex[noExt],
                      deltaMagWeighted[thisObsFitUse[noExt]] * (
                    (self.fgcmPars.expDeltaUT[thisObsExpIndex[thisObsFitUse[noExt]]]*
                     dLdPWV[thisObsFitUse[noExt]] -
                     magdLdPWVSlope[tempNightIndex[noExt],
                                    thisObsBandIndex[thisObsFitUse[noExt]]])))

            partialArray[self.fgcmPars.parPWVSlopeLoc +
                         uNightIndex] *= (2.0 / unitDict['pwvSlopeUnit'])

            #############
            ## Tau External
            #############

            if (self.fgcmPars.hasExternalTau):
                hasExt,=np.where(self.fgcmPars.externalTauFlag[thisObsExpIndex[thisObsFitUse]])

                # Tau Nightly Offset
                np.add.at(magdLdTauOffset,
                          (tempNightIndex[hasExt],thisObsBandIndex[thisObsFitUse[hasExt]]),
                          dLdTau[thisObsFitUse[hasExt]] / thisMagErr2[thisObsFitUse[hasExt]])
                np.multiply.at(magdLdTauOffset,
                               (tempNightIndex[hasExt],
                                thisObsBandIndex[thisObsFitUse[hasExt]]),
                               thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[hasExt]]])

                np.add.at(partialArray[self.fgcmPars.parExternalTauOffsetLoc:
                                           (self.fgcmPars.parExternalTauOffsetLoc+
                                            self.fgcmPars.nCampaignNights)],
                          tempNightIndex[hasExt],
                          deltaMagWeighted[thisObsFitUse[hasExt]] * (
                        (dLdTau[thisObsFitUse[hasExt]] -
                         magdLdTauOffset[tempNightIndex[hasExt],
                                         thisObsBandIndex[thisObsFitUse[hasExt]]])))

                partialArray[self.fgcmPars.parExternalTauOffsetLoc +
                             uNightIndex] *= (2.0 / unitDict['tauUnit'])

                # Tau Global Scale
                np.add.at(magdLdTauScale,
                          thisObsBandIndex[thisObsFitUse[hasExt]],
                          self.fgcmPars.expTau[thisObsExpIndex[thisObsFitUse[hasExt]]] *
                          dLdTau[thisObsFitUse[hasExt]] / thisMagErr2[thisObsFitUse[hasExt]])
                np.multiply.at(magdLdTauScale,
                               thisObsBandIndex[thisObsFitUse[hasExt]],
                               thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[hasExt]]])

                partialArray[self.fgcmPars.parExternalTauScaleLoc] = 2.0 * (
                    np.sum(deltaMagWeighted[thisObsFitUse[hasExt]] * (
                            self.fgcmPars.expTau[thisObsExpIndex[thisObsFitUse[hasExt]]] *
                            dLdTau[thisObsFitUse[hasExt]] -
                            magdLdTauScale[thisObsBandIndex[thisObsFitUse[hasExt]]])) /
                    unitDict['tauUnit'])

            ###########
            ## Tau No External
            ###########

            noExt,=np.where(~self.fgcmPars.externalTauFlag[thisObsExpIndex[thisObsFitUse]])

            # Tau Nightly Intercept

            np.add.at(magdLdTauIntercept,
                      (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                      dLdTau[thisObsFitUse[noExt]] / thisMagErr2[thisObsFitUse[noExt]])
            np.multiply.at(magdLdTauIntercept,
                           (tempNightIndex[noExt],
                            thisObsBandIndex[thisObsFitUse[noExt]]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[noExt]]])

            np.add.at(partialArray[self.fgcmPars.parTauInterceptLoc:
                                       (self.fgcmPars.parTauInterceptLoc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex[noExt],
                      deltaMagWeighted[thisObsFitUse[noExt]] * (
                    (dLdTau[thisObsFitUse[noExt]] -
                     magdLdTauIntercept[tempNightIndex[noExt],
                                        thisObsBandIndex[thisObsFitUse[noExt]]])))
            partialArray[self.fgcmPars.parTauInterceptLoc +
                         uNightIndex] *= (2.0 / unitDict['tauUnit'])

            # Tau Nightly Slope

            np.add.at(magdLdTauSlope,
                      (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                      self.fgcmPars.expDeltaUT[thisObsExpIndex[thisObsFitUse[noExt]]] *
                      dLdTau[thisObsFitUse[noExt]] / thisMagErr2[thisObsFitUse[noExt]])
            np.multiply.at(magdLdTauSlope,
                           (tempNightIndex[noExt],thisObsBandIndex[thisObsFitUse[noExt]]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse[noExt]]])

            np.add.at(partialArray[self.fgcmPars.parTauSlopeLoc:
                                       (self.fgcmPars.parTauSlopeLoc+
                                        self.fgcmPars.nCampaignNights)],
                      tempNightIndex[noExt],
                      deltaMagWeighted[thisObsFitUse[noExt]] * (
                    (self.fgcmPars.expDeltaUT[thisObsExpIndex[thisObsFitUse[noExt]]]*
                     dLdTau[thisObsFitUse[noExt]] -
                     magdLdTauSlope[tempNightIndex[noExt],
                                    thisObsBandIndex[thisObsFitUse[noExt]]])))

            partialArray[self.fgcmPars.parTauSlopeLoc +
                         uNightIndex] *= (2.0 / unitDict['tauSlopeUnit'])

            #############
            ## Washes (QE Sys)
            #############

            tempWashIndex = self.fgcmPars.expWashIndex[thisObsExpIndex[thisObsFitUse]]
            uWashIndex = np.unique(tempWashIndex)

            # Wash Intercept

            np.add.at(magdLdWashIntercept,
                      (tempWashIndex,thisObsBandIndex[thisObsFitUse]),
                      1./thisMagErr2[thisObsFitUse])
            np.multiply.at(magdLdWashIntercept,
                           (tempWashIndex,thisObsBandIndex[thisObsFitUse]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse]])

            np.add.at(partialArray[self.fgcmPars.parQESysInterceptLoc:
                                       (self.fgcmPars.parQESysInterceptLoc +
                                        self.fgcmPars.nWashIntervals)],
                      tempWashIndex,
                      deltaMagWeighted[thisObsFitUse] * (
                    (1.0 - magdLdWashIntercept[tempWashIndex,
                                               thisObsBandIndex[thisObsFitUse]])))

            partialArray[self.fgcmPars.parQESysInterceptLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysUnit'])

            # Wash Slope

            np.add.at(magdLdWashSlope,
                      (tempWashIndex,thisObsBandIndex[thisObsFitUse]),
                      (self.fgcmPars.expMJD[thisObsExpIndex[thisObsFitUse]] -
                       self.fgcmPars.washMJDs[self.fgcmPars.expWashIndex[thisObsExpIndex[thisObsFitUse]]])/
                      thisMagErr2[thisObsFitUse])
            np.multiply.at(magdLdWashSlope,
                           (tempWashIndex,thisObsBandIndex[thisObsFitUse]),
                           thisObjMagStdMeanErr2[thisObsBandIndex[thisObsFitUse]])

            np.add.at(partialArray[self.fgcmPars.parQESysSlopeLoc:
                                       (self.fgcmPars.parQESysSlopeLoc +
                                        self.fgcmPars.nWashIntervals)],
                      tempWashIndex,
                      deltaMagWeighted[thisObsFitUse] * (
                    (self.fgcmPars.expMJD[thisObsExpIndex[thisObsFitUse]] -
                     self.fgcmPars.washMJDs[tempWashIndex]) -
                    magdLdWashSlope[tempWashIndex,thisObsBandIndex[thisObsFitUse]]))
            partialArray[self.fgcmPars.parQESysSlopeLoc +
                         uWashIndex] *= (2.0 / unitDict['qeSysSlopeUnit'])

        # note that this doesn't need locking because we are only accessing
        #   a single array from a single process
        if self.debug:
            thisCore = 0
        else:
            thisCore = multiprocessing.current_process()._identity[0]
        totalArr = snmm.getArray(self.totalHandleDict[thisCore])
        totalArr[:] = totalArr[:] + partialArray

        # Nothing to return
        return None