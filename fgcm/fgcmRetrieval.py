from __future__ import print_function

import numpy as np
import os
import sys
import esutil
import time
import matplotlib.pyplot as plt

from fgcmUtilities import _pickle_method


import types
import copy_reg

import multiprocessing
from multiprocessing import Pool

from sharedNumpyMemManager import SharedNumpyMemManager as snmm


copy_reg.pickle(types.MethodType, _pickle_method)

class FgcmRetrieval(object):
    """
    """
    def __init__(self,fgcmConfig,fgcmPars,fgcmStars,fgcmLUT):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.log('INFO','Initializing FgcmRetrieval')

        self.fgcmPars = fgcmPars

        self.fgcmStars = fgcmStars

        ## FIXME
        #self.I0Std = fgcmLUT.I0Std
        #self.I10Std = fgcmLUT.I10Std
        self.I10StdBand = fgcmConfig.I10StdBand

        # and record configuration variables
        self.illegalValue = fgcmConfig.illegalValue
        self.nCore = fgcmConfig.nCore
        self.minStarPerCCD = fgcmConfig.minStarPerCCD
        self.ccdStartIndex = fgcmConfig.ccdStartIndex
        self.nExpPerRun = fgcmConfig.nExpPerRun

        self._prepareRetrievalArrays()

    def _prepareRetrievalArrays(self):
        """
        """

        self.r0Handle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        self.r10Handle = snmm.createArray((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')

        r0 = snmm.getArray(self.r0Handle)
        r0[:] = self.illegalValue
        r10 = snmm.getArray(self.r10Handle)
        r10[:] = self.illegalValue

    def computeRetrievalIntegrals(self,debug=False):
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run fgcmChisq before fgcmRetrieval.")

        startTime = time.time()
        self.fgcmLog.log('INFO','Computing retrieval integrals')

        # reset arrays
        r0 = snmm.getArray(self.r0Handle)
        r10 = snmm.getArray(self.r10Handle)

        r0[:] = self.illegalValue
        r10[:] = self.illegalValue


        # select good stars
        goodStars,=np.where(snmm.getArray(self.fgcmStars.objFlagHandle) == 0)

        self.fgcmLog.log('INFO','Found %d good stars for retrieval' % (goodStars.size))

        if (goodStars.size == 0):
            raise ValueError("No good stars to fit!")

        # and pre-matching stars and observations

        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        preStartTime=time.time()
        self.fgcmLog.log('INFO','Pre-matching stars and observations...')
        goodStarsSub,goodObs = esutil.numpy_util.match(goodStars,
                                                       obsObjIDIndex,
                                                       presorted=True)

        if (goodStarsSub[0] != 0.0):
            raise ValueError("Very strange that the goodStarsSub first element is non-zero.")

        # cut out all bad exposures and bad observations
        gd,=np.where((self.fgcmPars.expFlag[obsExpIndex[goodObs]] == 0) &
                     (obsFlag[goodObs] == 0))

        # crop out both goodObs and goodStarsSub
        goodObs=goodObs[gd]
        goodStarsSub=goodStarsSub[gd]

        self.goodObsHandle = snmm.createArray(goodObs.size,dtype='i4')
        snmm.getArray(self.goodObsHandle)[:] = goodObs

        self.fgcmLog.log('INFO','Pre-matching done in %.1f sec.' %
                         (time.time() - preStartTime))

        # which exposures have stars?
        uExpIndex = np.unique(obsExpIndex[goodObs])

        self.debug=debug
        if (self.debug):
            # debug mode: single core

            #self._worker(self.fgcmPars.expArray[uExpIndex])
            self._worker(uExpIndex)

        else:
            # regular multi-core
            self.fgcmLog.log('INFO','Running retrieval on %d cores' % (self.nCore))

            # split exposures into a list of arrays of roughly equal size
            #nSections = self.fgcmPars.expArray.size // self.nExpPerRun + 1
            nSections = uExpIndex.size // self.nExpPerRun + 1

            uExpIndexList = np.array_split(uExpIndex,nSections)

            # may want to sort by nObservations, but only if we pre-split

            pool = Pool(processes=self.nCore)
            pool.map(self._worker, uExpIndexList, chunksize=1)
            pool.close()
            pool.join()

        # free memory!
        snmm.freeArray(self.goodObsHandle)

        # and we're done
        self.fgcmLog.log('INFO','Computed retrieved integrals in %.2f seconds.' %
                         (time.time() - startTime))


    def _worker(self,uExpIndex):
        """
        """

        workerStarTime = time.time()

        goodObsAll = snmm.getArray(self.goodObsHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)

        _,temp = esutil.numpy_util.match(uExpIndex, obsExpIndex[goodObsAll])
        goodObs = goodObsAll[temp]

        obsExpIndexGO = obsExpIndex[goodObs]

        temp = None

        # arrays we need...
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanNoChrom = snmm.getArray(self.fgcmStars.objMagStdMeanNoChromHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)

        obsObjIDIndexGO = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)[goodObs]
        obsBandIndexGO = snmm.getArray(self.fgcmStars.obsBandIndexHandle)[goodObs]
        #obsLUTFilterIndexGO = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)[goodObs]
        obsCCDIndexGO = snmm.getArray(self.fgcmStars.obsCCDHandle)[goodObs] - self.ccdStartIndex
        obsMagADUGO = snmm.getArray(self.fgcmStars.obsMagADUHandle)[goodObs]
        obsMagErrGO = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)[goodObs]

        # compute delta mags
        # deltaMag = m_b^inst(i,j)  - <m_b^std>(j) + QE_sys
        #   we want to take out the gray term from the qe

        #deltaMagGO = (obsMagADUGO -
        #              objMagStdMeanNoChrom[obsObjIDIndexGO,
        #                                   obsBandIndexGO] +
        #              self.fgcmPars.expQESys[obsExpIndexGO])
        deltaMagGO = (obsMagADUGO -
                      objMagStdMean[obsObjIDIndexGO,
                                    obsBandIndexGO] +
                      self.fgcmPars.expQESys[obsExpIndexGO])

        deltaMagErr2GO = (obsMagErrGO**2. +
                          objMagStdMeanErr[obsObjIDIndexGO,
                                           obsBandIndexGO]**2.)

        # and to flux space
        fObsGO = 10.**(-0.4*deltaMagGO)
        fObsErr2GO = deltaMagErr2GO * ((2.5/np.log(10.)) * fObsGO)**2.
        deltaStdGO = (1.0 + objSEDSlope[obsObjIDIndexGO,
                                        obsBandIndexGO] *
                      self.I10StdBand[obsBandIndexGO])

        deltaStdWeightGO = 1./(fObsErr2GO * deltaStdGO * deltaStdGO)

        # and compress obsExpIndexGO
        theseObsExpIndexGO=np.searchsorted(uExpIndex, obsExpIndexGO)

        r0 = snmm.getArray(self.r0Handle)
        r10 = snmm.getArray(self.r10Handle)

        # set up the matrices

        # IMatrix[0,0] = sum (1./sigma_f^2)
        # IMatrix[0,1] = sum (F'_nu/sigma_f^2)
        # IMatrix[1,0] = IMatrix[0,1]
        # IMatrix[1,1] = sum (F'_nu^2/sigma_f^2)

        # RHS[0] = sum (f^obs / (sigma_f^2 * deltaStd))
        # RHS[1] = sum ((F'_nu * f^obs / (sigma_f^2 * deltaStd))


        IMatrix = np.zeros((2,2,uExpIndex.size,self.fgcmPars.nCCD),dtype='f8')
        RHS = np.zeros((2,uExpIndex.size,self.fgcmPars.nCCD),dtype='f8')
        nStar = np.zeros((uExpIndex.size,self.fgcmPars.nCCD),dtype='i4')

        np.add.at(IMatrix,
                  (0,0,theseObsExpIndexGO,obsCCDIndexGO),
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (0,1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (1,0,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  deltaStdWeightGO)
        np.add.at(IMatrix,
                  (1,1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO]**2. *
                  deltaStdWeightGO)
        np.add.at(nStar,
                  (theseObsExpIndexGO,obsCCDIndexGO),
                  1)

        np.add.at(RHS,
                  (0,theseObsExpIndexGO,obsCCDIndexGO),
                  fObsGO / (fObsErr2GO * deltaStdGO))
        np.add.at(RHS,
                  (1,theseObsExpIndexGO,obsCCDIndexGO),
                  objSEDSlope[obsObjIDIndexGO,
                              obsBandIndexGO] *
                  fObsGO / (fObsErr2GO * deltaStdGO))

        # which can be computed?
        expIndexUse, ccdIndexUse = np.where(nStar >= self.minStarPerCCD)

        # loop, doing the linear algebra
        for i in xrange(expIndexUse.size):
            try:
                IRetrieved = np.dot(np.linalg.inv(IMatrix[:,:,expIndexUse[i],
                                                              ccdIndexUse[i]]),
                                    RHS[:,expIndexUse[i],ccdIndexUse[i]])
            except:
                continue

            # record these in the shared array ... should not step
            #  on each others' toes
            r0[uExpIndex[expIndexUse[i]],ccdIndexUse[i]] = IRetrieved[0]
            r10[uExpIndex[expIndexUse[i]],ccdIndexUse[i]] = IRetrieved[1]/IRetrieved[0]




    def computeRetrievedIntegrals(self,debug=False):
        ### THIS IS NOT USED I THINK
        """
        """

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run fgcmChisq before fgcmRetrieval.")

        startTime = time.time()
        self.fgcmLog.log('INFO','Computing retrieved integrals')

        # reset arrays
        r0 = snmm.getArray(self.r0Handle)
        r10 = snmm.getArray(self.r10Handle)

        r0[:] = self.illegalValue
        r10[:] = self.illegalValue

        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)

        obsIndex = snmm.getArray(self.fgcmStars.obsIndexHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)
        #obsLUTFilterIndex = snmm.getArray(self.fgcmStars.obsLUTFilterIndexHandle)
        obsCCDIndex = snmm.getArray(self.fgcmStars.obsCCDHandle) - self.ccdStartIndex
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        objNobs = snmm.getArray(self.fgcmStars.objNobsHandle)
        objSEDSlope = snmm.getArray(self.fgcmStars.objSEDSlopeHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)

        obsMagADU = snmm.getArray(self.fgcmStars.obsMagADUHandle)
        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUErrHandle)


        # limit to good observations of calibration stars
        minObs = objNGoodObs[:,self.fgcmStars.bandRequiredIndex].min(axis=1)
        goodStars, = np.where((minObs >= self.fgcmStars.minPerBand) &
                              (objFlag == 0))

        #_,b=esutil.numpy_util.match(objID[goodStars],objID[obsObjIDIndex])
        # NOTE: this relies on np.where returning a sorted array.
        _,goodObs=esutil.numpy_util.match(goodStars,obsObjIDIndex,presorted=True)

        gd,=np.where(obsFlag[goodObs] == 0)
        goodObs=goodObs[gd]

        self.fgcmLog.log('INFO','FgcmRetrieval using %d observations from %d good stars.' %
                         (goodObs.size,goodStars.size))

        IMatrix = np.zeros((2,2,self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        RHS = np.zeros((2,self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='f8')
        nStar = np.zeros((self.fgcmPars.nExp,self.fgcmPars.nCCD),dtype='i4')

        # deltaMag = m_b^inst(i,j)  - <m_b^std>(j) + QE_sys
        #   we want to take out the gray term from the qe

        # compute deltaMag and error
        deltaMag = np.zeros(self.fgcmStars.nStarObs,dtype='f8')
        deltaMagErr2 = np.zeros_like(deltaMag)
        deltaMag[obsIndex[goodObs]] = (obsMagADU[obsIndex[goodObs]] -
                                 objMagStdMean[obsObjIDIndex[obsIndex[goodObs]],
                                               obsBandIndex[obsIndex[goodObs]]] +
                                 self.fgcmPars.expQESys[obsExpIndex[goodObs]])
        deltaMagErr2[obsIndex[goodObs]] = (obsMagErr[obsIndex[goodObs]]**2. +
                                     objMagStdMeanErr[obsObjIDIndex[obsIndex[goodObs]],
                                                      obsBandIndex[obsIndex[goodObs]]]**2.)

        # and put in flux units
        # these can work on the full array, even if they're bad.
        fObs = 10.**(-0.4*deltaMag)
        fObsErr2 = deltaMagErr2 * ((2.5/np.log(10.)) * fObs)**2.
        deltaStd = (1.0 + objSEDSlope[obsObjIDIndex[obsIndex],
                                      obsBandIndex[obsIndex]] *
                    self.I10StdBand[obsBandIndex[obsIndex]])
        deltaStdWeight = fObsErr2 * deltaStd * deltaStd

        # do all the exp/ccd sums

        # IMatrix[0,0] = sum (1./sigma_f^2)
        # IMatrix[0,1] = sum (F'_nu/sigma_f^2)
        # IMatrix[1,0] = IMatrix[0,1]
        # IMatrix[1,1] = sum (F'_nu^2/sigma_f^2)

        # RHS[0] = sum (f^obs / (sigma_f^2 * deltaStd))
        # RHS[1] = sum ((F'_nu * f^obs / (sigma_f^2 * deltaStd))

        np.add.at(IMatrix,
                  (0,0,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  1./deltaStdWeight[obsIndex[goodObs]])
        np.add.at(IMatrix,
                  (0,1,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  objSEDSlope[obsObjIDIndex[obsIndex[goodObs]],
                              obsBandIndex[obsIndex[goodObs]]] /
                  deltaStdWeight[obsIndex[goodObs]])
        np.add.at(IMatrix,
                  (1,0,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  objSEDSlope[obsObjIDIndex[obsIndex[goodObs]],
                              obsBandIndex[obsIndex[goodObs]]] /
                  deltaStdWeight[obsIndex[goodObs]])
        np.add.at(IMatrix,
                  (1,1,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  objSEDSlope[obsObjIDIndex[obsIndex[goodObs]],
                              obsBandIndex[obsIndex[goodObs]]]**2. /
                  deltaStdWeight[obsIndex[goodObs]])
        np.add.at(nStar,
                  (obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  1)

        np.add.at(RHS,
                  (0,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  fObs[obsIndex[goodObs]] / (fObsErr2[obsIndex[goodObs]] *
                                             deltaStd[obsIndex[goodObs]]))
        np.add.at(RHS,
                  (1,obsExpIndex[obsIndex[goodObs]],obsCCDIndex[obsIndex[goodObs]]),
                  objSEDSlope[obsObjIDIndex[obsIndex[goodObs]],
                              obsBandIndex[obsIndex[goodObs]]] *
                  fObs[obsIndex[goodObs]] /
                  (fObsErr2[obsIndex[goodObs]] *
                   deltaStd[obsIndex[goodObs]]))

        # which ones can we compute?
        expIndexUse,ccdIndexUse=np.where(nStar >= self.minStarPerCCD)

        self.fgcmLog.log('INFO','Found %d CCDs to compute retrieved integrals.' %
                         (ccdIndexUse.size))

        # and loop, to do the linear algebra solution
        for i in xrange(expIndexUse.size):
            try:
                IRetrieved = np.dot(np.linalg.inv(IMatrix[:,:,expIndexUse[i],
                                                              ccdIndexUse[i]]),
                                    RHS[:,expIndexUse[i],ccdIndexUse[i]])
            except:
                ## FIXME: write exposure number, ccd number not index
                self.fgcmLog.log('DEBUG','Failed to compute R0, R1 for %d/%d'
                                 % (expIndexUse[i],ccdIndexUse[i]))
                continue

            r0[expIndexUse[i],ccdIndexUse[i]] = IRetrieved[0]
            r10[expIndexUse[i],ccdIndexUse[i]] = IRetrieved[1]/IRetrieved[0]


        # and we're done
        self.fgcmLog.log('INFO','Computed retrieved integrals in %.2f seconds.' %
                         (time.time() - startTime))


