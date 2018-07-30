from __future__ import division, absolute_import, print_function
from past.builtins import xrange

import numpy as np
import os
import sys
import esutil
import time
import healpy as hp

import matplotlib.pyplot as plt

from .fgcmUtilities import objFlagDict

from .sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmCorrelations(object):
    """
    Class to compute residual correlation functions.

    parameters
    ----------
    fgcmConfig: FgcmConfig
    fgcmPars: FgcmParameters
    fgcmStars: FgcmStars

    """

    def __init__(self,fgcmConfig,fgcmPars,fgcmStars):

        self.fgcmLog = fgcmConfig.fgcmLog

        self.fgcmLog.info('Initializing FgcmCorrelations')

        # need fgcmPars because it has the sigFgcm
        self.fgcmPars = fgcmPars

        # need fgcmStars because it has the stars (duh)
        self.fgcmStars = fgcmStars

        # and config numbers
        self.sigFgcmMaxEGray = fgcmConfig.sigFgcmMaxEGray
        self.sigFgcmMaxErr = fgcmConfig.sigFgcmMaxErr
        self.plotPath = fgcmConfig.plotPath
        self.outfileBaseWithCycle = fgcmConfig.outfileBaseWithCycle
        self.cycleNumber = fgcmConfig.cycleNumber
        self.bandRequiredIndex = fgcmConfig.bandRequiredIndex
        self.bandNotRequiredIndex = fgcmConfig.bandNotRequiredIndex
        self.nCore = fgcmConfig.nCore

    def computeCorrelations(self,reserved=False,crunch=False):
        """
        Compute correlations

        parameters
        ----------
        reserved: bool, default=False
           Use reserved stars instead of fit stars?
        crunch: bool, default=False
           Compute based on ccd-crunched values?
        """

        try:
            import treecorr
        except ImportError:
            self.fgcmLog.info("Trecorr not available.  Skipping correlation functions.")
            return

        if (not self.fgcmStars.magStdComputed):
            raise ValueError("Must run FgcmChisq to compute magStd before computeCCDAndExpGray")


        startTime = time.time()
        self.fgcmLog.info('Computing Correlations.')

        # input numbers
        objID = snmm.getArray(self.fgcmStars.objIDHandle)
        objMagStdMean = snmm.getArray(self.fgcmStars.objMagStdMeanHandle)
        objMagStdMeanErr = snmm.getArray(self.fgcmStars.objMagStdMeanErrHandle)
        objNGoodObs = snmm.getArray(self.fgcmStars.objNGoodObsHandle)
        objFlag = snmm.getArray(self.fgcmStars.objFlagHandle)
        objRA = snmm.getArray(self.fgcmStars.objRAHandle)
        objDec = snmm.getArray(self.fgcmStars.objDecHandle)

        obsMagStd = snmm.getArray(self.fgcmStars.obsMagStdHandle)
        obsMagErr = snmm.getArray(self.fgcmStars.obsMagADUModelErrHandle)
        obsBandIndex = snmm.getArray(self.fgcmStars.obsBandIndexHandle)

        objObsIndex = snmm.getArray(self.fgcmStars.objObsIndexHandle)
        obsObjIDIndex = snmm.getArray(self.fgcmStars.obsObjIDIndexHandle)
        obsExpIndex = snmm.getArray(self.fgcmStars.obsExpIndexHandle)
        obsFlag = snmm.getArray(self.fgcmStars.obsFlagHandle)

        goodStars = self.fgcmStars.getGoodStarIndices(onlyReserve=True, checkMinObs=True)

        _, goodObs = self.fgcmStars.getGoodObsIndices(goodStars, expFlag=self.fgcmPars.expFlag)

        # we need to compute E_gray == <mstd> - mstd for each observation
        # compute EGray, GO for Good Obs
        EGrayGO = (objMagStdMean[obsObjIDIndex[goodObs],obsBandIndex[goodObs]] -
                   obsMagStd[goodObs])
        # and need the error for Egray: sum in quadrature of individual and avg errs
        EGrayErr2GO = (obsMagErr[goodObs]**2. -
                       objMagStdMeanErr[obsObjIDIndex[goodObs],obsBandIndex[goodObs]]**2.)

        for bandIndex, band in enumerate(self.fgcmStars.bands):
            fig = plt.figure(figsize=(9, 6))
            fig.clf()

            ax = fig.add_subplot(111)

            use, = np.where((np.abs(EGrayGO) < self.sigFgcmMaxEGray) &
                            (EGrayErr2GO > 0.0) &
                            (EGrayErr2GO <  self.sigFgcmMaxErr**2.) &
                            (EGrayGO != 0.0) &
                            (obsBandIndex[goodObs] == bandIndex))

            cat = treecorr.Catalog(ra=objRA[obsObjIDIndex[goodObs[use]]],
                                   dec=objDec[obsObjIDIndex[goodObs[use]]],
                                   k=EGrayGO[use],
                                   ra_units='deg', dec_units='deg')
            kk = treecorr.KKCorrelation(bin_size=0.1, min_sep=0.5, max_sep=200.0,
                                        sep_units='arcmin')
            kk.process(cat, num_threads=self.nCore)

            ax.plot(kk.meanr, kk.xi, 'r.')
            ax.set_xscale('log')
            ax.set_xlabel('Distance (arcmin)', fontsize=14)
            ax.set_ylabel(r'$E_\mathrm{gray}$ correlation', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=14)
            text = r'$(%s)$' % (self.fgcmPars.bands[bandIndex]) + '\n' + \
                r'$\mathrm{Cycle\ %d}$' % (self.cycleNumber)
            ax.annotate(text,(0.95,0.93),xycoords='axes fraction',ha='right',va='top',fontsize=14)

            if reserved:
                extraName = 'reserved-stars'
            else:
                extraName = 'all-stars'

            if crunch:
                extraName += '_crunched'

            ax.set_title(extraName)
            fig.tight_layout()

            fig.savefig('%s/%s_egraycorr_%s_%s.png' % (self.plotPath,
                                                       self.outfileBaseWithCycle,
                                                       extraName,
                                                       self.fgcmPars.bands[bandIndex]))
            plt.close()

            tempCat = np.zeros(use.size, dtype=[('ra', 'f8'),
                                                ('dec', 'f8'),
                                                ('egray', 'f4'),
                                                ('expnum', 'i4'),
                                                ('mjd', 'f8'),
                                                ('nightindex', 'i4')])
            tempCat['ra'] = objRA[obsObjIDIndex[goodObs[use]]]
            tempCat['dec'] = objDec[obsObjIDIndex[goodObs[use]]]
            tempCat['egray'] = EGrayGO[use]
            tempCat['expnum'] = self.fgcmPars.expArray[obsExpIndex[goodObs[use]]]
            tempCat['mjd'] = self.fgcmPars.expMJD[obsExpIndex[goodObs[use]]]
            tempCat['nightindex'] = self.fgcmPars.expNightIndex[obsExpIndex[goodObs[use]]]
            import fitsio
            fitsio.write('%s_egraycorr_%s_%s.fits' % (self.outfileBaseWithCycle,
                                                     extraName,
                                                     self.fgcmPars.bands[bandIndex]), tempCat, clobber=True)

        self.fgcmLog.info('Done computing correlations in %.2f sec.' %
                          (time.time() - startTime))
