from __future__ import print_function

import numpy as np
import fitsio
import os
import sys
import esutil
import scipy.optimize as optimize

from fgcmConfig import FgcmConfig
from fgcmParameters import FgcmParameters
from fgcmChisq import FgcmChisq
from fgcmStars import FgcmStars
from fgcmLUT import FgcmLUTSHM
from fgcmSuperStarFlat import FgcmSuperStarFlat
from fgcmRetrieval import FgcmRetrieval
from fgcmApertureCorrection import FgcmApertureCorrection
from fgcmBrightObs import FgcmBrightObs
from fgcmExposureSelector import FgcmExposureSelector

from sharedNumpyMemManager import SharedNumpyMemManager as snmm

class FgcmFitCycle(object):
    """
    """
    def __init__(self,configFile):
        self.fgcmConfig = FgcmConfig(configFile)

    def run(self):
        """
        """

        # Check if this is the initial cycle
        initialCycle = False
        if (self.fgcmConfig.cycleNumber == 0):
            initialCycle = True

        # Generate or Read Parameters
        if (initialCycle):
            self.fgcmPars = FgcmParameters(fgcmConfig=self.fgcmConfig)
        else:
            self.fgcmPars = FgcmParameters(parFile=self.fgcmConfig.inParameterFile)


        # Read in Stars
        self.fgcmStars = FgcmStars(self.fgcmConfig)

        # Read in LUT
        self.fgcmLUT = FgcmLUTSHM(self.fgcmConfig.lutFile)

        # And prepare the chisq function
        self.fgcmChisq = FgcmChisq(self.fgcmConfig,self.fgcmPars,
                                   self.fgcmStars,self.fgcmLUT)

        # And the exposure selector
        self.expSelector = FgcmExposureSelector(self.fgcmConfig,self.fgcmPars)

        # And the Gray code
        self.fgcmGray = FgcmGray(self.fgcmConfig,self.fgcmPars,self.fgcmStars)

        # Apply aperture corrections and SuperStar if available
        # select exposures...
        if (not initialCycle):
            ## FIXME: write code to apply aperture corrections and superstar flats

            pass


        # Flag stars with too few exposures
        goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
        self.fgcmStars.selectStarsMinObs(goodExpsIndex)

        # Get m^std, <m^std>, SED for all the stars.
        parArray = fgcmPars.getParArray(fitterUnits=False)
        if (not initialCycle):
            # get the SED from the chisq function
            self.fgcmChisq(parArray,computeSEDSlopes=True)

            # flag stars that are outside the color cuts
            self.fgcmStars.performColorCuts()
        else:
            # need to go through the bright observations

            # first, use fgcmChisq to compute m^std for every observation
            self.fgcmChisq(parArray)

            # run the bright observation algorithm, computing SEDs
            brightObs = FgcmBrightObs(self.fgcmConfig,self.fgcmPars,self.fgcmStars)
            brightObs.selectGoodStars(computeSEDSlopes=True)

            # flag stars that are outside our color cuts
            self.fgcmStars.performColorCuts()

            # get expGray for the initial selection
            self.fgcmGray.computeExpGrayForInitialSelection()

            # and select good exposures/flag bad exposures
            self.expSelector.selectGoodExposuresInitialSelection(self.fgcmGray)

            # reflag bad stars with too few observations
            #  (we don't go back and select exposures at this point)
            goodExpsIndex, = np.where(self.fgcmPars.expFlag == 0)
            self.fgcmStars.selectStarsMinObs(goodExpsIndex)


        # Select calibratable nights
        self.expSelector.selectCalibratableNights()

        # Perform Fit (subroutine)
        self._doFit()

        # Compute CCD^gray and EXP^gray
        gray.computeCCDAndExpGray()

        # Compute Retrieved chromatic integrals
        retrieval = FgcmRetrieval(self.fgcmConfig,self.fgcmPars,self.fgcmLUT)
        retrieval.computeRetrievedIntegrals()

        # Compute SuperStar Flats
        superStarFlat = FgcmSuperStarFlat(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        superStarFlat.computeSuperStarFlats()

        # Compute Aperture Corrections
        aperCorr = FgcmApertureCorrection(self.fgcmConfig,self.fgcmPars,self.fgcmGray)
        aperCorr.computeApertureCorrections()


        # Make Zeropoints
        ## FIXME: Write this code

        # Save parameters
        outParFile = '%s/%s_cycle%02d_parameters.fits' % (self.fgcmConfig.outputPath,
                                                          self.fgcmConfig.outfileBase,
                                                          self.fgcmConfig.cycleNumber)
        self.fgcmPars.saveParFile(outParFile)

        # Save yaml for input to next fit cycle
        ## FIXME: Write this code

    def _doFit(self):
        """
        """
        # get the initial parameters
        parInitial = self.fgcmPars.getParArray(fitterUnits=True)
        # and the fit bounds
        parBounds = self.fgcmPars.getParBounds(fitterUnits=True)

        pars, chisq, info = optimize.fmin_l_bfgs_b(self.fgcmChisq,   # chisq function
                                                   parInitial,       # initial guess
                                                   fprime=None,      # in fgcmChisq()
                                                   args=(True,True), # fitterUnits, deriv
                                                   approx_grad=False,# don't approx grad
                                                   bounds=parBounds, # boundaries
                                                   m=10,             # "variable metric conditions"
                                                   factr=1e2,        # highish accuracy
                                                   pgtol=1e-9,       # gradient tolerance
                                                   maxfun=self.fgcmConfig.maxIter,
                                                   maxIter=self.fgcmConfig.maxIter,
                                                   ipring=0,         # only one output
                                                   callback=None)    # no callback

        # FIXME: add plotting of chisq

        # save new parameters
        self.fgcmPars.reloadParArray(pars, fitterUnits=True)