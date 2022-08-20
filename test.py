import iss.call_spots.test.test_base
import iss.find_spots.test as find_spots
import iss.extract.test as extract
import iss.omp.test.test_all
import iss.utils.test as utils
import iss.stitch.test as stitch
import iss.setup.test as setup
import iss.register.test as register
import iss.spot_colors.test as spot_colors
import iss.call_spots.test as call_spots
import iss.omp.test as omp
import unittest
import sys
import re


def suite_utils(optimised: bool = False):
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(utils.TestMorphology, 'test'))
    suite.addTest(unittest.makeSuite(utils.TestNPY, 'test'))
    if optimised:
        suite.addTest(unittest.makeSuite(utils.TestOptimisedImfilter, 'test'))
    return suite


def suite_setup():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(setup.TestConfig, 'test'))
    suite.addTest(unittest.makeSuite(setup.TestNotebook, 'test'))
    suite.addTest(unittest.makeSuite(setup.TestTilePos, 'test'))
    return suite


def suite_extract():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(extract.TestFstack, 'test'))
    suite.addTest(unittest.makeSuite(extract.TestStripHack, 'test'))
    return suite


def suite_find_spots(optimised: bool = False):
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(find_spots.TestIsolated, 'test'))
    suite.addTest(unittest.makeSuite(find_spots.TestNeighbour, 'test'))
    if optimised:
        suite.addTest(unittest.makeSuite(find_spots.TestDetectSpots, 'test'))
    return suite


def suite_stitch():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(stitch.TestShift, 'test'))
    suite.addTest(unittest.makeSuite(stitch.TestTileOrigin, 'test'))
    return suite


def suite_register():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(register.TestGetTransform, 'test'))
    suite.addTest(unittest.makeSuite(register.TestGetAverageTransform, 'test'))
    suite.addTest(unittest.makeSuite(register.TestIterate, 'test'))
    return suite


def suite_spot_colors():
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(spot_colors.TestSpotColors, 'test'))
    return suite


def suite_call_spots(optimised: bool = False):
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(call_spots.TestGetDyeChannelIntensityGuess, 'test'))
    suite.addTest(unittest.makeSuite(call_spots.TestGetBleedMatrix, 'test'))
    suite.addTest(unittest.makeSuite(call_spots.TestScaledKMeans, 'test'))
    suite.addTest(unittest.makeSuite(call_spots.TestGetGeneEfficiency, 'test'))
    suite.addTest(unittest.makeSuite(call_spots.TestColorNormalisation, 'test'))
    if optimised:
        suite.addTest(unittest.makeSuite(call_spots.TestGetSpotIntensity, 'test'))
        suite.addTest(unittest.makeSuite(call_spots.TestDotProductScore, 'test'))
        suite.addTest(unittest.makeSuite(call_spots.TestFitBackground, 'test'))
    return suite


def suite_omp(optimised: bool = False):
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(iss.omp.test.test_all.TestFittingStandardDeviation, 'test'))
    suite.addTest(unittest.makeSuite(omp.TestCountSpotNeighbours, 'test'))
    if optimised:
        suite.addTest(unittest.makeSuite(omp.TestFitCoefs, 'test'))
        suite.addTest(unittest.makeSuite(omp.TestGetBestGene, 'test'))
        suite.addTest(unittest.makeSuite(omp.TestGetAllCoefs, 'test'))
    return suite


def suite_all(optimised: bool = False):
    suite = suite_utils(optimised)
    suite.addTest(suite_setup())
    suite.addTest(suite_extract())
    suite.addTest(suite_find_spots(optimised))
    suite.addTest(suite_stitch())
    suite.addTest(suite_register())
    if optimised:
        suite.addTest(suite_spot_colors())
    suite.addTest(suite_call_spots(optimised))
    suite.addTest(suite_omp(optimised))
    return suite


if __name__ == '__main__':
    # This does unittests on optimised functions if extra argument "-o" passed.
    # If no argument passed, it will only do unit tests on functions with no optimisation.
    if len(sys.argv) == 1:
        optimised = False
    elif len(sys.argv) > 2:
        raise ValueError('Too many arguments passed')
    else:
        test_type = " ".join(re.findall("[a-zA-Z]+", sys.argv[1])).lower()   # remove all punctuation
        if test_type in ["optimised", "jax", "o"]:
            optimised = True
        elif test_type in ["plotting", ""]:
            optimised = False
        else:
            raise ValueError(f"Only valid argument to pass is optimised but\n{sys.argv[1]}\nwas passed. "
                             f"Provide no input argument to not test optimised functions.")
    suite = suite_omp(optimised)
    sys.argv = sys.argv[:1]  # get error in unittest.main if sys.argv contains input other than file
    unittest.main(defaultTest='suite', exit=True)
