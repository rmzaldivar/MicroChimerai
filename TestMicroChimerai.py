"""
This module contains the unit tests for the MicroChimerai class.
"""

import unittest
import numpy as np

import MicroChimerai

def testMicroChimerai(unittest.TestCase):
    # We use the empty_MCAI to test the functions of the MicroChimerai class by making sure all the member
    # variables are initialized correctly and that the functions return the correct values.
    empty_MCAI = MicroChimerai.MicroChimerai()

    def test_sample(self):
        pass

    def test_update_WL(self):
        # We'll test this function by making sure that the W and L matrices are updated correctly.

        # Generate a random set of lagged values and we make random values in P equal to 1 to simulate us being mid training.
        #lagged_values = np.random.rand(225)
        # Add 1 to random values in P
        #empty_MCAI.P = empty_MCAI.P.add(np.random.randint(2, size=(empty_MCAI.agent_count, empty_MCAI.agent_count)))

        #prediction_guess = empty_MCAI.update_WL(lagged_values)

        pass



    def test_create_prediction(self):
        pass

