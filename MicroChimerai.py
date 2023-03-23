from microprediction import MicroCrawler
import numpy as np
import pandas as pd

from random import uniform
from math import sin, cos

class MicroChimerai(MicroCrawler):
    def __init__(self, **kwargs):
        super().init(**kwargs)

        self.agent_count = int(1000) # 3X the length of the lagged_values/input data.
        self.prev_pred_diff = 0
        self.W = pd.DataFrame(np.ones((self.agent_count, self.agent_count)))
        self.L = pd.DataFrame(np.ones((self.agent_count, self.agent_count)))
        self.P = pd.DataFrame(np.zeros((self.agent_count, self.agent_count)))
        self.AV = pd.DataFrame(np.ones((self.agent_count, 1)))

    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        # First we'll compare lagged values to our predictions of it in AV i.e the first len(lagged_values) elements of AV.
        diff = self.AV.values[:len(lagged_values)] - lagged_values
        curr_avg_diff = np.average(diff)

        # If we have a better prediction than last time, we consider this a success in terms of our Bayesian updating.
        if self.prev_pred_diff <= curr_avg_diff:
            self.W = self.W.add(self.P)
        else:
            self.L = self.L.add(self.P)
        self.P = self.P.sub(self.P)

        # Now we assign all the values from lagged_values to however many slots it takes up in AV.
        self.AV.loc[:len(lagged_values)-1, 0] = lagged_values

        # Now we create all matrices needed to calculate the prediction for the next timestep.
        G = self.W.div(self.W.add(self.L))
        H = pd.DataFrame(np.ones(self.agent_count, 1))
        r = uniform(0, 1)

        queued_connections_row, queued_connections_col = np.where(G<r)

        # TODO Parallelize this later with each calc queued because they all are just mulitplications to H.
        for i,j in zip(queued_connections_row, queued_connections_col):
            self.P.iat[i,j] = 1
            x_n, y_n = e_conv(self.AV[i].values[0], self.AV[j].values[0])
            H[i] *= x_n
            H[j] *= y_n

        # Now we calculate the prediction for the next timestep.
        self.AV = self.AV.mul(H)
        prediction = self.AV[0].values[0]

        # We update the previous prediction difference.
        self.prev_pred_diff = curr_avg_diff

        return prediction


def e_conv(x, y):
    x_new = x*cos(x*y) - y*sin(x*y)
    y_new = x*sin(x*y) + y*cos(x*y)