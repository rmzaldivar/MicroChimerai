from microprediction import MicroCrawler
import numpy as np
import pandas as pd

from random import uniform
from math import sin, cos

class MicroChimerai(MicroCrawler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.agent_count = int(1000) # 3X the length of the lagged_values/input data.
        self.prev_pred_diff = 0
        self.W = pd.DataFrame(np.ones((self.agent_count, self.agent_count)))
        self.L = pd.DataFrame(np.ones((self.agent_count, self.agent_count)))
        self.P = pd.DataFrame(np.zeros((self.agent_count, self.agent_count)))
        self.AV = pd.DataFrame(np.ones((self.agent_count)))

    def sample(self, lagged_values, lagged_times=None, name=None, delay=None, **ignored):
        # First we update the W and L matrices based on if we got the previous prediction correct or not.
        avg_diff = self.update_WL(lagged_values)

        # We assign all the values from lagged_values to however many slots it takes up in AV.
        self.AV.loc[:len(lagged_values)-1, 0] = lagged_values

        # Create all matrices needed to calculate the prediction for the next timestep.
        G = self.W.div(self.W.add(self.L))
        H = pd.DataFrame(np.ones(self.agent_count))
        r = uniform(0, 1)
        queued_connections_row, queued_connections_col = np.where(G<r)

        # Given all the connections between agents we need to use, we create a prediction for the next timestep.
        prediction = self.create_prediction(queued_connections_row, queued_connections_col, H)

        # We update the previous prediction difference.
        self.prev_pred_diff = avg_diff

        return prediction
    
    def update_WL(self, curr_values):
        # First we'll compare curr_vales to our predictions of it in AV i.e the first len(lagged_values) elements of AV.
        diff = self.AV.values[:len(curr_values)] - curr_values
        curr_avg_diff = np.average(diff)

        # If we have a better prediction than last time, we consider this a success in terms of our Bayesian updating.
        if self.prev_pred_diff <= curr_avg_diff:
            self.W = self.W.add(self.P)
        else:
            self.L = self.L.add(self.P)
        self.P = self.P.sub(self.P)

        return curr_avg_diff
    
    def create_prediction(self, queued_rows, queued_cols, H_matrix):
        for i,j in zip(queued_rows, queued_cols):
            self.P.iat[i,j] = 1
            x_n, y_n = e_conv(self.AV.iloc[i], self.AV.iloc[j])
            H_matrix.loc[i] = H_matrix.loc[i].mul(x_n)
            H_matrix.loc[j] = H_matrix.loc[j].mul(y_n)

        self.AV = self.AV.mul(H_matrix)
        curr_prediction = self.AV[0].values[0]

        return curr_prediction

"""
This function convolves the float values of x and y using rotation matrices to create an operation that can be used to
improve x and y if the model learns the timing to apply the operation as opposed to improving an operation like other
models.
"""
def e_conv(x : float, y : float):
    x_new = x*cos(x*y) - y*sin(x*y)
    y_new = x*sin(x*y) + y*cos(x*y)
    return x_new, y_new