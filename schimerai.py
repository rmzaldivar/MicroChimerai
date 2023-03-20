"""
This module contains the SchimerAI function, a functional and online AI that works on live time series data streams.
"""

import numpy as np
import pandas as pd

from random import uniform
from math import sin, cos

"""
    :param y: The input data for the current timestep.
    :param e: The maximum allowed computation time.
    :param s: The state of the agent at the current timestep.

    :return x: The float output of the agent at the current timestep.
    :return w: The width/standard deviation of the agent's estimate.
    :return s: The state of the agent for the next current timestep.
"""
def schimerai(y : list, e : float, s : dict):
    """
            Initialization Step
    """
    # Before anything, we upload the state of the AI from s.
    agent_count = s['agent_count'] # float
    extra_data_agent_count = s['extra_data_agent_count'] # float
    prev_pred_diff = s['prev_pred_diff'] # float
    W = pd.DataFrame.from_dict(s['W']) # (agent_count x agent_count) matrix
    L = pd.DataFrame.from_dict(s['L']) # (agent_count x agent_count) matrix
    P = pd.DataFrame.from_dict(s['P']) # (agent_count x agent_count) matrix
    AV = pd.DataFrame.from_dict(s['AV']) # (agent_count x 1) list

    # Replace the first extra_data_agent_count columns with the values from y.
    AV[1:extra_data_agent_count] = y[1:extra_data_agent_count]

    # If this is the first timestep, we need to initialize the state of the AI.
    if s == {}:
        agent_count = 100 #arbitary
        extra_data_agent_count = 9 #arbitary
        prev_pred_diff = 0
        W = pd.DataFrame(np.ones((agent_count, agent_count)))
        L = pd.DataFrame(np.ones((agent_count, agent_count)))
        P = pd.DataFrame(np.zeros((agent_count, agent_count)))
        AV = pd.DataFrame(np.ones((agent_count, 1)))

    """
            Update Step
    """
    # Check if we were successfull last time step by seeing if our prediction was closer and update accordingly.
    curr_pred_diff = AV[0] - y[0]
    if curr_pred_diff <= prev_pred_diff:
        W = W.add(P)
    else:
        L = L.add(P)
    P = P.sub(P)
        
    """
            Prediction Step
    """
    # Calculate the prediction for the current timestep.
    G = W.div(W.add(L))
    H = pd.DataFrame(np.ones((agent_count, 1)))
    r = uniform(0, 1)

    queued_connections_row, queued_connections_col = np.where(G<r)

    # Parallelize this later with each calc queued because they all are just mulitplications to H.
    for i,j in zip(queued_connections_row, queued_connections_col):
        P.iat[i,j] = 1
        x, y = e_conv(AV[i], AV[j])
        H[i] *= x
        H[j] *= y

    # Calculate the prediction for the current timestep.
    AV = AV.mul(H)
    x = AV[0]
    w = np.sqrt(abs(curr_pred_diff-prev_pred_diff))

    """
            Output Step
    """
    #Store all the updated values in the state dictionary.
    s['agent_count'] = agent_count
    s['prev_pred_diff'] = curr_pred_diff
    s['W'] = pd.DataFrame.to_dict(W)
    s['L'] = pd.DataFrame.to_dict(L)
    s['P'] = pd.DataFrame.to_dict(P)
    s['AV'] = pd.DataFrame.to_dict(AV)

    return x, w, s


"""
The following functions are used by the schimerai function as helper functions.
"""


def e_conv(x : float,y : float):
    x_new = x*cos(x*y) - y*sin(x*y)
    y_new = x*sin(x*y) + y*cos(x*y)
    return x_new, y_new