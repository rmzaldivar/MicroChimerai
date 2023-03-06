"""
This python script is used to implement a novel AI algorithm called Chi. This algorithm is designed to create an agent with a probabilistic degree of free will that learns to 
use it to accomplish an objective. The algorithm is run in a simulatiion for a variable number of time steps. This script is a bare bones implementation of the algorithm 
which is used to accomplish the following scenario which ends up being the basis for any type of free will derived answer by an agent:

Suppose we decide two sequence lengths of n and m where n > m. First, the agent is given a pandas dataframe to operate on that we will use to teach oscillate uniformly
between some amount of states encoded by m bits where the amount of states is dependent on uniformly oscillating between some n-m bits. We will be simulating a quantum
environment represented by a set of probabilistic waves that operate on the aformentioned bits. The total nums of bits in our sim will be 2n - m + 2. The left and right most bits will
oscillate between 0 and 1. The n-m bits to the right of the right most bit and to the left of the left most bit will be the ones that oscillate between our predetermined n-m states.
We display the same n-m bits on both sides of the left and right most bits so the agent can learn to use the bits to oscillate between the states.

The agent acts by willfully choosing to flip bits over time in order to oscillate based on the repeated patterns of the bits. The waves learns over time by randomly growing and
shrinking on different timesteps as a random function based on how many time steps have passed since they were created. On timesteps that correspond to the
fibonacci sequence the agent will flip a coin that either grows it to the left by one or to the right by one. For us, on timesteps that correspond to the fibonacci sequence we will
insert an extra bit to the left and right of the middle m bits.

Whenever bits are flipped they do one of two actions:
If a bit is flipped from 0 to 1, a new probabilistic wave is created on the bit that was flipped. If a bit is flipped from 1 to 0, all waves on that bit are destroyed.

Whether a bit is flipped or not is determined by the agent's will deterministically. A bit is flipped as a function of all probability waves that are on it. Each probability wave
has a constant weight of .5 and we take .5 raised to the number of waves on the bit, which we'll call p. We sample a random number between 0 and 1 and if the random number is less
than p, we flip the bit. If the random number is greater than p, we do not flip the bit. The agent's will is determined by the probability of flipping a bit. Groups of these waves
learn to cooperate on not flipping bits because the waves that don't flip bits will have a higher probability of flipping a bit to zero because they are more random, they have to
grow in number on the same bits they decide not to flip to increase the probability of it not flipping. Therefore, they are equally disincentived from shrinking and growing and are
only able to operate around the least unlikely bits to flip. This causes it to learn to grow around following our "guranteed" oscillation of n-m bits.
"""

# First, we import the necessary libraries
import numpy as np
import pandas as pd
import random

"""
First we'll define the Chi class. It is an AI agent which lives a pandas dataframe with a height of two and a width of 2n - m + 2. It also has a dictionary that every timestep stores
all waves created in that timestep with the key as the timestep and the value as all the waves. The top row contains the bits that oscillate between 0 and 1 and the bottom row
contains a list of all wave times on each bit. 

The Chi class has the following methods:
__init__(self, n, m): Initializes the Chi class.
load_fib_csv(self, csv_name): Loads a csv file that contains the fibonacci sequence up to 2000th number. This is used to determine which timesteps the agent will grow and shrink
update_bits(self): Updates the bits in the dataframe based on the waves on each bit. Delete all waves from the keys of the dictionary that are stored in the 
bottom row of the dataframe. 
update_waves(self, sim_time): Updates the waves in the dictionary based on time each wave has been alive since the start of the sim according to the dictionary keys.
age_chi(self, sim_time_add): Starts counting up by sim_time_add time steps and applying all mentioned operations to the dataframe that represents the Chi agent.
"""
class Chi:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.fib_nums = set(pd.read_csv("first-2000-fibonacci.csv", header=None).iloc[:, 0])

        self.chi_body = pd.DataFrame(columns=range(2 * self.n - self.m + 2))
        self.chi_mind = {}
        self.chi_age = 0

    def update_bits(self):
        rand_this_timestep = random.uniform(0, 1)
        for i in range(2 * self.n - self.m + 2):
            wave_count = len(self.chi_body.iloc[1, i])
            p = .5 ** wave_count
            if rand_this_timestep < p:
                self.chi_body.iloc[0, i] = 1 - self.chi_body.iloc[0, i]
            if self.chi_body.iloc[0, i] == 0:
                self.chi_body.iloc[1, i] = []

    def update_waves(self, sim_time):
        #for wave_time in self.chi_mind:
            #if self.chi_age - wave_time