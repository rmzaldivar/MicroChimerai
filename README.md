# Skater chimer·ai Implementation

This repository implements the chimer·ai algorithm by Richard Zaldivar of [chimer.ai](http://chimer.ai/),  with "skater" functions, created and implemented by Dr. Peter Cotton, inventor of [microprediction](https://www.microprediction.com/) and writer of [Microprediction
Building an Open AI Network](https://mitpress.mit.edu/9780262047326/), at https://github.com/microprediction/timemachines. 

## Algorithm + Implementation Summary

The chimer·ai algorithm is a multi agent reinforcement learning algorithm that aims to create a group of agents that have a single way to communicate with each other and simply optimize the timestep in which they do so within the timesteps of an overall simulation.

Skater functions take a single float input and an optional, but not guranteed, set of supplementary data and create a single float output for each timestep we want to predict. We can track a set of values between timesteps that are detailed on [this site](https://microprediction.github.io/timemachines/interface), but we only use the input, y, the expiry time, e, and the state dictionary, s. The following algorithm details how we'll create a skater function out of the chimer·ai algorithm.

## Skater chimer·ai Algorithm
###Variable Key
agent_count; int = Total number of "agents" that we keep track of. This includes the first agent that provides input and the others that provide suplementary data as well.
W; (agent_count, agent_count) matrix =
L; (agent_count, agent_count) matrix = 
P; (agent_count, agent_count) matrix = 
 to be continued


### Input
y: Vector of values to update our model with.

e: A float that specifies how much time to allow before needing an output to submit it. Increase the amount depending on if we start to predict multiple times.

s: A dictionary of our current state.

### Output
x: A float that is our prediction for y’s first index value’s next value

m: Standard deviation for our estimate

s: Our updated dictionary of our updated state

### Algorithm

1. Load


## Explanation
