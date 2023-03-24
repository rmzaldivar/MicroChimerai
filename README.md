# Microprediction - chimer·ai (MicroChimerai) Implementation

This repository implements the chimer·ai algorithm by Richard Zaldivar of [chimer.ai](http://chimer.ai/) using the algorithms created and implemented by Dr. Peter Cotton, inventor of [microprediction](https://www.microprediction.com/) and writer of [Microprediction
Building an Open AI Network](https://mitpress.mit.edu/9780262047326/), at https://github.com/microprediction/microprediction.

## Algorithm + Implementation Summary

The chimer·ai algorithm is a multi agent reinforcement learning algorithm that aims to create a group of agents that have a single way to communicate with each other and simply optimize the timestep in which they do so within the timesteps of an overall simulation. This optimization is done by updating a beta binomial model for each agent.


## Microchimerai Algorithm
### Variable Key
agent_count; int = Total number of "agents" that we keep track of. This includes 225 agents that provide our input from some source each timestep as well as an arbitrary number of additional agents at least of size 225 as well.

prev_pred_diff = This is the average difference between our guess for the 225 input values and the true 225 output. Depending on how this changes between timesteps we either consider the timestep a success or failure in terms of the beta binomial models.

W; (agent_count, agent_count) matrix = This matrix tracks the "a"/alpha variable of each the beta-binomial model. There is a model for each agent's connection to each other and hence the size of the matrix. It is initialized to all 1's.

L; (agent_count, agent_count) matrix = This matrix is functionally the same as W but tracks the "b"/beta value for the beta-binomials. It is initialized to all 1's.

P; (agent_count, agent_count) matrix = This matrix is initialized to all zeroes and tracks the connections that were used from the last loop. This is passed to us the next time step so we can update accordingly once we receive the correct answer for the last timestep.

AV; (1, agent_count) matrix = This matrix contains the float "value" of each agent. For the agents that are used to import data, this value is our input data. For the others, this value is preserved from last timestep.

### Algorithm

Input: lagged_values ((1, 225) matrix)

1. Subtract lagged_values from the first 225 values of AV and find the average value of that output. We'll call it avg_diff.

2. If avg_diff is less than our prev_pred_diff, we have gotten more accurate and add our P matrix to W to represent adding a win to our beta-binomial model. We add it to L if we have gotten less accurate.

3. Next we place lagged_values in the first 225 slots of AV so when the next timestep is run all agents have access to the new info.

4. Generate a set of helper matrices to help with the prediction that will occur over the next couple steps. These include a matrix G which contains (a/(a+b)) for each beta-binomial model, H to contain the factors that we multiply by AV at the end to generate the predictions, and two arrays that list the indices of the connections to check.

5. We generate our predictions by sampling a uniform [0,1] distribution and seeing which values of G are less than that. Each of those indices are used to queue a convolution on the corresponding two values in AV. Those convolutions are multiplied by the initial values of 1 of H to create a list of factors to multiply AV by and generate a prediction for all values in the next timestep.

6. Update our prev_pred_diff with avg_diff and return our prediction.

Return: prediction ((1, agent_count) matrix)

## Explanation

The idea behind the chimer·ai algorithm is that random processes and trivial operations are the demonstrable foundation for many types of what we would consider intelligence in the real world. If we take the idea to the extreme, we can consider a bit that flips between 0 and 1 randomly. If we wanted it to seem like there was a pattern to the bit flipping to another person in order to trick them into thinking it had a pattern, what could we do?

If we could stop the bit and then show people it, we could convince them it blinks in any pattern we want. What exaxtly are we doing here? Because we have no gurantee of any type of behavior from the random bit, there is no gurantee that we can trick others into an exact pattern. Basically the bit creates a pattern which effects our output by us trying to do the same to it. To zoom out a bit and give the random bit the respect it deserves we can see the person and bit as a system with 2 agents where each "freezes" the other while they undergo a random process from the perspective of the frozen agent. Some state of the random process triggers the other to start which causes the current one to freeze (there's probably some neat comparison to the halting problem but maybe I'll get to that at some point ;) ). How are we getting this bit to appear as 1 most of the time to everybody we show when it's just two random processes that upon reaching some state stop each other?

Will continue this readme soon. Planning on talking about how given the microprediction skater function format, we can have a group of agents have a connection value to each other. The agents only optimize their stop time on those connections which causes an operation that is "random" from the perspective of the agent but can be optimized observably with improving stop time.




