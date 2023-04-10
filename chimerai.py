import torch
import torch.nn as nn
from hyena_operator import HyenaOperator
from t_digest import TDigest

class ChimerAI(nn.Module):
    """
    ChimerAI model for time series prediction.
    
    This model combines a HyenaOperator with a TDigest to provide efficient
    time series prediction. It accepts an input vector of floats at each timestep
    and outputs a prediction vector of the same length.
    """

    def __init__(self, input_size, hyena_params, t_digest_params):
        """
        Initialize the ChimerAI model.
        
        Args:
            input_size (int): Length of the input sequence.
            hyena_params (dict): Parameters for the HyenaOperator.
            t_digest_params (dict): Parameters for the TDigest.
        """
        super(ChimerAI, self).__init__()
        self.input_size = input_size
        self.hyena_operator = HyenaOperator(**hyena_params)
        self.t_digest = TDigest(**t_digest_params)

    def forward(self, x):
        """
        Forward pass of the ChimerAI model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, input_size).
        """
        hyena_output = self.hyena_operator(x)
        t_digest_output = self.t_digest(hyena_output)
        return t_digest_output

    def optimize(self, x, y, optimizer):
        """
        Perform one optimization step.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, sequence_length, input_size).
            optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
        """
        optimizer.zero_grad()
        output = self.forward(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
        return loss.item()

    def run(self, data, optimizer, epochs, batch_size):
        """
        Train the ChimerAI model.
        
        Args:
            data (list): Time series data as a list of float values.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters.
            epochs (int): Number of training epochs.
            batch_size (int): Size of each batch during training.
        """
        for epoch in range(epochs):
            for i in range(0, len(data) - batch_size, batch_size):
                x = torch.tensor(data[i:i + batch_size], dtype=torch.float32).unsqueeze(0)
                y = torch.tensor(data[i + 1:i + batch_size + 1], dtype=torch.float32).unsqueeze(0)
                loss = self.optimize(x, y, optimizer)
                print(f"Epoch: {epoch + 1}/{epochs}, Batch: {i + 1}/{len(data) - batch_size}, Loss: {loss}")
