import torch
import torch.nn as nn
import numpy as np

class LearnableKalmanFilter(nn.Module):
    def __init__(self, state_dim=48, batch_size=None, measurement_var=1e-2, process_var=1e-3):
        super(LearnableKalmanFilter, self).__init__()
        self.state_dim = state_dim
        self.batch_size = batch_size
        
        # Initialize with the same values that worked well for you
        self.log_process_var = nn.Parameter(torch.ones(state_dim) * np.log(process_var))
        self.log_measurement_var = nn.Parameter(torch.ones(state_dim) * np.log(measurement_var))
        
        # Fixed matrices: same as your FilterPy setup
        self.F = nn.Parameter(torch.eye(state_dim), requires_grad=False)
        self.B = nn.Parameter(torch.eye(state_dim), requires_grad=False)
        self.H = nn.Parameter(torch.eye(state_dim), requires_grad=False)
        
        # State and covariance for each sequence in batch
        self.x = None          # Will be [batch_size, state_dim, 1]
        self.P = None          # Will be [batch_size, state_dim, state_dim]
        self.initialized = False
        
    def reset(self):
        """Reset filter states"""
        self.initialized = False
        self.x = None
        self.P = None
        self.batch_size = None
        
    def initialize(self, x_init):
        """Initialize states and covariances
        
        Args:
            x_init: Initial states [batch_size, state_dim] or [batch_size, state_dim, 1]
        """
        # Determine batch size from input
        if x_init.dim() == 3:  # [batch_size, state_dim, 1]
            self.batch_size = x_init.shape[0]
            self.x = x_init
        elif x_init.dim() == 2:  # [batch_size, state_dim]
            self.batch_size = x_init.shape[0]
            self.x = x_init.unsqueeze(-1)  # Add column dimension
        else:
            raise ValueError(f"Expected x_init with 2 or 3 dimensions, got {x_init.dim()}")
        
        # Initialize covariance matrices for each sequence
        self.P = torch.eye(self.state_dim, device=x_init.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.initialized = True
        
        return self.x
        
    def predict(self, u=None):
        """Prediction step (time update) for batch
        
        Args:
            u: Control inputs [batch_size, state_dim] or [batch_size, state_dim, 1]
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before predict")
        
        device = self.x.device
        
        # Handle control input
        if u is None:
            u = torch.zeros_like(self.x)
        elif u.dim() == 2:  # [batch_size, state_dim]
            u = u.unsqueeze(-1)  # [batch_size, state_dim, 1]
            
        if u.shape[0] != self.batch_size:
            raise ValueError(f"Control input batch size {u.shape[0]} doesn't match filter batch size {self.batch_size}")
            
        # Process noise covariance - create batch version
        Q_diag = torch.exp(self.log_process_var)
        Q = torch.diag_embed(Q_diag).to(device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # Expand system matrices for batch operations
        F_batch = self.F.unsqueeze(0).repeat(self.batch_size, 1, 1)
        B_batch = self.B.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # State prediction (batched matrix multiplication)
        self.x = torch.bmm(F_batch, self.x) + torch.bmm(B_batch, u)
        
        # Covariance prediction
        self.P = torch.bmm(torch.bmm(F_batch, self.P), F_batch.transpose(1, 2)) + Q
        
        return self.x
        
    def update(self, z):
        """Measurement update step for batch
        
        Args:
            z: Measurements [batch_size, state_dim] or [batch_size, state_dim, 1]
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before update")
            
        device = self.x.device
        
        # Handle measurement input
        if z.dim() == 2:  # [batch_size, state_dim]
            z = z.unsqueeze(-1)  # [batch_size, state_dim, 1]
            
        if z.shape[0] != self.batch_size:
            raise ValueError(f"Measurement batch size {z.shape[0]} doesn't match filter batch size {self.batch_size}")
            
        # Measurement noise covariance - create batch version
        R_diag = torch.exp(self.log_measurement_var)
        R = torch.diag_embed(R_diag).to(device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # Expand measurement matrix for batch operations
        H_batch = self.H.unsqueeze(0).repeat(self.batch_size, 1, 1)
        
        # Innovation (y = z - Hx) - batched
        y = z - torch.bmm(H_batch, self.x)
        
        # Innovation covariance (S = HPH' + R) - batched
        S = torch.bmm(torch.bmm(H_batch, self.P), H_batch.transpose(1, 2)) + R
        
        # Kalman gain (K = PH'S⁻¹) - batched
        # We need to carefully handle batch matrix inverse
        S_inv = torch.inverse(S)  # PyTorch handles batch inverse
        K = torch.bmm(torch.bmm(self.P, H_batch.transpose(1, 2)), S_inv)
        
        # State update - batched
        self.x = self.x + torch.bmm(K, y)
        
        # Covariance update (Joseph form) - batched
        I = torch.eye(self.state_dim, device=device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        KH = torch.bmm(K, H_batch)
        I_KH = I - KH
        
        self.P = torch.bmm(torch.bmm(I_KH, self.P), I_KH.transpose(1, 2)) + \
                torch.bmm(torch.bmm(K, R), K.transpose(1, 2))
                
        return self.x

    def get_state(self):
        """Get current state estimate"""
        if not self.initialized:
            return None
        return self.x.squeeze(-1)  # Return [batch_size, state_dim]
