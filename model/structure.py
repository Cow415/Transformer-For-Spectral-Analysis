"""Module Holds the Architecture Component Templates"""
# Data structure node formats to use
import numpy as np
import torch
import torch.nn as nn

# Perceptron Node Class
class Perceptron: 
    def __init__(self, learning_rate=0.01, epochs=10): 
        self.lr = learning_rate 
        self.epochs = epochs 
        self.weights = None 
        self.bias = None 
        
    def fit(self, X, y): 
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features) 
        self.bias = 0.0 
        
        for _ in range(self.epochs): 
            for idx, x_i in enumerate(X): 
                linear_output = np.dot(x_i, self.weights) + self.bias 
                y_predicted = 1 if linear_output >= 0 else 0 
                
                # Corrected weight and bias update
                error = y[idx] - y_predicted 
                self.weights += self.lr * error * x_i 
                self.bias += self.lr * error 
                
    def predict(self, X): 
        if self.weights is None or self.bias is None:
            raise ValueError("The model must be trained using .fit() before making predictions.")
            
        linear_output = np.dot(X, self.weights) + self.bias 
        return np.where(linear_output >= 0, 1, 0)

# Patch Embeddings
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (
            img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,
                                  stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)
    
# =======================================================================
# Denoising Autoencoder
# Class Encoder
class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(1,16,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(16,32,7,padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

    def forward(self,x):
        return self.net(x)

# Class Decoder
class Sig_Decoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                64,32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32,16,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16,1,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self,z):
        return self.net(z)
    
class BG_Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.ConvTranspose1d(
                64,32,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                32,16,
                kernel_size=2,
                stride=2
            ),
            nn.ReLU(),

            nn.ConvTranspose1d(
                16,1,
                kernel_size=2,
                stride=2
            )
        )

    def forward(self,z):
        return self.net(z)        
