"""
Transformer encoder-decoder components for time series modeling.

This module implements transformer architectures specifically designed for time series data,
including convolutional layers, encoder-decoder structures, and specialized components
like LoRA linear layers and adaptive Fourier neural operators.
"""

import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import defaultdict


class ConvLayer(nn.Module):
    """
    Convolutional layer for downsampling time series sequences.
    
    This layer applies 1D convolution followed by batch normalization,
    ELU activation, and max pooling to reduce the temporal dimension.
    """
    
    def __init__(self, c_in):
        """
        Initialize the convolutional layer.
        
        Args:
            c_in (int): Number of input channels.
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """
        Apply convolutional downsampling to input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            
        Returns:
            torch.Tensor: Downsampled tensor of reduced sequence length.
        """
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    Transformer encoder layer with self-attention and feed-forward networks.
    
    This layer implements the standard transformer encoder architecture with
    multi-head self-attention followed by position-wise feed-forward networks.
    """
    
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        """
        Initialize the encoder layer.
        
        Args:
            attention (nn.Module): Attention mechanism to use.
            d_model (int): Dimension of the model embeddings.
            d_ff (int, optional): Dimension of the feed-forward network. Defaults to 4*d_model.
            dropout (float): Dropout probability. Defaults to 0.1.
            activation (str): Activation function ("relu" or "gelu"). Defaults to "relu".
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        """
        Apply transformer encoder layer to input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            
        Returns:
            tuple: (output, attention_weights) where:
                - output: Encoded tensor of shape [B, L, D]
                - attention_weights: Attention weights from the attention mechanism
        """
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    Transformer encoder composed of multiple encoder layers.
    
    This module stacks multiple encoder layers and optionally includes
    convolutional layers for hierarchical feature extraction.
    """
    
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        """
        Initialize the encoder.
        
        Args:
            attn_layers (list): List of encoder layer modules.
            conv_layers (list, optional): List of convolutional layer modules.
            norm_layer (nn.Module, optional): Normalization layer to apply to final output.
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        """
        Apply transformer encoder to input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, D].
            attn_mask (torch.Tensor, optional): Attention mask tensor.
            
        Returns:
            tuple: (output, attention_weights_list) where:
                - output: Encoded tensor of shape [B, L, D]
                - attention_weights_list: List of attention weights from each layer
        """
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Lora_linear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) linear layer with complex number support.
    
    This layer implements LoRA adaptation technique using complex-valued
    matrices for enhanced representation capacity.
    """
    
    def __init__(self, alpha, r, d_model):
        """
        Initialize the LoRA linear layer.
        
        Args:
            alpha (float): Scaling factor for the LoRA adaptation.
            r (int): Rank of the low-rank decomposition.
            d_model (int): Dimension of the model embeddings.
        """
        super().__init__()
        self.alpha = alpha
        self.r = r
        # Decompose complex parameters into real and imaginary parts
        self.A_real = nn.Parameter(torch.randn(r, d_model), requires_grad=True)  # Real part
        self.A_imag = nn.Parameter(torch.randn(r, d_model), requires_grad=True)  # Imaginary part
        self.B_real = nn.Parameter(torch.zeros(d_model, r), requires_grad=True)  # Real part
        self.B_imag = nn.Parameter(torch.zeros(d_model, r), requires_grad=True)  # Imaginary part

    def forward(self, x):
        """
        Apply LoRA transformation to input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Transformed tensor with LoRA adaptation applied.
        """
        # Combine real and imaginary parts into complex form
        A = self.A_real + 1j * self.A_imag  # Complex A
        B = self.B_real + 1j * self.B_imag  # Complex B
        
        # Compute output using complex multiplication
        result = x @ (B @ A) * (self.alpha / self.r)
        return result


class AdaptiveFourierNeuralOperator(nn.Module):
    """
    Adaptive Fourier Neural Operator for time series processing.
    
    This module implements a frequency-domain operator that adapts to
    different temporal granularities using expert mixture and LoRA techniques.
    """
    
    def __init__(self, args, dim):
        """
        Initialize the Adaptive Fourier Neural Operator.
        
        Args:
            args (object): Configuration arguments containing model parameters.
            dim (int): Dimension of the input embeddings.
        """
        super().__init__()
        self.args = args
        self.hidden_size = dim
        self.scale = 0.02
        self.num_gran = self.args.num_gran
        
        # Weight parameters for Fourier transformations
        self.w1 = torch.nn.Parameter(self.scale * torch.randn(2, self.hidden_size//2+1, self.hidden_size//2+1))
        self.b1 = torch.nn.Parameter(self.scale * torch.randn(2,  self.hidden_size//2+1))
        self.w2 = torch.nn.Parameter(self.scale * torch.randn(2,  self.hidden_size//2+1, self.hidden_size//2+1))
        self.b2 = torch.nn.Parameter(self.scale * torch.randn(2, self.hidden_size//2+1))
        
        # Expert mixture layers for different temporal granularities
        self.gra_linear_1 = nn.ModuleList(
            [Lora_linear(4, 1, dim // 2 + 1) for _ in range(self.num_gran)]
        )
        self.gra_linear_2 = nn.ModuleList(
            [Lora_linear(4, 1, dim // 2 + 1) for _ in range(self.num_gran)]
        )
        
        # Granularity embedding and feature parameters
        self.gra_embedding = nn.Linear(5, self.hidden_size // 2 + 1)
        self.gra_feature = nn.Parameter(torch.randn(self.num_gran, self.hidden_size // 2 + 1))
        self.relu = nn.ReLU()

    def set_profiler(self, prof, name):
        """
        Set profiler for performance monitoring.
        
        Args:
            prof (object): Profiler object.
            name (str): Name for profiling.
        """
        self._prof = prof
        self._name = name

    def multiply(self, input, weights):
        """
        Perform Einstein summation multiplication.
        
        Args:
            input (torch.Tensor): Input tensor.
            weights (torch.Tensor): Weight tensor.
            
        Returns:
            torch.Tensor: Result of Einstein summation.
        """
        return torch.einsum('...bd,dk->...bk', input, weights)

    def replace_null(self, gra_list):
        """
        Identify and handle null granularity entries.
        
        Args:
            gra_list (list): List of granularity indicators.
            
        Returns:
            tuple: (processed_list, null_indices) where:
                - processed_list: Original list
                - null_indices: Indices of null entries
        """
        null_indices = []
        for i in range(len(gra_list)):
            if sum(gra_list[i]) == 0:
                null_indices.append(i)
        return gra_list, null_indices

    def forward(self, x, time_gra, spatial_size=None):
        """
        Apply adaptive Fourier neural operator to input sequence.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C].
            time_gra (list): Temporal granularity information.
            spatial_size (int, optional): Spatial dimension size.
            
        Returns:
            torch.Tensor: Transformed tensor in time domain.
        """
        B, N, C = x.shape
        x = torch.fft.rfft(x, dim=2, norm='ortho')

        # Process temporal granularity information
        time_gra = torch.stack(time_gra)
        time_gra = time_gra.permute(1, 0).to(x.device).to(torch.float)
        time_gra, null_index = self.replace_null(time_gra)
        
        # Generate granularity embeddings and attention
        gra = self.gra_embedding(time_gra)
        gra_atten = self.gra_attention(gra, self.gra_feature)
        gra_atten = gra_atten.permute(1, 0)
        
        # Prepare attention for different tasks
        gra_atten = gra_atten.unsqueeze(-1).repeat(1, 1, x.size(-1))
        if self.args.task_name == 'imputation':
            gra_atten = gra_atten.unsqueeze(-2).repeat(1, 1, x.size(1), 1)
        else:
            gra_atten = gra_atten.unsqueeze(-2).repeat(1, self.args.enc_in, x.size(1), 1)
            
        # First expert mixture layer
        moe_out_1 = self.gra_linear_1[0](gra_atten[0] * x)
        for i in range(1, len(self.gra_linear_1)):
            moe_out_1 += self.gra_linear_1[i](gra_atten[i] * x)
        zero_tensor = torch.zeros_like(moe_out_1).to(x.device)
        for index in null_index:
            moe_out_1[index] = zero_tensor[index]

        # First Fourier transformation layer
        x_real_1 = F.relu(self.multiply(x.real, self.w1[0]) - self.multiply(x.imag, self.w1[1]) + self.b1[0])
        x_imag_1 = F.relu(self.multiply(x.real, self.w1[1]) + self.multiply(x.imag, self.w1[0]) + self.b1[1])
        x = torch.stack([x_real_1, x_imag_1], dim=-1).float()
        x = torch.view_as_complex(x)
        x = x + moe_out_1
        
        # Second expert mixture layer
        moe_out_2 = self.gra_linear_2[0](gra_atten[0] * x)
        for i in range(1, len(self.gra_linear_2)):
            moe_out_2 += self.gra_linear_2[i](gra_atten[i] * x)
        zero_tensor = torch.zeros_like(moe_out_2).to(x.device)
        for index in null_index:
            moe_out_2[index] = zero_tensor[index]

        # Second Fourier transformation layer
        x_real_2 = self.multiply(x.real, self.w2[0]) - self.multiply(x.imag, self.w2[1]) + self.b2[0]
        x_imag_2 = self.multiply(x.real, self.w2[1]) + self.multiply(x.imag, self.w2[0]) + self.b2[1]
        x = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        x = torch.view_as_complex(x)
        x = x + moe_out_2

        # Convert back to time domain
        x = torch.fft.irfft(x, dim=2, norm="ortho")
        return x

    def gra_attention(self, input, gra_feature):
        """
        Compute granularity attention weights.
        
        Args:
            input (torch.Tensor): Input tensor.
            gra_feature (torch.Tensor): Granularity feature tensor.
            
        Returns:
            torch.Tensor: Attention weights.
        """
        attention_scores = torch.einsum('ik,jk->ij', input, gra_feature)
        attention_weights = F.softmax(attention_scores, dim=-1)
        return attention_weights


class DecoderLayer(nn.Module):
    """
    Transformer decoder layer with cross-attention and Fourier filtering.
    
    This layer implements a transformer decoder with cross-attention mechanism
    and an adaptive Fourier neural operator for enhanced decoding capabilities.
    """
    
    def __init__(self, args, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        """
        Initialize the decoder layer.
        
        Args:
            args (object): Configuration arguments.
            cross_attention (nn.Module): Cross-attention mechanism.
            d_model (int): Dimension of the model embeddings.
            d_ff (int, optional): Dimension of the feed-forward network. Defaults to 4*d_model.
            dropout (float): Dropout probability. Defaults to 0.1.
            activation (str): Activation function ("relu" or "gelu"). Defaults to "relu".
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.cross_attention = cross_attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.filter = AdaptiveFourierNeuralOperator(args, d_model)

    def forward(self, x, cross, time_gra, x_mask=None, cross_mask=None):
        """
        Apply transformer decoder layer to input sequence.
        
        Args:
            x (torch.Tensor): Decoder input tensor.
            cross (torch.Tensor): Encoder output tensor for cross-attention.
            time_gra (list): Temporal granularity information.
            x_mask (torch.Tensor, optional): Decoder attention mask.
            cross_mask (torch.Tensor, optional): Cross-attention mask.
            
        Returns:
            torch.Tensor: Decoded tensor.
        """
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=x_mask
        )[0])
        y = self.norm2(x)
        y = self.filter(y, time_gra)
        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    Transformer decoder composed of multiple decoder layers.
    
    This module stacks multiple decoder layers and applies final normalization
    and projection if specified.
    """
    
    def __init__(self, layers, norm_layer=None, projection=None):
        """
        Initialize the decoder.
        
        Args:
            layers (list): List of decoder layer modules.
            norm_layer (nn.Module, optional): Normalization layer for final output.
            projection (nn.Module, optional): Projection layer for output transformation.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, time_gra, x_mask=None, cross_mask=None):
        """
        Apply transformer decoder to input sequence.
        
        Args:
            x (torch.Tensor): Decoder input tensor.
            cross (torch.Tensor): Encoder output tensor for cross-attention.
            time_gra (list): Temporal granularity information.
            x_mask (torch.Tensor, optional): Decoder attention mask.
            cross_mask (torch.Tensor, optional): Cross-attention mask.
            
        Returns:
            torch.Tensor: Final decoded output tensor.
        """
        for layer in self.layers:
            x = layer(x, cross, time_gra, x_mask=x_mask, cross_mask=cross_mask)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
