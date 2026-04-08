import os
import sys

import torch
from einops import rearrange
from torch import nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layers.Embed import PatchEmbedding
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import Decoder, DecoderLayer

# Constants
EPSILON = 1e-5

class Transpose(nn.Module):
    """
    A module that transposes specified dimensions of input tensors.
    
    This utility class provides a convenient way to transpose tensor dimensions
    with an optional contiguous memory layout enforcement.
    """
    def __init__(self, *dims, contiguous=False): 
        """
        Initialize the Transpose module.
        
        Args:
            *dims: Variable length argument list specifying the dimensions to transpose.
            contiguous (bool): If True, ensures the output tensor has contiguous memory layout.
        """
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
        
    def forward(self, x):
        """
        Forward pass that transposes the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to transpose.
            
        Returns:
            torch.Tensor: Transposed tensor.
        """
        if self.contiguous: 
            return x.transpose(*self.dims).contiguous()
        else: 
            return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    """
    A prediction head that flattens and linearly projects input features.
    
    This module is typically used in the final stage of a model to map
    high-dimensional features to the target output dimension.
    """
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        """
        Initialize the FlattenHead module.
        
        Args:
            n_vars (int): Number of variables in the input.
            nf (int): Number of input features.
            target_window (int): Target output window size.
            head_dropout (float): Dropout probability for regularization.
        """
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        """
        Forward pass of the FlattenHead.
        
        Args:
            x (torch.Tensor): Input tensor with shape [bs x nvars x d_model x patch_num].
            
        Returns:
            torch.Tensor: Output tensor after flattening, linear projection, and dropout.
        """
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(nn.Module):
    """
    GTM (Graph-based Time Series Model) for various time series forecasting tasks.
    
    This model implements a patch-based transformer architecture for time series analysis,
    supporting multiple tasks including long-term forecasting, imputation, anomaly detection,
    and pre-training. The architecture follows the paper: https://arxiv.org/pdf/2211.14730.pdf
    
    Attributes:
        task_name (str): Name of the task (e.g., 'long_term_forecast', 'imputation').
        seq_len (int): Input sequence length.
        pred_len (int): Prediction sequence length.
        patch_len (int): Length of each patch.
        stride (int): Stride for patching.
        patch_embedding (PatchEmbedding): Module for patch embedding.
        decoder (Decoder): Transformer decoder module.
        head (nn.Linear): Prediction head for output projection.
    """

    def __init__(self, configs):
        """
        Initialize the GTM Model.
        
        Args:
            configs (object): Configuration object containing model hyperparameters.
                Expected attributes:
                - task_name (str): Type of task to perform.
                - seq_len (int): Input sequence length.
                - pred_len (int): Prediction length.
                - patch_len (int): Patch length for embedding.
                - stride (int): Stride for patching.
                - d_model (int): Model dimension.
                - dropout (float): Dropout probability.
                - d_layers (int): Number of decoder layers.
                - factor (int): Factor for attention mechanism.
                - n_heads (int): Number of attention heads.
                - d_ff (int): Dimension of feed-forward networks.
                - activation (str): Activation function.
                - enc_in (int): Number of input channels.
        """
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        padding = 0
        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs,configs.d_model, self.patch_len, self.stride, padding, configs.dropout)
        self.flatten_pred = nn.Flatten(start_dim=-2)
        self.device = configs.device
        self.decoder = Decoder(
            [
                DecoderLayer(
                    configs,
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Prediction Head
        self.head_nf = configs.d_model * (self.seq_len//self.patch_len)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            self.head = nn.Linear(configs.d_model, self.patch_len)
        if self.task_name == 'imputation':
            self.head = nn.Linear(configs.d_model, configs.enc_in)
        elif 'pre_train' in self.task_name:
            self.head = nn.Linear(configs.d_model, self.patch_len)

    def forecast(self, x, time_gra):
        """
        Perform long-term forecasting on input time series data.
        
        This method applies normalization, patch embedding, decoding, and denormalization
        to generate future predictions for time series data.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C] where B is batch size,
                             L is sequence length, and C is number of channels/variables.
            time_gra (torch.Tensor): Time granularity information for the decoder.
            
        Returns:
            torch.Tensor: Forecasted values of shape [B, pred_len, C].
        """
        B, L, C = x.size()
        # Normalize input data using mean and standard deviation
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + EPSILON)
        x /= stdev
        
        # Reorder dimensions for patching: [B, C, L]
        x = x.permute(0, 2, 1)
        
        # Extend sequence length if prediction length exceeds input length
        if self.pred_len > self.seq_len:
            pred_mask_len = ((self.pred_len-self.seq_len)// self.patch_len+1) * self.patch_len if self.pred_len % self.patch_len != 0 else self.pred_len - self.seq_len
            # Check if pred_mask_len is greater than 0
            if pred_mask_len > 0:
                # Create prediction mask directly on target device
                pred_mask = torch.zeros(B, C, pred_mask_len, device=x.device)
                x = torch.concat((x, pred_mask), dim=2)
                
        # Apply patch embedding to convert time series to patch representations
        # dec_in: [bs * nvars x patch_num x d_model]
        dec_in, n_vars = self.patch_embedding(x)

        # Decode the embedded patches using transformer decoder
        # dec_out: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=None, cross_mask=None)

        # Reshape decoder output to original batch structure
        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))
            
        # Apply prediction head and flatten
        dec_out = self.head(dec_out)
        dec_out = self.flatten_pred(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Denormalize to restore original scale
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1), 1))
        dec_out = dec_out + \
                  (means.repeat(1, dec_out.size(1), 1))
                  
        # Return only the prediction length portion
        return dec_out[:, -self.pred_len:, :]

    def imputation(self, x_enc, time_gra, mask):
        """
        Perform imputation on time series data with missing values.
        
        This method handles missing data by applying masked normalization,
        patch embedding, decoding, and denormalization to reconstruct missing values.
        
        Args:
            x_enc (torch.Tensor): Input tensor with missing values of shape [B, L, C].
            time_gra (torch.Tensor): Time granularity information for the decoder.
            mask (torch.Tensor): Binary mask indicating observed (1) and missing (0) values.
            
        Returns:
            torch.Tensor: Imputed time series data of shape [B, L, C].
        """
        B, L, C = x_enc.size()
        
        # Normalize input data using masked statistics
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        
        # Compute standard deviation using masked data
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + EPSILON)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # Apply patch embedding to convert time series to patch representations
        dec_in, n_vars = self.patch_embedding(x_enc)

        # Decode the embedded patches using transformer decoder
        # dec_out: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=None, cross_mask=None)
        
        # Apply prediction head for output projection
        dec_out = self.head(dec_out)
        
        # Denormalize to restore original scale
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1), 1))
        dec_out = dec_out + \
                  (means.repeat(1, dec_out.size(1), 1))
                  
        return dec_out

    def anomaly_detection(self, x_enc, time_gra):
        """
        Perform anomaly detection on time series data.
        
        This method applies normalization, patch embedding, decoding, and denormalization
        to reconstruct input data for anomaly detection by comparing reconstruction errors.
        
        Args:
            x_enc (torch.Tensor): Input tensor of shape [B, L, C] where B is batch size,
                                 L is sequence length, and C is number of channels/variables.
            time_gra (torch.Tensor): Time granularity information for the decoder.
            
        Returns:
            torch.Tensor: Reconstructed time series data of shape [B, L, C] for anomaly detection.
        """
        B, L, C = x_enc.size()
        
        # Normalize input data using mean and standard deviation
        means = x_enc.mean(1, keepdim=True).detach()
        x = x_enc - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + EPSILON)
        x /= stdev
        
        # Reorder dimensions for patching: [B, C, L]
        x = x.permute(0, 2, 1)
        
        # Apply patch embedding to convert time series to patch representations
        dec_in, n_vars = self.patch_embedding(x)

        # Decode the embedded patches using transformer decoder
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=None, cross_mask=None)

        # Apply prediction head for output projection
        dec_out = self.head(dec_out)
        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))

        # Flatten and reorder dimensions
        dec_out = self.flatten_pred(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Denormalize to restore original scale
        dec_out = dec_out * \
                  (stdev.repeat(1, dec_out.size(1), 1))
        dec_out = dec_out + \
                  (means.repeat(1, dec_out.size(1), 1))
                  
        return dec_out

    def pre_train(self, x, time_gra):
        """
        Perform pre-training with masked patch prediction.
        
        This method applies normalization, masked patch embedding, decoding, and denormalization
        for self-supervised pre-training of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, L, C] where B is batch size,
                             L is sequence length, and C is number of channels/variables.
            time_gra (torch.Tensor): Time granularity information for the decoder.
            
        Returns:
            tuple: (dec_out, x_patch) where:
                - dec_out (torch.Tensor): Predicted patches of shape [B, patch_num, patch_len, C].
                - x_patch (torch.Tensor): Original patches of shape [B, patch_num, patch_len, C].
        """
        B, L, C = x.size()
        
        # Normalize input data using mean and standard deviation
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(
            torch.var(x, dim=1, keepdim=True, unbiased=False) + EPSILON)
        x /= stdev

        # Reorder dimensions for patching: [B, C, L]
        x = x.permute(0, 2, 1)
        
        # Apply masked patch embedding
        # dec_in: [bs * nvars x patch_num x d_model]
        # x_patch: original patches for comparison
        # atten_mask: attention mask for masked training
        # mask_id: mask identifiers
        # end_token_indices: indices of end tokens
        dec_in, x_patch, atten_mask, mask_id, end_token_indices, pos_1, pos_2 = self.patch_embedding(x, mask=True)

        # Decode the embedded patches using transformer decoder with attention masking
        # dec_out: [bs * nvars x patch_num x d_model]
        dec_out = self.decoder(dec_in, dec_in, time_gra, x_mask=atten_mask, cross_mask=None)
        
        # Reshape decoder output to original batch structure
        # dec_out: [bs x nvars x patch_num x d_model]
        dec_out = torch.reshape(
            dec_out, (B, C, dec_out.shape[-2], dec_out.shape[-1]))
            
        # Apply prediction head for output projection
        dec_out = self.head(dec_out)  # dec_out: [bs x nvars x patch_num x patch_len]

        # Mask out end_token positions to exclude them from loss calculation
        mask_index = torch.where(end_token_indices, False, mask_id)
        mask_index = torch.reshape(mask_index, (B, C, dec_out.shape[-2]))
        mask_index = mask_index.unsqueeze(-1).expand(-1, -1, -1, self.patch_len)
        dec_out = dec_out * mask_index
        x_patch = torch.reshape(
            x_patch, (B, C, x_patch.shape[-2], x_patch.shape[-1]))
        x_patch = x_patch * mask_index
        
        # Reorder dimensions for output
        x_patch = x_patch.permute(0, 2, 3, 1)
        dec_out = dec_out.permute(0, 2, 3, 1)
        
        # Denormalize to restore original scale
        x_patch = x_patch * \
                  (stdev.unsqueeze(1).repeat(1, dec_in.shape[1], self.patch_len, 1))
        x_patch = x_patch + \
                  (means.unsqueeze(1).repeat(1, dec_in.shape[1], self.patch_len, 1))
                  
        dec_out = dec_out * \
                  (stdev.unsqueeze(1).repeat(1, dec_in.shape[1], self.patch_len, 1))
        dec_out = dec_out + \
                  (means.unsqueeze(1).repeat(1, dec_in.shape[1], self.patch_len, 1))

        return dec_out, x_patch

    def forward(self, x_enc, time_gra, mask=None):
        """
        Forward pass of the GTM model, routing input to appropriate task-specific method.
        
        This method serves as the entry point for all tasks, dispatching the input
        to the corresponding task-specific processing method based on task_name.
        
        Args:
            x_enc (torch.Tensor): Input tensor of shape [B, L, C] for most tasks,
                                 where B is batch size, L is sequence length, and C is channels.
            time_gra (torch.Tensor): Time granularity information for the decoder.
            mask (torch.Tensor, optional): Binary mask for imputation task. Defaults to None.
            
        Returns:
            torch.Tensor or tuple: Task-specific output:
                - For forecasting tasks: Tensor of shape [B, pred_len, C]
                - For imputation task: Tensor of shape [B, L, C]
                - For anomaly detection task: Tensor of shape [B, L, C]
                - For pre-training task: Tuple of (predicted_patches, original_patches)
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # Perform forecasting for long-term or short-term prediction tasks
            dec_out = self.forecast(x_enc, time_gra)
            return dec_out  # [B, pred_len, C]
        if self.task_name == 'imputation':
            # Perform imputation for missing value reconstruction
            dec_out = self.imputation(x_enc, time_gra, mask)
            return dec_out  # [B, L, C]
        if self.task_name == 'anomaly_detection':
            # Perform anomaly detection through reconstruction
            dec_out = self.anomaly_detection(x_enc, time_gra)
            return dec_out  # [B, L, C]
        if 'pre_train' in self.task_name:
            # Perform pre-training with masked patch prediction
            dec_out, x = self.pre_train(x_enc, time_gra)
            return dec_out, x  # [B, patch_num, patch_len, C], [B, patch_num, patch_len, C]
