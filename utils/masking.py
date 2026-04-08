"""
Masking utilities for time series data preprocessing.

This module provides various masking strategies for time series data,
including causal masking, probabilistic masking, and specialized
masking processors for different transformer architectures.
"""

import torch
import numpy as np


class TriangularCausalMask:
    """
    Triangular causal mask for autoregressive attention mechanisms.
    
    This mask ensures that each position in the sequence can only attend
    to previous positions, preventing information leakage from future tokens.
    """
    
    def __init__(self, B, L, device="cpu"):
        """
        Initialize the triangular causal mask.
        
        Args:
            B (int): Batch size.
            L (int): Sequence length.
            device (str): Device to place the mask tensor on. Defaults to "cpu".
        """
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        """
        Get the causal mask tensor.
        
        Returns:
            torch.Tensor: Upper triangular mask tensor of shape [B, 1, L, L].
        """
        return self._mask


class ProbMask:
    """
    Probabilistic mask for sparse attention mechanisms.
    
    This mask is used in sparse transformer variants to limit attention
    to a subset of positions based on learned probabilities or heuristics.
    """
    
    def __init__(self, B, H, L, index, scores, device="cpu"):
        """
        Initialize the probabilistic mask.
        
        Args:
            B (int): Batch size.
            H (int): Number of attention heads.
            L (int): Sequence length.
            index (torch.Tensor): Index tensor for selecting positions.
            scores (torch.Tensor): Attention scores for probability calculation.
            device (str): Device to place the mask tensor on. Defaults to "cpu".
        """
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        """
        Get the probabilistic mask tensor.
        
        Returns:
            torch.Tensor: Probabilistic mask tensor.
        """
        return self._mask


class BaseMaskProcessor:
    """
    Base class for time series masking processors.
    
    This class provides fundamental masking functionality for time series data,
    including random masking of patches and sequence padding utilities.
    
    Examples:
        >>> d = [p1, p2, p3, p4, p5, p6]
        >>> masked_d = [p1, <M>, p3, <M>, p5, <M>]
        >>> mask_ids = [0, 1, 0, 1, 0, 1]
        >>> attention_masks = [[
        >>>     1, 1, 1, 1, 1, 1,
        >>>     1, 1, 1, 1, 1, 1,
        >>>     1, 1, 1, 1, 1, 1,
        >>>     1, 1, 1, 1, 1, 1,
        >>>     1, 1, 1, 1, 1, 1,
        >>>     1, 1, 1, 1, 1, 1,
        >>> ]]
        >>> position_ids = [1,2,3,4,5,6]
    """

    def __init__(self, mask_ratio: float = 0.15):
        """
        Initialize the base mask processor.
        
        Args:
            mask_ratio (float): Proportion of tokens to mask. Defaults to 0.15.
        """
        self.mask_ratio = mask_ratio

    def _mask_one_sample(self, sample: np.ndarray, seq_len: int = None):
        """
        Apply random masking to a single time series sample.
        
        Args:
            sample (np.ndarray): Input time series sample of shape [L, D].
            seq_len (int, optional): Actual sequence length (for padded sequences).
            
        Returns:
            tuple: (masked_sample, mask_ids, attention_mask) where:
                - masked_sample: Sample with masked tokens
                - mask_ids: Binary mask indicating masked positions
                - attention_mask: Attention mask for the sample
        """
        L, D = sample.shape
        attention_mask = np.ones((L, L), dtype=int)
        
        # Handle padded sequences
        if not seq_len:
            seq_len = L
        attention_mask[seq_len:, :] = 0
        attention_mask[:, seq_len:] = 0
        
        # Calculate number of tokens to keep
        len_keep = int((1 - self.mask_ratio) * seq_len)
        noise = torch.rand(seq_len)
        
        # Sort noise for consistent masking
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = np.argsort(ids_shuffle)  # ids_restore: [L]

        # Separate kept and removed tokens
        ids_keep = ids_shuffle[:len_keep]
        x_kept = sample[ids_keep]
        x_removed = np.zeros((L - len_keep, D), dtype=sample.dtype)
        sample_masked = np.concatenate([x_kept, x_removed], axis=0)
        sample_masked[:seq_len] = sample_masked[:seq_len][ids_restore]

        # Generate mask IDs
        mask_id = np.zeros(L, dtype=int)
        mask_id[len_keep:seq_len] = 1
        mask_id[:seq_len] = mask_id[:seq_len][ids_restore]
        
        return sample_masked, mask_id, attention_mask

    def mask_batch(self, batch, seq_lengths=None):
        """
        Apply random masking to a batch of time series samples.
        
        Args:
            batch (list): List of arrays, each with shape [L_i, D].
            seq_lengths (list or np.ndarray, optional): Actual lengths of sequences.
            
        Returns:
            tuple: (batch_masked, mask_ids, attention_masks) where each is a list
                   with elements corresponding to each sample in the batch.
        """
        B = len(batch)
        batch_masked, mask_ids, attention_masks = [], [], []
        
        for i, sample in enumerate(batch):
            seq_len = seq_lengths[i] if seq_lengths is not None else None
            sample_masked, mask_id, attention_mask = self._mask_one_sample(sample, seq_len)
            batch_masked.append(sample_masked)
            mask_ids.append(mask_id)
            attention_masks.append(attention_mask)
            
        return batch_masked, mask_ids, attention_masks

    def get_position_ids(self, padded_mask_ids):
        """
        Generate position IDs for masked sequences.
        
        Args:
            padded_mask_ids (torch.Tensor): Padded mask IDs of shape [B, L].
            
        Returns:
            dict: Dictionary containing position IDs tensor.
        """
        B, L = padded_mask_ids.shape
        pos_ids = torch.arange(1, L + 1).unsqueeze_(0).repeat(B, 1)
        return dict(position_ids=pos_ids)

    @staticmethod
    def pad_sequences(sequences, mask_ids=None, attention_masks=None, max_len=None, pad_value=0.0):
        """
        Pad a batch of sequences to the same length.
        
        Args:
            sequences (list): List of sequence arrays.
            mask_ids (list, optional): List of mask ID arrays.
            attention_masks (list, optional): List of attention mask arrays.
            max_len (int, optional): Maximum length for padding.
            pad_value (float): Value to use for padding.
            
        Returns:
            tuple: (padded_data, nonpadding_mask) where:
                - padded_data: Tuple of (padded_sequences, padded_mask_ids, padded_attention_masks)
                - nonpadding_mask: Boolean mask indicating real vs padded tokens
        """
        batch_size = len(sequences)
        patch_len = sequences[0].shape[-1]
        max_seq_len = (max(seq.shape[0] for seq in sequences))
        max_len = max_seq_len if max_len is None else min(max_seq_len, max_len)

        # Initialize padded arrays
        padded_sequences = np.full((batch_size, max_len, patch_len), pad_value)
        nonpadding_mask = np.zeros((batch_size, max_len), dtype=int)
        
        if mask_ids is not None:
            padded_mask_ids = np.full((batch_size, max_seq_len), 0, dtype=int)
        else:
            padded_mask_ids = None

        if attention_masks is not None:
            padded_attention_masks = np.full((batch_size, max_len, max_len), 0, dtype=int)
        else:
            padded_attention_masks = None

        # Fill padded arrays
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            _len = min(max_len, seq_len)
            
            if mask_ids is not None:
                mask_id = mask_ids[i][:_len]
                padded_mask_ids[i, :_len] = mask_id
                
            if attention_masks is not None:
                _len = min(max_len, seq_len)
                attention_mask = attention_masks[i][:_len, :_len]
                padded_attention_masks[i, :_len, :_len] = attention_mask
                
            padded_sequences[i, :_len] = seq[:max_len]
            nonpadding_mask[i, :_len] = 1

        return (padded_sequences, padded_mask_ids, padded_attention_masks), nonpadding_mask

    def __call__(self, batch, mask_first=False):
        """
        Apply masking to a batch of sequences.
        
        Args:
            batch (torch.Tensor): Input batch of shape [B, L, D].
            mask_first (bool): Whether to mask before padding.
            
        Returns:
            dict: Dictionary containing masked inputs, labels, and masks.
        """
        B, L, D = batch.size()
        
        if mask_first:
            # Mask first, then pad
            masked_batch, mask_ids, attention_masks = self.mask_batch(batch)
            (padded_batch, _, _), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=int(L * (self.mask_ratio + 1)),
                pad_value=0.0
            )
            (padded_masked_batch, padded_mask_ids, attention_masks), _nonpadding_mask = self.pad_sequences(
                sequences=masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=int(L * (self.mask_ratio + 1)),
                pad_value=0.0
            )
            assert (nonpadding_mask - _nonpadding_mask).sum() == 0
            attention_masks = nonpadding_mask[:, :, None] * attention_masks
        else:
            # Pad first, then mask
            (padded_batch, _, __), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=int(L * (self.mask_ratio + 1)),
                pad_value=0.0
            )
            seq_lengths = nonpadding_mask.sum(-1)
            padded_masked_batch, padded_mask_ids, attention_masks = self.mask_batch(padded_batch,
                                                                                    seq_lengths=seq_lengths)
            padded_masked_batch = np.stack(padded_masked_batch, axis=0)
            padded_mask_ids = np.stack(padded_mask_ids, axis=0)
            attention_masks = np.stack(attention_masks, axis=0)
            
        position_id_dict = self.get_position_ids(padded_mask_ids)

        data_dict = dict(
            input_patches=torch.from_numpy(padded_masked_batch),  # [B, L, D]
            labels=torch.from_numpy(padded_batch),  # [B, L, D]
            mask_ids=torch.from_numpy(padded_mask_ids),  # [B, L]
            attention_masks=torch.from_numpy(attention_masks),  # [B, L, L]
        )
        data_dict.update(position_id_dict)
        return data_dict


class GLMMaskProcessor4TS(BaseMaskProcessor):
    """
    GLM-style masking processor for time series data.
    
    This processor implements a GLM-like masking strategy where masked tokens
    are moved to the end of the sequence, enabling bidirectional reconstruction.
    
    Examples:
        >>> d = [p1, p2, p3, p4, p5, p6]
        >>> masked_d = [p1, <M>, p4, <M>, p5, p6, p2, p3]
        >>> mask_ids = [0, 0, 0, 0, 1, 1, 1, 1]
        >>> attention_masks = [[
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 1, 0, 0, 0,
        >>>     1, 1, 1, 1, 1, 1, 0, 0,
        >>>     1, 1, 1, 1, 1, 1, 1, 0,
        >>>     1, 1, 1, 1, 1, 1, 1, 1,
        >>> ]]
        >>> position_ids_1 = [1,2,3,4,4,4,2,2]
        >>> position_ids_2 = [0,0,0,0,1,2,1,2]
    """

    def __init__(self, mask_ratio: float = 0.15, possion_lambda: int = 3):
        """
        Initialize the GLM mask processor.
        
        Args:
            mask_ratio (float): Proportion of tokens to mask. Defaults to 0.15.
            possion_lambda (int): Lambda parameter for Poisson distribution. Defaults to 3.
        """
        self.mask_ratio = mask_ratio
        self.possion_lambda = possion_lambda

    @staticmethod
    def _detect_overlap(start, end, spans_list):
        """
        Check if a span overlaps with existing spans.
        
        Args:
            start (int): Start position of the new span.
            end (int): End position of the new span.
            spans_list (list): List of existing spans.
            
        Returns:
            bool: True if overlap detected, False otherwise.
        """
        for span_s, span_e in spans_list:
            if (span_s <= start <= span_e) or (span_s <= end <= span_e):
                return True
        return False

    def _mask_one_sample(self, sample: np.ndarray, seq_len: int = None):
        """
        Apply GLM-style masking to a single time series sample.
        
        Args:
            sample (np.ndarray): Input time series sample of shape [L, D].
            seq_len (int, optional): Actual sequence length (for padded sequences).
            
        Returns:
            tuple: (masked_sample, mask_ids, attention_mask) with GLM-style arrangement.
        """
        L, D = sample.shape
        
        # Handle padded sequences
        if not seq_len:
            seq_len = L
            
        # Calculate total tokens to mask
        total_to_mask = int(self.mask_ratio * seq_len)
        mask_count = 0
        spans_list = []

        # Generate non-overlapping spans to mask
        while mask_count < total_to_mask:
            span_length = np.random.poisson(lam=self.possion_lambda)
            if span_length == 0 or span_length >= seq_len:
                continue  # Skip zero-length spans

            start_pos = np.random.randint(0, seq_len - span_length)
            end_pos = min(start_pos + span_length, seq_len)

            # Ensure no overlap with existing spans
            if self._detect_overlap(start_pos, end_pos, spans_list):
                continue
                
            new_mask_count = mask_count + (end_pos - start_pos)
            if new_mask_count > total_to_mask:
                continue  # Skip if would exceed mask ratio
                
            # Add span to mask
            spans_list.append((start_pos, end_pos))
            mask_count += end_pos - start_pos
            
        # Calculate partition lengths
        all_span_len = sum([end_pos - start_pos for start_pos, end_pos in spans_list])
        part_A_len = (seq_len + len(spans_list)) - all_span_len  # Known sequence + mask tokens
        part_B_len = all_span_len  # Masked tokens
        new_L = part_A_len + part_B_len
        
        # Initialize partition arrays
        masked_sample_A = np.zeros((part_A_len, D), dtype=sample.dtype)
        masked_sample_B = np.zeros((part_B_len, D), dtype=sample.dtype)

        # Initialize masks
        attention_mask = np.ones((new_L, new_L), dtype=int)
        mask_id = np.zeros(new_L, dtype=int)

        # Process spans and build masked sequence
        last_end = 0
        cur_j_A = 0
        cur_j_B = 0
        position_id_1_dict = {}

        # Process known sequence part (A)
        for i, (start_pos, end_pos) in enumerate(sorted(spans_list)):
            # Copy unmasked tokens before this span
            masked_sample_A[cur_j_A:cur_j_A + (start_pos - last_end)] = sample[last_end:start_pos]
            cur_j_A += start_pos - last_end
            
            # Add mask token
            mask_id[cur_j_A] = -(cur_j_A + 1)
            position_id_1_dict[(start_pos, end_pos)] = cur_j_A + 1
            cur_j_A += 1
            
            last_end = end_pos
            
        # Copy remaining unmasked tokens
        masked_sample_A[cur_j_A:] = sample[last_end:]
        attention_mask[:, len(masked_sample_A):] = 0

        # Process masked sequence part (B)
        for i, (start_pos, end_pos) in enumerate(spans_list):
            # Copy masked tokens to B partition
            masked_sample_B[cur_j_B:cur_j_B + (end_pos - start_pos)] = sample[start_pos:end_pos]
            
            # Set position IDs for masked tokens
            _s, _e = part_A_len + cur_j_B, part_A_len + cur_j_B + (end_pos - start_pos)
            mask_id[_s: _e] = position_id_1_dict[(start_pos, end_pos)]
            cur_j_B += (end_pos - start_pos)

        # Set causal attention mask for B partition
        attention_mask[-mask_count:, -mask_count:] = np.tril(
            np.ones((mask_count, mask_count), dtype=int),
        )
        
        # Concatenate partitions
        masked_sample = np.concatenate((masked_sample_A, masked_sample_B), axis=0)

        return masked_sample, mask_id, attention_mask

    def _get_position_id2(self, tensor):
        """
        Generate secondary position IDs for consecutive tokens.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Secondary position IDs.
        """
        unique, indexes, counts = torch.unique_consecutive(tensor, return_counts=True, return_inverse=True)
        counts = torch.cat([torch.zeros(1, dtype=counts.dtype), counts], dim=0)
        counts_cumsum = counts.cumsum_(dim=0)
        pos_id = torch.arange(1, (tensor.size(0) + 1)) - counts_cumsum[indexes]
        return pos_id

    def get_position_ids(self, padded_mask_ids, nonpadding_mask):
        """
        Generate GLM-style position IDs.
        
        Args:
            padded_mask_ids (np.ndarray or torch.Tensor): Padded mask IDs.
            nonpadding_mask (np.ndarray or torch.Tensor): Padding mask.
            
        Returns:
            dict: Dictionary containing position IDs tensors.
        """
        # Convert to torch tensors if needed
        if isinstance(padded_mask_ids, np.ndarray):
            padded_mask_ids = torch.from_numpy(padded_mask_ids)
        if isinstance(nonpadding_mask, np.ndarray):
            nonpadding_mask = torch.from_numpy(nonpadding_mask)
            
        B, L = padded_mask_ids.shape
        
        # Generate primary position IDs
        position_id_1 = torch.arange(1, L + 1, dtype=torch.long).unsqueeze_(0).repeat(B, 1)
        position_id_2 = torch.zeros((B, L), dtype=torch.long)
        
        # Set position IDs for masked tokens
        position_id_1[padded_mask_ids > 0] = padded_mask_ids[padded_mask_ids > 0]
        position_id_1[padded_mask_ids < 0] = - padded_mask_ids[padded_mask_ids < 0]
        position_id_1[nonpadding_mask == 0] = 0
        
        # Generate secondary position IDs
        for b in range(B):
            padded_mask_id = padded_mask_ids[b]
            position_id_2[b][padded_mask_id > 0] = self._get_position_id2(padded_mask_id[padded_mask_id > 0])

        # Convert mask IDs to boolean
        padded_mask_ids[padded_mask_ids < 0] = 0
        padded_mask_ids[padded_mask_ids > 0] = 1
        padded_mask_ids = padded_mask_ids.type(torch.bool)
        
        return dict(
            position_ids_1=position_id_1,
            position_ids_2=position_id_2,
            mask_ids=padded_mask_ids
        )

    def __call__(self, batch, mask_first=True, to_device=True):
        """
        Apply GLM-style masking to a batch of sequences.
        
        Args:
            batch (torch.Tensor): Input batch of shape [B, L, D].
            mask_first (bool): Whether to mask before padding.
            to_device (bool): Whether to move results to the input device.
            
        Returns:
            dict: Dictionary containing masked inputs, labels, and masks.
        """
        device = batch.device
        
        if not mask_first:
            # Pad first, then mask
            B, L, D = batch.size()
            batch = [batch[i].detach().cpu().numpy() for i in range(batch.size(0))]
            (padded_batch, _, __), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=int(L * (self.mask_ratio + 1)),
                pad_value=0.0
            )
            seq_lengths = nonpadding_mask.sum(-1)
            masked_batch, mask_ids, attention_masks = self.mask_batch(batch, seq_lengths=seq_lengths)
            
            # GLM masking mechanism would cause variable lengths for a padded batch
            (padded_masked_batch, padded_mask_ids, attention_masks), nonpadding_mask = self.pad_sequences(
                masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=None
            )
            attention_masks = nonpadding_mask[:, :, None] * attention_masks
        else:
            # Mask first, then pad
            batch = [batch[i].detach().cpu().numpy() for i in range(batch.size(0))]
            masked_batch, mask_ids, attention_masks = self.mask_batch(batch)
            (padded_masked_batch, padded_mask_ids, attention_masks), nonpadding_mask = self.pad_sequences(
                sequences=masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=None,
                pad_value=0.0
            )
            
        position_id_dict = self.get_position_ids(padded_mask_ids, nonpadding_mask=nonpadding_mask)

        data_dict = dict(
            input_patches=torch.from_numpy(padded_masked_batch.astype(np.float32)),  # [B, L, D]
            labels=torch.from_numpy(padded_masked_batch.astype(np.float32)),  # [B, L, D]
            mask_ids=torch.from_numpy(padded_mask_ids.astype(bool)),  # [B, L]
            attention_masks=torch.from_numpy(attention_masks.astype(bool)),  # [B, L, L]
        )
        data_dict.update(position_id_dict)

        if to_device:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}

        return data_dict


class GLMMaskProcessor4TSGPU(BaseMaskProcessor):
    """
    GPU-optimized GLM-style masking processor for time series data.
    
    This processor implements a GLM-like masking strategy with start/end tokens
    for improved reconstruction capabilities on GPU devices.
    
    Examples:
        >>> d = [p1, p2, p3, p4, p5, p6]
        >>> masked_input = [p1, <M>, p4, <M>, <s>, p5, p6, <s>, p2, p3]
        >>> masked_output = [p1, <M>, p4, <M>, p5, p6, <e>, p2, p3, <e>]
        >>> # Indices for replacing learnable tokens
        >>> start_token_indices = [False, False, False, False, True, False, False, True, False, False]
        >>> end_token_indices = [False, False, False, False, False, False, True, False, False, True]
        >>> mask_token_indices = [False, True, False, True, False, False, False, False, False, False]
        >>> mask_ids = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        >>> attention_masks = [[
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 0, 0, 0, 0,
        >>>     1, 1, 1, 1, 1, 0, 0, 0,
        >>>     1, 1, 1, 1, 1, 1, 0, 0,
        >>>     1, 1, 1, 1, 1, 1, 1, 0,
        >>>     1, 1, 1, 1, 1, 1, 1, 1,
        >>> ]]# Attention masks with start tokens
        >>> position_ids_1 = [1,2,3,4,4,4,2,2]
        >>> position_ids_2 = [0,0,0,0,1,2,1,2]
    """

    def __init__(self, mask_ratio: float = 0.15, possion_lambda: int = 3):
        """
        Initialize the GPU-optimized GLM mask processor.
        
        Args:
            mask_ratio (float): Proportion of tokens to mask. Defaults to 0.15.
            possion_lambda (int): Lambda parameter for Poisson distribution. Defaults to 3.
        """
        self.mask_ratio = mask_ratio
        self.possion_lambda = possion_lambda

    @staticmethod
    def _detect_overlap(start, end, spans_list):
        """
        Check if a span overlaps with existing spans.
        
        Args:
            start (int): Start position of the new span.
            end (int): End position of the new span.
            spans_list (list): List of existing spans.
            
        Returns:
            bool: True if overlap detected, False otherwise.
        """
        for span_s, span_e in spans_list:
            if (span_s <= end and span_e >= start):  # Check for overlap
                return True
        return False

    def _mask_one_sample(self, sample: torch.Tensor, pred_ratio: torch.float, seq_len: int = None, add_start_end_token: bool = False):
        """
        Apply GPU-optimized GLM-style masking to a single time series sample.
        
        Args:
            sample (torch.Tensor): Input time series sample of shape [L, D].
            pred_ratio (torch.float): Prediction ratio for special masking.
            seq_len (int, optional): Actual sequence length (for padded sequences).
            add_start_end_token (bool): Whether to add start/end tokens.
            
        Returns:
            tuple: (masked_input, masked_output, mask_ids, attention_mask, start_indices, end_indices)
        """
        L, D = sample.shape
        
        # Handle padded sequences
        if not seq_len:
            seq_len = L
            
        # Calculate total tokens to mask
        total_to_mask = int(self.mask_ratio * seq_len)
        mask_count = 0
        spans_list = []
        
        # Special handling for prediction ratio
        if np.random.random() <= pred_ratio:
            spans_list = [[11, 15]]
        else:
            # Generate non-overlapping spans to mask
            while mask_count < total_to_mask:
                span_length = np.random.poisson(lam=self.possion_lambda)
                if span_length == 0 or span_length >= seq_len:
                    continue  # Skip zero-length spans

                start_pos = np.random.randint(0, seq_len - span_length)
                end_pos = min(start_pos + span_length, seq_len)

                # Ensure no overlap with existing spans
                if self._detect_overlap(start_pos, end_pos, spans_list):
                    continue
                    
                new_mask_count = mask_count + (end_pos - start_pos)
                if new_mask_count > total_to_mask:
                    continue  # Skip if would exceed mask ratio
                    
                # Add span to mask
                spans_list.append((start_pos, end_pos))
                mask_count += end_pos - start_pos
                
        # Calculate partition lengths
        all_span_len = sum([end_pos - start_pos for start_pos, end_pos in spans_list])
        part_A_len = (seq_len + len(spans_list)) - all_span_len  # Known sequence + mask tokens
        part_B_len = all_span_len  # Masked tokens
        
        if add_start_end_token:
            part_B_len += len(spans_list)  # Additional tokens for start/end
            
        new_L = part_A_len + part_B_len
        
        # Initialize partition tensors on the same device
        masked_sample_A = torch.zeros((part_A_len, D), dtype=sample.dtype, device=sample.device)
        masked_sample_B_in = torch.zeros((part_B_len, D), dtype=sample.dtype, device=sample.device)
        masked_sample_B_out = torch.zeros((part_B_len, D), dtype=sample.dtype, device=sample.device)

        # Initialize token indices lists
        start_token_indices = []
        end_token_indices = []

        # Initialize masks
        attention_mask = torch.ones((new_L, new_L), dtype=torch.bool)
        mask_id = torch.zeros(new_L, dtype=torch.int32)
        mask_token_indices = []

        # Process spans and build masked sequence
        last_end = 0
        cur_j_A = 0
        cur_j_B_in = 0
        cur_j_B_out = 0
        position_id_1_dict = {}

        # Process known sequence part (A)
        for i, (start_pos, end_pos) in enumerate(sorted(spans_list)):
            # Copy unmasked tokens before this span
            masked_sample_A[cur_j_A:cur_j_A + (start_pos - last_end)] = sample[last_end:start_pos]
            cur_j_A += start_pos - last_end
            
            # Add mask token
            mask_token_indices.append(cur_j_A)
            mask_id[cur_j_A] = -(cur_j_A + 1)
            position_id_1_dict[(start_pos, end_pos)] = cur_j_A + 1
            cur_j_A += 1
            
            last_end = end_pos
            
        # Copy remaining unmasked tokens
        masked_sample_A[cur_j_A:] = sample[last_end:]
        attention_mask[:, len(masked_sample_A):] = 0

        # Process masked sequence part (B)
        for i, (start_pos, end_pos) in enumerate(spans_list):
            # Add start token if requested
            if add_start_end_token:
                start_token_indices.append(cur_j_B_in + part_A_len)
                cur_j_B_in += 1  # Add start token
                
            # Copy masked tokens to B partitions
            masked_sample_B_in[cur_j_B_in:cur_j_B_in + (end_pos - start_pos)] = sample[start_pos:end_pos]
            masked_sample_B_out[cur_j_B_out:cur_j_B_out + (end_pos - start_pos)] = sample[start_pos:end_pos]
            
            # Set position IDs for masked tokens
            _s, _e = part_A_len + cur_j_B_in, part_A_len + cur_j_B_in + (end_pos - start_pos)
            mask_id[_s - 1: _e] = position_id_1_dict[(start_pos, end_pos)]
            cur_j_B_in += (end_pos - start_pos)
            cur_j_B_out += (end_pos - start_pos)
            
            # Add end token if requested
            if add_start_end_token:
                end_token_indices.append(cur_j_B_out + part_A_len)
                cur_j_B_out += 1  # Add end token

        # Set causal attention mask for B partition
        attention_mask[-masked_sample_B_in.size(0):, -masked_sample_B_in.size(0):] = torch.tril(
            torch.ones((masked_sample_B_in.size(0), masked_sample_B_in.size(0)), dtype=torch.bool),
        )
        
        # Concatenate partitions
        masked_sample_in = torch.concat((masked_sample_A, masked_sample_B_in), axis=0)
        masked_sample_out = torch.concat((masked_sample_A, masked_sample_B_out), axis=0)

        return masked_sample_in, masked_sample_out, mask_id, attention_mask, start_token_indices, end_token_indices

    def mask_batch(self, batch, pred_ratio):
        """
        Apply GPU-optimized GLM-style masking to a batch of time series samples.
        
        Args:
            batch (torch.Tensor): Input batch of shape [B, L, D].
            pred_ratio (float): Prediction ratio for special masking.
            
        Returns:
            tuple: (masked_inputs, masked_outputs, mask_ids, attention_masks, start_indices, end_indices)
        """
        B = len(batch)
        batch_start_token_indices, batch_end_token_indices = [], []
        batch_masked_in, batch_masked_out, mask_ids, attention_masks = [], [], [], []
        
        for i, sample in enumerate(batch):
            sample_masked_in, sample_masked_out, mask_id, attention_mask, s_token_indices, e_token_indices = self._mask_one_sample(
                sample,
                pred_ratio,
                seq_len=None,
                add_start_end_token=True
            )
            batch_start_token_indices.extend([i, indice] for indice in s_token_indices)
            batch_end_token_indices.extend([i, indice] for indice in e_token_indices)
            batch_masked_in.append(sample_masked_in)
            batch_masked_out.append(sample_masked_out)
            mask_ids.append(mask_id)
            attention_masks.append(attention_mask)

        return batch_masked_in, batch_masked_out, mask_ids, attention_masks, torch.tensor(batch_start_token_indices,
                                                                                          dtype=torch.long), torch.tensor(
            batch_end_token_indices, dtype=torch.long)

    def pad_sequences(self, sequences, mask_ids=None, attention_masks=None, max_len=None, pad_value=0.0):
        """
        Pad a batch of sequences to the same length (GPU-optimized).
        
        Args:
            sequences (list): List of sequence tensors.
            mask_ids (list, optional): List of mask ID tensors.
            attention_masks (list, optional): List of attention mask tensors.
            max_len (int, optional): Maximum length for padding.
            pad_value (float): Value to use for padding.
            
        Returns:
            tuple: (padded_data, nonpadding_mask) where:
                - padded_data: Tuple of (padded_sequences, padded_mask_ids, padded_attention_masks)
                - nonpadding_mask: Boolean mask indicating real vs padded tokens
        """
        batch_size = len(sequences)
        patch_len = sequences[0].shape[-1]
        max_seq_len = (max(seq.shape[0] for seq in sequences))
        max_len = max_seq_len if max_len is None else max_len

        # Initialize padded tensors
        padded_sequences = torch.full((batch_size, max_len, patch_len), pad_value, dtype=sequences[0].dtype)
        nonpadding_mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        
        if mask_ids is not None:
            padded_mask_ids = torch.full((batch_size, max_len), 0, dtype=mask_ids[0].dtype)
        else:
            padded_mask_ids = None

        if attention_masks is not None:
            padded_attention_masks = torch.full((batch_size, max_len, max_len), 0, dtype=attention_masks[0].dtype)
        else:
            padded_attention_masks = None

        # Fill padded tensors
        for i, seq in enumerate(sequences):
            seq_len = len(seq)
            _len = min(max_len, seq_len)
            
            if mask_ids is not None:
                mask_id = mask_ids[i][:_len]
                padded_mask_ids[i, :_len] = mask_id
                
            if attention_masks is not None:
                _len = min(max_len, seq_len)
                attention_mask = attention_masks[i][:_len, :_len]
                padded_attention_masks[i, :_len, :_len] = attention_mask
                
            padded_sequences[i, :_len] = seq[:max_len]
            nonpadding_mask[i, :_len] = 1

        return (padded_sequences, padded_mask_ids, padded_attention_masks), nonpadding_mask

    def _get_position_id2(self, tensor):
        """
        Generate secondary position IDs for consecutive tokens (GPU-optimized).
        
        Args:
            tensor (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Secondary position IDs.
        """
        unique, indexes, counts = torch.unique_consecutive(tensor, return_counts=True, return_inverse=True)
        counts = counts.type(torch.int32)
        counts = torch.cat([torch.zeros(1, dtype=counts.dtype), counts], dim=0)
        counts_cumsum = counts.cumsum_(dim=0)
        pos_id = torch.arange(1, (tensor.size(0) + 1), dtype=torch.int32) - counts_cumsum[indexes]
        return pos_id

    def get_position_ids(self, padded_mask_ids, nonpadding_mask):
        """
        Generate GLM-style position IDs (GPU-optimized).
        
        Args:
            padded_mask_ids (torch.Tensor): Padded mask IDs.
            nonpadding_mask (torch.Tensor): Padding mask.
            
        Returns:
            dict: Dictionary containing position IDs tensors.
        """
        B, L = padded_mask_ids.shape
        
        # Generate primary position IDs
        position_id_1 = torch.arange(1, L + 1, dtype=torch.int32).unsqueeze_(0).repeat(B, 1)
        position_id_2 = torch.zeros((B, L), dtype=torch.int32)
        
        # Set position IDs for masked tokens
        position_id_1[padded_mask_ids > 0] = padded_mask_ids[padded_mask_ids > 0]
        position_id_1[padded_mask_ids < 0] = - padded_mask_ids[padded_mask_ids < 0]
        mask_token_indices = padded_mask_ids < 0
        position_id_1[nonpadding_mask == 0] = 0
        
        # Generate secondary position IDs
        for b in range(B):
            padded_mask_id = padded_mask_ids[b]
            position_id_2[b][padded_mask_id > 0] = self._get_position_id2(padded_mask_id[padded_mask_id > 0])

        # Convert mask IDs to boolean
        padded_mask_ids[padded_mask_ids < 0] = 0
        padded_mask_ids[padded_mask_ids > 0] = 1
        
        return dict(
            position_ids_1=position_id_1.type(torch.long),
            position_ids_2=position_id_2.type(torch.long),
            mask_ids=padded_mask_ids.type(torch.bool),
            mask_token_indices=mask_token_indices.type(torch.bool),
        )

    def __call__(self, batch, pred_ratio=0.3, to_device=True):
        """
        Apply GPU-optimized GLM-style masking to a batch of sequences.
        
        Args:
            batch (torch.Tensor): Input batch of shape [B, L, D].
            pred_ratio (float): Prediction ratio for special masking.
            to_device (bool): Whether to move results to the input device.
            
        Returns:
            dict: Dictionary containing masked inputs, labels, and masks.
        """
        device = batch.device
        B, L, D = batch.size()
        
        # Apply masking to batch
        masked_batch_in, masked_batch_out, mask_ids, attention_masks, batch_start_token_indices, batch_end_token_indices = self.mask_batch(
            batch, pred_ratio)
            
        # Pad masked inputs
        (padded_masked_batch_in, padded_mask_ids, attention_masks), nonpadding_mask = self.pad_sequences(
            sequences=masked_batch_in,
            mask_ids=mask_ids,
            attention_masks=attention_masks,
            max_len=int(L * (2 * self.mask_ratio + 1)),
            pad_value=0.0
        )
        
        # Get mask positions
        mask_pos = [[i for i, x in enumerate(seq) if x < 0] for seq in mask_ids]
        
        # Pad masked outputs
        (padded_masked_batch_out, _, _), _ = self.pad_sequences(masked_batch_out)
        
        # Generate position IDs
        position_id_dict = self.get_position_ids(padded_mask_ids, nonpadding_mask=nonpadding_mask)
        
        # Create token indices tensors
        start_token_indices = torch.zeros_like(padded_mask_ids, dtype=torch.bool)
        end_token_indices = torch.zeros_like(padded_mask_ids, dtype=torch.bool)
        start_token_indices[batch_start_token_indices[:, 0], batch_start_token_indices[:, 1]] = True
        end_token_indices[batch_end_token_indices[:, 0], batch_end_token_indices[:, 1]] = True

        data_dict = dict(
            input_patches=padded_masked_batch_in,  # [B, L, D]
            labels=padded_masked_batch_in,  # [B, L, D]
            attention_masks=attention_masks,  # [B, L, L]
            start_token_indices=start_token_indices,  # [B, L]
            end_token_indices=end_token_indices,  # [B, L]
        )
        data_dict.update(position_id_dict)

        if to_device:
            data_dict = {k: v.to(device) for k, v in data_dict.items()}

        data_dict["mask_pos"] = mask_pos
        return data_dict


if __name__ == "__main__":
    # L = np.random.randint(10, 100, 20)
    L = 50
    D = 24
    # batch = [np.random.rand(l, D) for l in L]
    batch = torch.rand((20, 50, D)).cuda()

    print(L)

    # processor_mlm = BaseMaskProcessor(0.15)
    # data = processor_mlm(batch, 50, mask_first=False)
    # print(data)

    processor = GLMMaskProcessor4TSGPU(0.15)
    data = processor(batch, to_device=True)
    print(data)
