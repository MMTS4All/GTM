"""
Mask processors for time series data preprocessing.

This module provides various masking strategies for time series data,
including random masking, span masking, and GLM-style masking for
pre-training transformer models on time series data.
"""

from copy import deepcopy
from typing import *
import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class BaseMaskProcessor:
    """
    Base mask processor for time series data.
    
    This class implements the basic random masking strategy where individual
    patches are randomly masked with a specified ratio. It handles both
    padded and unpadded sequences and generates appropriate attention masks.
    
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
    mask_ratio: float=0.15

    def _mask_one_sample(self, sample: np.ndarray, seq_len:int=None):
        """
        Mask a single time series sample using random masking.
        
        Args:
            sample (np.ndarray): Input time series sample with shape [L, D].
            seq_len (int, optional): Sequence length for padded samples.
            
        Returns:
            tuple: (masked_sample, mask_ids, attention_mask) where:
                - masked_sample: Sample with masked positions set to zero
                - mask_ids: Binary array indicating masked positions
                - attention_mask: Attention mask for transformer models
        """
        L, D = sample.shape
        attention_mask = np.ones((L, L), dtype=int)
        if not seq_len:
            # the sample is not padded
            seq_len = L
        attention_mask[seq_len:, :] = 0
        attention_mask[:, seq_len:] = 0
        len_keep = int((1 - self.mask_ratio) * seq_len)
        noise = torch.rand(seq_len)
        # sort noise for each sample
        ids_shuffle = np.argsort(noise)  # ascend: small is keep, large is remove
        ids_restore = np.argsort(ids_shuffle) # ids_restore: [L]

        ids_keep = ids_shuffle[:len_keep]
        x_kept = sample[ids_keep]
        x_removed = np.zeros((L-len_keep, D), dtype=sample.dtype)
        sample_masked = np.concatenate([x_kept, x_removed], axis=0)
        sample_masked[:seq_len] = sample_masked[:seq_len][ids_restore]

        mask_id = np.zeros(L, dtype=int)
        mask_id[len_keep:seq_len] = 1
        mask_id[:seq_len] = mask_id[:seq_len][ids_restore]
        return sample_masked, mask_id, attention_mask

    
    def mask_batch(self, batch: List[np.ndarray], seq_lengths=None):
        """
        Apply random masking to a batch of time series samples.
        
        Args:
            batch (list): List of arrays with shape [L_i, D] for each sample.
            seq_lengths: List or np.ndarray of actual sequence lengths for padded samples.
            
        Returns:
            tuple: (batch_masked, mask_ids, attention_masks) where:
                - batch_masked: List of masked samples
                - mask_ids: List of mask indicator arrays
                - attention_masks: List of attention masks
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
        Generate position IDs for transformer models.
        
        Args:
            padded_mask_ids (np.ndarray): Padded mask IDs with shape [B, L].
            
        Returns:
            dict: Dictionary containing position_ids tensor.
        """
        B, L = padded_mask_ids.shape
        pos_ids = torch.arange(1, L+1).unsqueeze_(0).repeat(B, 1)
        return dict(position_ids=pos_ids)
    

    @staticmethod
    def pad_sequences(sequences, mask_ids=None, attention_masks=None, max_len=None, pad_value=0.0):
        """
        Pad a batch of sequences to the same length and generate attention masks.
        
        Args:
            sequences: List of arrays representing input sequences.
            mask_ids: List of arrays representing mask IDs.
            attention_masks: List of arrays representing attention masks.
            max_len: Maximum length to pad sequences. If None, uses longest sequence.
            pad_value: Value to use for padding. Defaults to 0.0.
            
        Returns:
            tuple: ((padded_sequences, padded_mask_ids, padded_attention_masks), nonpadding_mask)
        """
        batch_size = len(sequences)
        patch_len = sequences[0].shape[-1]
        max_seq_len = (max(seq.shape[0] for seq in sequences))
        max_len = max_seq_len if max_len is None else min(max_seq_len, max_len)

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
    

    def __call__(self, batch, max_len: int, mask_first=False):
        """
        Process a batch of time series data with masking.
        
        Args:
            batch: List of time series samples.
            max_len: Maximum sequence length for padding.
            mask_first: Whether to mask before padding. Defaults to False.
            
        Returns:
            dict: Dictionary containing processed tensors for model input.
        """
        if mask_first:
            # maskfirst, then padding
            masked_batch, mask_ids, attention_masks = self.mask_batch(batch)
            (padded_batch, _, _), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=max_len,
                pad_value=0.0
            )
            (padded_masked_batch, padded_mask_ids, attention_masks), _nonpadding_mask = self.pad_sequences(
                sequences=masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=max_len,
                pad_value=0.0
            )
            assert (nonpadding_mask-_nonpadding_mask).sum() == 0
            attention_masks = nonpadding_mask[:, :, None] * attention_masks 
        else:
            # padding first, then mask
            (padded_batch, _, __), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=max_len,
                pad_value=0.0
            )
            seq_lengths = nonpadding_mask.sum(-1)
            padded_masked_batch, padded_mask_ids, attention_masks = self.mask_batch(padded_batch, seq_lengths=seq_lengths)
            padded_masked_batch = np.stack(padded_masked_batch, axis=0)
            padded_mask_ids = np.stack(padded_mask_ids, axis=0)
            attention_masks = np.stack(attention_masks, axis=0)
            # padded_masked_batch, padded_mask_ids = torch.from_numpy(padded_masked_batch), torch.from_numpy(padded_mask_ids)
        position_id_dict = self.get_position_ids(padded_mask_ids)

        data_dict = dict(
            input_patches=torch.from_numpy(padded_masked_batch), # [B, L, D]
            labels=torch.from_numpy(padded_batch),  # [B, L, D]
            mask_ids=torch.from_numpy(padded_mask_ids), # [B, L]
            attention_masks=torch.from_numpy(attention_masks), # [B, L, L]
        )
        data_dict.update(position_id_dict)
        return data_dict


class SpanMaskProcessor(BaseMaskProcessor):
    """
    Span mask processor for time series data.
    
    This class implements span-based masking where consecutive patches
    are masked together as spans, using Poisson distribution for span lengths.
    This approach mimics real-world missing data patterns more closely.
    
    Examples:
    >>> d = [p1, p2, p3, p4, p5, p6]
    >>> masked_d = [p1, <M>, <M>, p4, <M>, <M>]
    >>> mask_ids = [0, 1, 1, 0, 1, 1]
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
    mask_ratio: float = 0.15
    possion_lambda: int = 3

    @staticmethod
    def _detect_overlap(start, end, spans_list):
        """
        Check if a span overlaps with existing spans.
        
        Args:
            start: Start position of the new span.
            end: End position of the new span.
            spans_list: List of existing spans.
            
        Returns:
            bool: True if overlap detected, False otherwise.
        """
        for span_s, span_e in spans_list:
            if (span_s <= start <= span_e) or (span_s <= end <= span_e):
                return True
        return False

    def _mask_one_sample(self, sample: np.ndarray, seq_len: int = None):
        """
        Mask a single time series sample using span masking.
        
        Args:
            sample (np.ndarray): Input time series sample with shape [L, D].
            seq_len (int, optional): Sequence length for padded samples.
            
        Returns:
            tuple: (masked_sample, mask_ids, attention_mask) where:
                - masked_sample: Sample with masked spans set to zero
                - mask_ids: Binary array indicating masked positions
                - attention_mask: Attention mask for transformer models
        """
        L, D = sample.shape
        masked_sample = deepcopy(sample)
        mask_id = np.zeros(L, dtype=int)
        attention_mask = np.ones((L, L), dtype=int)
        if not seq_len:
            # the sample is not padded
            seq_len = L
        attention_mask[seq_len:, :] = 0
        attention_mask[:, seq_len:] = 0
        
        total_to_mask = int(self.mask_ratio * seq_len)
        mask_count = 0
        spans_list = []
        
        while mask_count < total_to_mask:
            span_length = np.random.poisson(lam=self.possion_lambda)
            if span_length == 0 or span_length >= seq_len:
                continue  # skip span with length = 0
            
            start_pos = np.random.randint(0, seq_len-span_length)
            end_pos = min(start_pos + span_length, seq_len)

            # ensure there is no overlap between spans
            if self._detect_overlap(start_pos, end_pos, spans_list):
                continue
            
            # mask
            masked_sample[start_pos: end_pos] = 0.0
            mask_id[start_pos: end_pos] = 1
            mask_count += end_pos - start_pos
        
        return masked_sample, mask_id, attention_mask


class GLMMaskProcessor4TS(BaseMaskProcessor):
    """
    GLM-style mask processor for time series data.
    
    This class implements a GLM-inspired masking strategy where masked spans
    are moved to the end of the sequence, enabling bidirectional reconstruction
    while maintaining causal attention patterns.
    
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
    mask_ratio: float = 0.15
    possion_lambda: int = 3

    @staticmethod
    def _detect_overlap(start, end, spans_list):
        """
        Check if a span overlaps with existing spans.
        
        Args:
            start: Start position of the new span.
            end: End position of the new span.
            spans_list: List of existing spans.
            
        Returns:
            bool: True if overlap detected, False otherwise.
        """
        for span_s, span_e in spans_list:
            if (span_s <= start <= span_e) or (span_s <= end <= span_e):
                return True
        return False

    def _mask_one_sample(self, sample: np.ndarray, seq_len: int = None):
        """
        Mask a single time series sample using GLM-style masking.
        
        Args:
            sample (np.ndarray): Input time series sample with shape [L, D].
            seq_len (int, optional): Sequence length for padded samples.
            
        Returns:
            tuple: (masked_sample, mask_ids, attention_mask) where:
                - masked_sample: Sample with masked spans moved to end
                - mask_ids: Position indicators for masked spans
                - attention_mask: Causal attention mask for GLM training
        """
        L, D = sample.shape
        if not seq_len:
            # the sample is not padded
            seq_len = L
        total_to_mask = int(self.mask_ratio * seq_len)
        mask_count = 0
        spans_list = []
        
        while mask_count < total_to_mask:
            span_length = np.random.poisson(lam=self.possion_lambda)
            if span_length == 0 or span_length >= seq_len:
                continue  # skip span with length = 0
            
            start_pos = np.random.randint(0, seq_len-span_length)
            end_pos = min(start_pos + span_length, seq_len)

            # ensure there is no overlap between spans
            if self._detect_overlap(start_pos, end_pos, spans_list):
                continue
            
            # mask
            spans_list.append((start_pos, end_pos))
            mask_count += end_pos - start_pos

        all_span_len = sum([end_pos - start_pos for start_pos, end_pos in spans_list])
        part_A_len = (seq_len + len(spans_list)) - all_span_len
        part_B_len = all_span_len
        new_L = part_A_len + part_B_len
        masked_sample_A = np.zeros((part_A_len, D), dtype=sample.dtype)
        masked_sample_B = np.zeros((part_B_len, D), dtype=sample.dtype)

        attention_mask = np.ones((new_L, new_L), dtype=int)
        mask_id = np.zeros(new_L, dtype=int)
        # attention_mask[seq_len:, :] = 0
        # attention_mask[:, seq_len:] = 0
        
        last_end = 0
        cur_j_A = 0
        cur_j_B = 0

        for i, (start_pos, end_pos) in enumerate(sorted(spans_list)):
            masked_sample_A[cur_j_A:cur_j_A+(start_pos-last_end)] = sample[last_end:start_pos]
            cur_j_A += start_pos - last_end
            mask_id[cur_j_A] = -(cur_j_A + 1)   # meaning this one is mask token, usful for generating position ids
            # position_id_1_dict[(start_pos, end_pos)] = cur_j_A + 1
            last_end = end_pos

        for i, (start_pos, end_pos) in enumerate(spans_list):
            masked_sample_B[cur_j_B:cur_j_B+(end_pos-start_pos)] = sample[start_pos:end_pos]
            # masked_sample[start_pos:end_pos] = 0.0
            _s, _e = part_A_len+cur_j_B, part_A_len+cur_j_B+(end_pos-start_pos)
            # mask_id[_s: _e] = position_id_1_dict[(start_pos, end_pos)]
            mask_id[_s: _e] = i + 1
            attention_mask[_s:_e, _s:_e] = np.tril(
                np.ones((end_pos-start_pos, end_pos-start_pos), dtype=int),
            )
            cur_j_B += (end_pos-start_pos)

        masked_sample = np.concatenate((masked_sample_A, masked_sample_B), axis=0)
        
        return masked_sample, mask_id, attention_mask

    def _get_position_id2(self, tensor):
        """
        Generate secondary position IDs for GLM masking.
        
        Args:
            tensor: Input tensor for position ID generation.
            
        Returns:
            torch.Tensor: Secondary position IDs.
        """
        # diffs = torch.cat([torch.tensor([True]), tensor[1:] != tensor[:-1]])
        # cumsum = diffs.cumsum(dim=0)
        unique, counts = torch.unique_consecutive(tensor, return_counts=True)
        counts = torch.cat([torch.zeros(2, dtype=counts.dtype), counts], dim=0)
        counts_cumsum = counts.cumsum_(dim=0)
        pos_id = torch.arange(1, (tensor.size(0)+1)) - counts_cumsum[tensor]
        return pos_id

    def get_position_ids(self, padded_mask_ids, nonpadding_mask):
        """
        Generate position IDs for GLM-style masking.
        
        Args:
            padded_mask_ids (np.ndarray): Padded mask IDs with shape [B, L].
            nonpadding_mask: Padding mask indicating real tokens.
            
        Returns:
            dict: Dictionary containing position_ids_1, position_ids_2, and mask_ids.
        """
        if isinstance(padded_mask_ids, np.ndarray):
            padded_mask_ids = torch.from_numpy(padded_mask_ids)
        B, L = padded_mask_ids.shape
        position_id_1 = torch.arange(1, L+1, dtype=torch.long).unsqueeze_(0).repeat(B, 1)
        position_id_2 = torch.zeros((B, L), dtype=torch.long)
        # position_id_1[]
        position_id_1[padded_mask_ids > 0] = padded_mask_ids[padded_mask_ids > 0]
        position_id_1[padded_mask_ids < 0] = - padded_mask_ids[padded_mask_ids < 0]
        position_id_1[nonpadding_mask==0] = 0
        for b in range(B):
            padded_mask_id = padded_mask_ids[b]
            position_id_2[b][padded_mask_id > 0] = self._get_position_id2(padded_mask_id[padded_mask_id > 0])
        
        padded_mask_ids[padded_mask_ids < 0] = 0
        padded_mask_ids[padded_mask_ids > 0] = 1
        return dict(
            position_ids_1=position_id_1,
            position_ids_2=position_id_2,
            mask_ids=padded_mask_ids
        )

    def __call__(self, batch, max_len: int, mask_first=False):
        """
        Process a batch of time series data with GLM-style masking.
        
        Args:
            batch: List of time series samples.
            max_len: Maximum sequence length for padding.
            mask_first: Whether to mask before padding. Defaults to False.
            
        Returns:
            dict: Dictionary containing processed tensors for GLM model input.
        """
        if mask_first:
            # maskfirst, then padding
            masked_batch, mask_ids, attention_masks = self.mask_batch(batch)
            (padded_batch, _, _), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=max_len,
                pad_value=0.0
            )
            (padded_masked_batch, padded_mask_ids, attention_masks), _nonpadding_mask = self.pad_sequences(
                sequences=masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=max_len,
                pad_value=0.0
            )
            assert (nonpadding_mask-_nonpadding_mask).sum() == 0
            attention_masks = nonpadding_mask[:, :, None] * attention_masks 
        else:
            # padding first, then mask
            (padded_batch, _, __), nonpadding_mask = self.pad_sequences(
                sequences=batch,
                max_len=max_len,
                pad_value=0.0
            )
            seq_lengths = nonpadding_mask.sum(-1)
            masked_batch, mask_ids, attention_masks = self.mask_batch(padded_batch, seq_lengths=seq_lengths)
            # GLM masking mechanism would cause variable lengths for a padded batch
            (padded_masked_batch, padded_mask_ids, attention_masks), nonpadding_mask = self.pad_sequences(
                masked_batch,
                mask_ids=mask_ids,
                attention_masks=attention_masks,
                max_len=None
            )
            attention_masks = nonpadding_mask[:, :, None] * attention_masks 
            # padded_masked_batch = np.stack(padded_masked_batch, axis=0)
            # padded_mask_ids = np.stack(padded_mask_ids, axis=0)
            # attention_masks = np.stack(attention_masks, axis=0)
            # padded_masked_batch, padded_mask_ids = torch.from_numpy(padded_masked_batch), torch.from_numpy(padded_mask_ids)
        position_id_dict = self.get_position_ids(padded_mask_ids, nonpadding_mask=nonpadding_mask)

        data_dict = dict(
            input_patches=torch.from_numpy(padded_masked_batch), # [B, L, D]
            labels=torch.from_numpy(padded_masked_batch),  # [B, L, D]
            mask_ids=torch.from_numpy(padded_mask_ids), # [B, L]
            attention_masks=torch.from_numpy(attention_masks), # [B, L, L]
        )
        data_dict.update(position_id_dict)
        return data_dict


if __name__ == "__main__":
    L = np.random.randint(10, 100, 20)
    D = 24
    batch = [np.random.rand(l, D) for l in L]

    print(L)
    
    processor_mlm = BaseMaskProcessor(0.15)
    data = processor_mlm(batch, 50, mask_first=False)
    print(data)

    processor = GLMMaskProcessor4TS(0.15)
    data = processor(batch, 50, mask_first=False)
    print(data)

    processor_span = SpanMaskProcessor(0.15)
    data = processor_span(batch, 50, mask_first=False)
    print(data)
