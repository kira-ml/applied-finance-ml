"""
Time-series cross-validation splitter.

Purpose:
Generate train/validation index pairs that respect temporal order
and prevent data leakage through proper chronological splitting.

This module intentionally avoids:
- Accessing feature values
- Model training
- Metric computation
- Any data transformation
"""

from typing import Generator, Tuple, Optional
import numpy as np


def time_series_split(
    n_samples: int,
    n_splits: int = 5,
    train_size: Optional[int] = None,
    initial_train_size: Optional[int] = None,
    valid_size: int = 1,
    gap: int = 0,
    expanding: bool = True
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate train/validation indices for time-series cross-validation.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the dataset.
    n_splits : int
        Number of train/validation splits to generate.
    train_size : int, optional
        Fixed training window size for sliding window mode (expanding=False).
        Ignored when expanding=True.
    initial_train_size : int, optional
        Starting training size for fold 1 in expanding window mode.
        If None, defaults to n_samples - n_splits * (valid_size + gap),
        which utilises all available data across all folds.
    valid_size : int
        Number of samples to use for validation per fold.
    gap : int
        Number of samples to skip between train and validation sets.
        Prevents leakage from adjacent time points.
    expanding : bool
        If True, training set expands with each split (expanding window).
        If False, training set is a sliding window of fixed size.

    Yields
    ------
    Tuple[np.ndarray, np.ndarray]
        - train_indices: Array of indices for training set
        - valid_indices: Array of indices for validation set

    Raises
    ------
    ValueError
        If parameters are invalid or cannot produce requested splits.
    """
    # Input validation
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    
    if n_splits <= 0:
        raise ValueError(f"n_splits must be positive, got {n_splits}")
    
    if valid_size <= 0:
        raise ValueError(f"valid_size must be positive, got {valid_size}")
    
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    
    # Validate train_size if provided
    if train_size is not None:
        if train_size <= 0:
            raise ValueError(f"train_size must be positive, got {train_size}")
        if train_size + valid_size > n_samples:
            raise ValueError(
                f"train_size ({train_size}) + valid_size ({valid_size}) "
                f"exceeds n_samples ({n_samples})"
            )
    
    # Calculate available samples for splitting
    min_required = valid_size
    if train_size is not None:
        min_required += train_size
    
    if min_required > n_samples:
        raise ValueError(
            f"Minimum required samples ({min_required}) exceeds "
            f"available samples ({n_samples})"
        )
    
    # Handle None train_size based on mode
    if train_size is None:
        if expanding:
            # Expanding window: default initial size uses ALL available data
            # so that fold 1 is already well-trained and folds converge quickly.
            default_min_train = n_samples - n_splits * (valid_size + gap)
            if default_min_train <= 0:
                raise ValueError(
                    f"Cannot create {n_splits} expanding splits with "
                    f"valid_size={valid_size} and gap={gap}. "
                    f"Need at least {n_splits * (valid_size + gap) + 1} samples, "
                    f"have {n_samples}"
                )
            start_train = initial_train_size if initial_train_size is not None else default_min_train
            if start_train <= 0:
                raise ValueError(
                    f"initial_train_size must be positive, got {start_train}"
                )
            train_sizes = [
                start_train + i * (valid_size + gap)
                for i in range(n_splits)
            ]
        else:
            # Sliding window: use as much as possible while leaving room for validation
            max_possible_train = n_samples - (n_splits * valid_size)
            if max_possible_train <= 0:
                raise ValueError(
                    f"Cannot create {n_splits} sliding splits with valid_size={valid_size}. "
                    f"Need at least {n_splits * valid_size + 1} samples, have {n_samples}"
                )
            train_size = max_possible_train
            train_sizes = [train_size] * n_splits
    else:
        # Fixed train_size (sliding mode only — expanding handled above)
        train_sizes = [train_size] * n_splits
    
    # Generate splits
    for split_idx in range(n_splits):
        current_train_size = train_sizes[split_idx]
        
        # Calculate split boundaries
        if expanding:
            # Expanding: train end moves forward each split
            train_end = current_train_size
        else:
            # Sliding: both train start and end move forward
            train_start = split_idx * valid_size
            train_end = train_start + current_train_size
        
        valid_start = train_end + gap
        valid_end = valid_start + valid_size
        
        # Validate boundaries
        if valid_end > n_samples:
            raise ValueError(
                f"Split {split_idx + 1}: validation set [{valid_start}:{valid_end}] "
                f"exceeds dataset size {n_samples}"
            )
        
        # Generate index arrays
        train_indices = np.arange(train_end - current_train_size, train_end)
        valid_indices = np.arange(valid_start, valid_end)
        
        yield train_indices, valid_indices


# Minimal usage example
if __name__ == "__main__":
    # Example: Generate splits for 100 days of data
    n_samples = 100
    
    print("Expanding window splits (growing training set):")
    for i, (train_idx, valid_idx) in enumerate(time_series_split(
        n_samples=n_samples,
        n_splits=5,
        valid_size=10,
        gap=0,
        expanding=True
    )):
        print(f"Split {i+1}: train {train_idx[0]}-{train_idx[-1]} "
              f"({len(train_idx)} days) | "
              f"valid {valid_idx[0]}-{valid_idx[-1]} ({len(valid_idx)} days)")
    
    print("\nSliding window splits (fixed training size):")
    for i, (train_idx, valid_idx) in enumerate(time_series_split(
        n_samples=n_samples,
        n_splits=5,
        valid_size=10,
        gap=0,
        expanding=False
    )):
        print(f"Split {i+1}: train {train_idx[0]}-{train_idx[-1]} "
              f"({len(train_idx)} days) | "
              f"valid {valid_idx[0]}-{valid_idx[-1]} ({len(valid_idx)} days)")
    
    print("\nWith gap between train and validation (prevents leakage):")
    for i, (train_idx, valid_idx) in enumerate(time_series_split(
        n_samples=n_samples,
        n_splits=3,
        valid_size=10,
        gap=5,  # Skip 5 days between train and validation
        expanding=True
    )):
        print(f"Split {i+1}: train {train_idx[0]}-{train_idx[-1]} | "
              f"gap {train_idx[-1]+1}-{valid_idx[0]-1} | "
              f"valid {valid_idx[0]}-{valid_idx[-1]}")