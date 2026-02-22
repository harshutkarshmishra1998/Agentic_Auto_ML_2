from dataclasses import dataclass


@dataclass
class DatasetSignals:
    rows: int
    cols: int

    is_small_data: bool
    is_large_data: bool
    is_high_dim: bool
    overfit_risk: bool

    onehot_count: int
    freq_encoded_count: int
    label_encoded_count: int

    clustering_ready: bool