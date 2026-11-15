"""
Aligns the modalities of a multimodal run on the samples they have in
common, so the feature blocks share one row order and one label vector
(an error if a sample's label differs between modalities).
"""

import numpy as np


class ModalityAligner:
    @staticmethod
    def align(X_raw: dict) -> tuple[dict, np.ndarray, list]:
        modalities = list(X_raw.keys())
        common_samples_set = set(X_raw[modalities[0]]["ids"])
        for modality in modalities[1:]:
            common_samples_set = common_samples_set.intersection(X_raw[modality]["ids"])
        common_samples = sorted(common_samples_set)
        X_aligned = {}
        y_aligned = None
        for modality in modalities:
            position_by_id = {
                sample_id: indices for indices, sample_id in enumerate(X_raw[modality]["ids"])
            }
            aligned_indices = [position_by_id[sample_id] for sample_id in common_samples]
            raw_values = X_raw[modality]["values"]
            aligned_values = (
                raw_values.iloc[aligned_indices]
                if hasattr(raw_values, "iloc")
                else raw_values[aligned_indices]
            )
            X_aligned[modality] = {
                "values": aligned_values,
                "feature_names": X_raw[modality].get("feature_names"),
            }
            raw_labels = X_raw[modality]["labels"]
            modality_labels = np.array([raw_labels[indices] for indices in aligned_indices])
            if y_aligned is None:
                y_aligned = modality_labels
            elif not np.array_equal(y_aligned, modality_labels):
                raise ValueError(f"Labels are not consistent for modality: {modality}")
        if y_aligned is None:
            raise ValueError("X_raw must contain at least one modality")
        return X_aligned, y_aligned, common_samples
