from collections import defaultdict
from typing import Any, Callable, Optional

import lightning as L
import numpy as np
import torch
from jaxtyping import Bool, Float, Int

from uni2ts.transform import (
    AddObservedMask,
    AddTimeIndex,
    AddVariateIndex,
    DefaultPatchSizeConstraints,
    DummyValueImputation,
    ExtendMask,
    FlatPackCollection,
    FlatPackFields,
    GetPatchSize,
    ImputeTimeSeries,
    MaskedPrediction,
    PackFields,
    PatchCrop,
    Patchify,
    SampleDimension,
    SelectFields,
    SequencifyField,
    Transformation,
)

class MoiraiVoid(L.LightningModule):
    seq_fields: tuple[str, ...] = (
        "target",
        "observed_mask",
        "time_id",
        "variate_id",
        "prediction_mask",
        "patch_size",
    )
    pad_func_map: dict[str, Callable[[tuple[int, ...], np.dtype], np.ndarray]] = {
        "target": np.zeros,
        "observed_mask": np.zeros,
        "time_id": np.zeros,
        "variate_id": np.zeros,
        "prediction_mask": np.zeros,
        "patch_size": np.zeros,
    }

    def __init__(
        self,
        min_patches: int,
        min_mask_ratio: float,
        max_mask_ratio: float,
        max_dim: int,
        patch_sizes: list[int],
        max_seq_len: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        
    def forward(
        self,
        target: Float[torch.Tensor, "*batch seq_len max_patch"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len max_patch"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        time_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        patch_size: Int[torch.Tensor, "*batch seq_len"],
    ) -> dict[str, torch.Tensor]:
        """
        Simply return the input tensors for validation
        """
        return {
            "target": target,
            "observed_mask": observed_mask,
            "sample_id": sample_id,
            "time_id": time_id,
            "variate_id": variate_id,
            "prediction_mask": prediction_mask,
            "patch_size": patch_size,
        }

    def training_step(self, *args, **kwargs):
        return None

    def configure_optimizers(self):
        return None

    @property
    def train_transform_map(self) -> dict[str, Callable[..., Transformation]]:
        """
        Get a dictionary of Transforms for data validation
        """
        def default_train_transform():
            return (
                SampleDimension(
                    max_dim=self.hparams.max_dim,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + GetPatchSize(
                    min_time_patches=self.hparams.min_patches,
                    target_field="target",
                    patch_sizes=self.hparams.patch_sizes,
                    patch_size_constraints=DefaultPatchSizeConstraints(),
                    offset=True,
                )
                + PatchCrop(
                    min_time_patches=self.hparams.min_patches,
                    max_patches=self.hparams.max_seq_len,
                    will_flatten=True,
                    offset=True,
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + PackFields(
                    output_field="target",
                    fields=("target",),
                    feat=False,
                )
                + PackFields(
                    output_field="past_feat_dynamic_real",
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=False,
                )
                + AddObservedMask(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    observed_mask_field="observed_mask",
                    collection_type=dict,
                )
                + ImputeTimeSeries(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    imputation_method=DummyValueImputation(value=0.0),
                )
                + Patchify(
                    max_patch_size=max(self.hparams.patch_sizes),
                    fields=("target", "observed_mask"),
                    optional_fields=("past_feat_dynamic_real",),
                )
                + AddVariateIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    variate_id_field="variate_id",
                    expected_ndim=3,
                    max_dim=self.hparams.max_dim,
                    randomize=True,
                    collection_type=dict,
                )
                + AddTimeIndex(
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    time_id_field="time_id",
                    expected_ndim=3,
                    collection_type=dict,
                )
                + MaskedPrediction(
                    min_mask_ratio=self.hparams.min_mask_ratio,
                    max_mask_ratio=self.hparams.max_mask_ratio,
                    target_field="target",
                    truncate_fields=("variate_id", "time_id", "observed_mask"),
                    optional_truncate_fields=("past_feat_dynamic_real",),
                    prediction_mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + ExtendMask(
                    fields=tuple(),
                    optional_fields=("past_feat_dynamic_real",),
                    mask_field="prediction_mask",
                    expected_ndim=3,
                )
                + FlatPackCollection(
                    field="variate_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="time_id",
                    feat=False,
                )
                + FlatPackCollection(
                    field="prediction_mask",
                    feat=False,
                )
                + FlatPackCollection(
                    field="observed_mask",
                    feat=True,
                )
                + FlatPackFields(
                    output_field="target",
                    fields=("target",),
                    optional_fields=("past_feat_dynamic_real",),
                    feat=True,
                )
                + SequencifyField(field="patch_size", target_field="target")
                + SelectFields(fields=list(self.seq_fields))
            )

        return defaultdict(lambda: default_train_transform)
