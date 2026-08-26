"""Adaptation subpackage — SAL/LBN, CPEP, Variance Transfer."""

from subvocal.emg_core.adaptation.sal_lbn import LBN, SAL, SAL_LBN, adapt_sal_lbn

try:
    from subvocal.emg_core.adaptation.cpep import (
        CPEPFramework,
        EMGEncoder,
        PoseEncoder,
        contrastive_loss,
        embedding_knn_classify,
        knn_classify,
        knn_predict,
        l2_normalize_embeddings,
        pose_emg_contrastive_loss,
        zero_shot_classify,
        zero_shot_knn_classify,
    )
except ImportError:
    CPEPFramework = None  # type: ignore[assignment]
    EMGEncoder = None  # type: ignore[assignment]
    PoseEncoder = None  # type: ignore[assignment]
    pose_emg_contrastive_loss = None  # type: ignore[assignment]
    contrastive_loss = None  # type: ignore[assignment]
    knn_classify = None  # type: ignore[assignment]
    embedding_knn_classify = None  # type: ignore[assignment]
    zero_shot_knn_classify = None  # type: ignore[assignment]
    zero_shot_classify = None  # type: ignore[assignment]
    knn_predict = None  # type: ignore[assignment]
    l2_normalize_embeddings = None  # type: ignore[assignment]

try:
    from subvocal.emg_core.adaptation.variance_transfer import (
        GaussianClassificationModel,
        VarianceTransferGCM,
        predict,
        pretrain_variance_transfer,
        transfer_to_target,
    )
except ImportError:
    GaussianClassificationModel = None  # type: ignore[assignment]
    VarianceTransferGCM = None  # type: ignore[assignment]
    pretrain_variance_transfer = None  # type: ignore[assignment]
    transfer_to_target = None  # type: ignore[assignment]
    predict = None  # type: ignore[assignment]

__all__ = [
    "SAL",
    "LBN",
    "SAL_LBN",
    "adapt_sal_lbn",
    "EMGEncoder",
    "PoseEncoder",
    "pose_emg_contrastive_loss",
    "contrastive_loss",
    "CPEPFramework",
    "knn_classify",
    "embedding_knn_classify",
    "zero_shot_knn_classify",
    "zero_shot_classify",
    "knn_predict",
    "l2_normalize_embeddings",
    "GaussianClassificationModel",
    "VarianceTransferGCM",
    "pretrain_variance_transfer",
    "transfer_to_target",
    "predict",
]
