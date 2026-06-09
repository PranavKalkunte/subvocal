"""Pydantic schema for reproducible model training configuration."""

from pydantic import BaseModel, Field
from typing import Optional


class TrainingConfig(BaseModel):
    """Configuration class for reproducible training runs."""

    model_type: str = Field(
        default="rf",
        description="Model type to train: 'rf', 'svm', 'cnn', 'gru', 'transformer'"
    )
    seed: int = Field(
        default=42,
        description="Random seed to ensure reproducibility"
    )
    test_size: float = Field(
        default=0.2,
        description="Fraction of data used for validation split"
    )

    # Classical ML (Random Forest & Support Vector Machine)
    rf_n_estimators: int = Field(
        default=200,
        description="Number of trees in Random Forest classifier"
    )
    svm_c: float = Field(
        default=1.0,
        description="C regularization parameter for SVM"
    )
    svm_kernel: str = Field(
        default="rbf",
        description="Kernel function for SVM (e.g. 'rbf', 'linear', 'poly')"
    )
    lda_components: Optional[int] = Field(
        default=None,
        description="Number of components for Linear Discriminant Analysis dimensionality reduction"
    )

    # Deep Learning (CNN, GRU, Transformer)
    epochs: int = Field(
        default=40,
        description="Number of training epochs for deep models"
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for gradient descent"
    )
    lr: float = Field(
        default=1e-3,
        description="Initial learning rate"
    )
    weight_decay: float = Field(
        default=1e-4,
        description="Weight decay regularization coefficient"
    )

    # Architecture specific hyperparameters
    hidden_size: int = Field(
        default=64,
        description="Hidden state size for GRU, or model embedding dimension for Transformer"
    )
    num_layers: int = Field(
        default=2,
        description="Number of recurrent or Transformer layers"
    )
