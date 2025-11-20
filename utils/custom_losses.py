import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

class KDSoftLoss(nn.Module):
    """
    Soft knowledge-distillation loss using KL divergence.
    
    Args:
        logits_dim: Dimension over which to apply softmax (usually channel dim).
        temperature: Soften/sharpen the logits for distillation.
        log_target: Whether the teacher probabilities are provided as log-probs.
        reduction: KL reduction mode ("sum", "batchmean", or "mean").
    """
    def __init__(self, logits_dim: int = 1, temperature: float = 0.5, log_target: bool = False, reduction: Literal["sum", "batchmean", "mean"] = "batchmean"):
        super().__init__()
        self.temperature = temperature
        # self.alpha = alpha
        self.logits_dim = logits_dim
        self.log_target = log_target
        self.reduction = reduction

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        """
        Compute the softened KL divergence between teacher and student outputs.

        Args:
            student_logits: Raw logits predicted by the student model.
            teacher_logits: Raw logits predicted by the teacher model.

        Returns:
            Temperature-scaled KL divergence loss.
        """
        # Convert teacher logits to (log-)probabilities depending on log_target setting
        teacher_probabilities = F.log_softmax(teacher_logits / self.temperature , dim=self.logits_dim) if self.log_target else F.softmax(teacher_logits / self.temperature , dim=self.logits_dim)
        # Always compute student in log-space for KL-div
        student_probabilities = F.log_softmax(student_logits / self.temperature, dim=self.logits_dim)
        # KL divergence between student and teacher distributions
        kl_div = F.kl_div(student_probabilities, teacher_probabilities, reduction=self.reduction, log_target=self.log_target)
        # return self.alpha * (self.temperature ** 2) * kl_div
        # Scale by T^2 as in Hinton KD formulation
        return (self.temperature ** 2) * kl_div

class KDHardSoftLoss(nn.Module):
    """
    Combined KD loss that mixes hard-label loss and soft-teacher loss.

    Args:
        hard_loss: Standard supervised loss (e.g., CE, DiceCE).
        soft_loss: Soft KD loss (e.g., KDSoftLoss instance).
        alpha: Weight controlling contribution of soft vs hard loss.
               0 → only hard loss, 1 → only soft loss.
    """
    def __init__(self, hard_loss: nn.Module, soft_loss: nn.Module, alpha:float=0.5):
        super().__init__()
        assert 0.0 <= alpha <= 1.0
        self.hard_loss = hard_loss
        self.soft_loss = soft_loss
        # adjust weight trade-off between hard loss and soft loss
        self.alpha = alpha
    
    def forward(self, student_logits: torch.Tensor, targets: torch.Tensor, teacher_logits: torch.Tensor):
        """
        Compute combined supervised (hard) + distillation (soft) loss.

        Args:
            student_logits: Predictions from the student network.
            targets: Ground-truth labels for hard loss.
            teacher_logits: Teacher model logits (detached automatically).

        Returns:
            Weighted sum of hard and soft losses.
        """
        # avoids backprop into the teacher by accident.
        teacher_logits = teacher_logits.detach()
        return (1.0 - self.alpha) * self.hard_loss(student_logits, targets) +  self.alpha * self.soft_loss(student_logits, teacher_logits)