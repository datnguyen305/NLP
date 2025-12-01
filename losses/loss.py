import torch
import torch.nn as nn

class TextSumLoss(nn.Module):
    def __init__(self, pad_idx: int, label_smoothing: float = 0.1):
        super(TextSumLoss, self).__init__()
        
        # PyTorch hỗ trợ sẵn label_smoothing trong CrossEntropyLoss từ bản 1.10+
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_idx,      # Quan trọng: Bỏ qua padding
            label_smoothing=label_smoothing # Quan trọng: Giúp model tổng quát hóa tốt hơn
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán loss.
        
        Args:
            logits: Output của model. Shape (Batch_Size, Seq_Len, Vocab_Size)
            targets: Nhãn thực tế (đã shift). Shape (Batch_Size, Seq_Len)
        """
        
        # 1. Reshape để phù hợp với input của CrossEntropyLoss
        # CrossEntropyLoss yêu cầu input: (N, C) và target: (N)
        # N = Batch_Size * Seq_Len
        # C = Vocab_Size
        
        logits_flat = logits.reshape(-1, logits.size(-1)) 
        targets_flat = targets.reshape(-1)
        
        loss = self.criterion(logits_flat, targets_flat)
        
        return loss