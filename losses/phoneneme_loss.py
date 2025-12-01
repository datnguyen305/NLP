import torch.nn as nn

class PhonemeLoss(nn.Module):
    def __init__(self, padding_idx=0):
        super().__init__()
        # ignore_index=padding_idx để không tính loss cho các token padding
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)
    
    def forward(self, outputs, targets):
        """
        outputs: Tuple (logit_onset, logit_medial, logit_nucleus, logit_coda)
                 Mỗi cái có shape (Batch, Seq_Len, Vocab_Size)
                 
        targets: Tensor (Batch, Seq_Len, 4) - Ground Truth Label
        """
        p_onset, p_medial, p_nucleus, p_coda = outputs
        
        # Tách target ra 4 phần tương ứng
        t_onset = targets[..., 0]
        t_medial = targets[..., 1]
        t_nucleus = targets[..., 2]
        t_coda = targets[..., 3]
        
        # Tính Loss cho từng phần
        # CrossEntropy yêu cầu input (Batch, Class, Seq) hoặc flatten
        # Ta reshape: (Batch * Seq_Len, Vocab_Size) vs (Batch * Seq_Len)
        
        vocab_size = p_onset.shape[-1]
        
        loss_onset = self.criterion(p_onset.reshape(-1, vocab_size), t_onset.reshape(-1))
        loss_medial = self.criterion(p_medial.reshape(-1, vocab_size), t_medial.reshape(-1))
        loss_nucleus = self.criterion(p_nucleus.reshape(-1, vocab_size), t_nucleus.reshape(-1))
        loss_coda = self.criterion(p_coda.reshape(-1, vocab_size), t_coda.reshape(-1))
        
        # Tổng Loss
        total_loss = loss_onset + loss_medial + loss_nucleus + loss_coda
        
        return total_loss