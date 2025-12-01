import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
from utils.instance import Instance # Import class Instance của bạn

class ViCollator:
    def __init__(self, padding_idx: int = 0):
        """
        Args:
            padding_idx: Giá trị dùng để padding (mặc định là 0).
                         Khi pad, vector sẽ là [0, 0, 0, 0]
        """
        self.padding_idx = padding_idx

    def __call__(self, batch: List[Instance]):
        """
        Hàm này sẽ được DataLoader gọi để gộp list các Instance thành một Batch.
        """
        src_ids_list = [item.src_ids for item in batch]
        decoder_input_ids_list = [item.decoder_input_ids for item in batch]
        labels_list = [item.labels for item in batch]
        ids = [item.id for item in batch] # Lưu lại ID để tracking nếu cần

        padded_src = pad_sequence(
            src_ids_list, 
            batch_first=True, 
            padding_value=self.padding_idx
        )

        padded_decoder_input = pad_sequence(
            decoder_input_ids_list, 
            batch_first=True, 
            padding_value=self.padding_idx
        )

        padded_labels = pad_sequence(
            labels_list, 
            batch_first=True, 
            padding_value=self.padding_idx
        )

        return {
            "id": ids,
            "src": padded_src,                 # (B, Max_Len_Src, 4)
            "decoder_input": padded_decoder_input, # (B, Max_Len_Tgt, 4)
            "labels": padded_labels            # (B, Max_Len_Tgt, 4)
        }