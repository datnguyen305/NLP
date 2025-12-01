from typing import List
from utils.instance import Instance, InstanceList
import torch
from torch.nn.utils.rnn import pad_sequence

class Collator:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Instance]):
        # 1. Tách các trường dữ liệu ra thành list riêng biệt
        # Giả sử trong Instance có thuộc tính input_ids và label là các List[int] hoặc Tensor 1D
        input_ids_list = [torch.tensor(item.input_ids) for item in batch]
        labels_list = [torch.tensor(item.label) for item in batch]
        
        # 2. Padding (Quan trọng!)
        # batch_first=True -> Output shape: (Batch_Size, Max_Len)
        input_ids_padded = pad_sequence(input_ids_list, batch_first=True, padding_value=self.pad_idx)
        labels_padded = pad_sequence(labels_list, batch_first=True, padding_value=self.pad_idx)
        
        # 3. Trả về dictionary chứa Tensor đã sẵn sàng cho model
        return {
            "input_ids": input_ids_padded, # Tensor
            "label": labels_padded         # Tensor
        }