from torch.utils.data import Dataset
import json
import torch

from utils.instance import Instance
from vocabs.vocab import Vocab
from vocabs.utils import preprocess_sentence

class ViTextSumDataset(Dataset):
    def __init__(self, config, vocab: Vocab) -> None:
        super().__init__()

        path: str = config.path
        # Load file JSON
        self._data = json.load(open(path, encoding='utf-8'))
        self._keys = list(self._data.keys())
        self._vocab = vocab
        
        # ⚠️ QUAN TRỌNG: Lưu lại độ dài tối đa từ config (ví dụ: 256)
        self.max_len = config.MAX_LEN 

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> Instance:
        key = self._keys[index]
        item = self._data[key]
        
        # --- 1. XỬ LÝ SOURCE (Văn bản nguồn) ---
        raw_source = item["source"]
        
        # Xử lý trường hợp source là Dictionary (nối các đoạn văn) hoặc String
        if isinstance(raw_source, dict):
            # Nối các value trong dict lại với nhau
            source_text = " ".join([" ".join(p) for _, p in raw_source.items()])
        else:
            source_text = str(raw_source)

        source_words = preprocess_sentence(source_text)
        
        # Encode: List[str] -> Tensor (Seq_Len_Src, 4)
        encoded_source = self._vocab.encode_caption(source_words)

        # --- 2. XỬ LÝ TARGET (Tóm tắt) ---
        target_text = item["target"]
        target_words = preprocess_sentence(target_text)
        
        # Encode: List[str] -> Tensor (Seq_Len_Tgt, 4)
        encoded_full_target = self._vocab.encode_caption(target_words)

        # ===========================================================
        # ✂️ CẮT NGẮN DỮ LIỆU (TRUNCATION) - FIX LỖI RUNTIME ERROR
        # ===========================================================
        
        # Nếu câu nguồn dài hơn giới hạn mô hình (ví dụ > 256), ta cắt bớt phần đuôi
        if encoded_source.shape[0] > self.max_len:
            encoded_source = encoded_source[:self.max_len, :]
            
        # Nếu câu đích dài hơn giới hạn, cũng cắt bớt
        if encoded_full_target.shape[0] > self.max_len:
            encoded_full_target = encoded_full_target[:self.max_len, :]
            
        # ===========================================================

        # --- 3. TÁCH INPUT/LABEL CHO DECODER ---
        
        # Decoder Input: Bỏ token cuối cùng
        decoder_input_ids = encoded_full_target[:-1] 

        # Label: Bỏ token đầu tiên (<bos>)
        labels = encoded_full_target[1:]

        # --- 4. TRẢ VỀ INSTANCE ---
        return Instance(
            id=key,
            src_ids=encoded_source,          # (L_src, 4)
            decoder_input_ids=decoder_input_ids, # (L_tgt-1, 4)
            labels=labels                    # (L_tgt-1, 4)
        )