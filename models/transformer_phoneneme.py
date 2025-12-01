import torch
import torch.nn as nn
import math

class TransformerPhonemeEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 512, padding_idx: int = 0, dropout_rate: float = 0.1):
        """
        Khởi tạo lớp Embedding phù hợp với Transformer.

        Args:
            vocab_size (int): Kích thước từ điển của tất cả các âm vị.
            d_model (int): Kích thước cuối cùng của vector đầu ra (kích thước ẩn của Transformer).
                           Phải chia hết cho 4 nếu dùng phương pháp Concatenate.
            max_len (int): Chiều dài tối đa của câu.
            padding_idx (int): Chỉ số của token PAD (thường là 0).
            dropout_rate (float): Tỷ lệ Dropout.
        """
        super().__init__()
        
        # 1. Phoneme Embedding Setup
        # Kích thước embedding cho mỗi thành phần âm vị
        # Giả sử ta dùng Concatenate, mỗi thành phần sẽ có kích thước d_model / 4
        assert d_model % 4 == 0, "d_model phải chia hết cho 4 cho Concatenate."
        self.phoneme_embed_dim = d_model // 4
        
        self.onset_embed = nn.Embedding(vocab_size, self.phoneme_embed_dim, padding_idx=padding_idx)
        self.medial_embed = nn.Embedding(vocab_size, self.phoneme_embed_dim, padding_idx=padding_idx)
        self.nucleus_embed = nn.Embedding(vocab_size, self.phoneme_embed_dim, padding_idx=padding_idx)
        self.coda_embed = nn.Embedding(vocab_size, self.phoneme_embed_dim, padding_idx=padding_idx)
        
        self.d_model = d_model
        
        # 2. Positional Encoding Setup (Được học)
        # Tốt hơn là dùng Positional Encoding cố định (như Transformer gốc)
        self.pos_encoder = self._get_fixed_positional_encoding(d_model, max_len)
        
        # 3. Dropout
        self.dropout = nn.Dropout(dropout_rate)


    def _get_fixed_positional_encoding(self, d_model: int, max_len: int) -> nn.Parameter:
        """
        Tạo Positional Encoding cố định (dạng sin/cos) theo kiến trúc Transformer gốc.
        """
        # Tạo ma trận PE (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Thêm một chiều Batch, biến thành Parameter để nó được lưu trữ (nhưng không được học)
        pe = pe.unsqueeze(0) 
        return nn.Parameter(pe, requires_grad=False)
        

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tensor: Tensor có shape (Batch_Size, Seq_Len, 4).
                          4 cột là ID của Onset, Medial, Nucleus, Coda.
                          
        Returns:
            Tensor có shape (Batch_Size, Seq_Len, d_model) sẵn sàng cho Decoder/Encoder.
        """
        B, L, _ = input_tensor.shape
        
        # 1. PHONEME EMBEDDING (Học Biểu diễn Âm vị)
        
        # Tách input tensor thành 4 tensor riêng biệt (chỉ số 0, 1, 2, 3)
        onset_ids = input_tensor[..., 0]
        medial_ids = input_tensor[..., 1]
        nucleus_ids = input_tensor[..., 2]
        coda_ids = input_tensor[..., 3]
        
        # Lấy Embedding cho từng thành phần
        onset_embedded = self.onset_embed(onset_ids)      
        medial_embedded = self.medial_embed(medial_ids)    
        nucleus_embedded = self.nucleus_embed(nucleus_ids)  
        coda_embedded = self.coda_embed(coda_ids)          
        
        # CONCATENATE 4 vector lại (B, L, d_model)
        phoneme_embedding = torch.cat(
            (onset_embedded, medial_embedded, nucleus_embedded, coda_embedded), 
            dim=-1 
        )
        
        # 2. POSITIONAL ENCODING (Thêm Vị trí)
        # Lấy Positional Encoding cho chiều dài hiện tại L
        # 
        positional_encoding = self.pos_encoder[:, :L, :]
        
        # Cộng Positional Encoding vào Phoneme Embedding
        output = phoneme_embedding + positional_encoding
        
        # 3. DROPOUT
        return self.dropout(output)

class ViSeq2SeqTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len, device, dropout=0.1):
        super().__init__()
        
        self.device = device
        
        # 1. EMBEDDING & ENCODER
        # Embedding cho Source (Văn bản gốc)
        self.src_embedding = TransformerPhonemeEmbedding(vocab_size, d_model, max_len, padding_idx=0, dropout_rate=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # 2. DECODER (Đã bao gồm Embedding cho Target bên trong class PhonemeDecoder ta thiết kế trước đó)
        # Lưu ý: Chúng ta cần sửa lại PhonemeDecoder một chút để nó nhận embedding từ bên ngoài hoặc tự tạo.
        # Để tiện nhất, ta khai báo Embedding Target riêng ở đây cho đồng bộ.
        self.tgt_embedding = TransformerPhonemeEmbedding(vocab_size, d_model, max_len, padding_idx=0, dropout_rate=dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 3. GENERATOR HEADS (4 đầu ra)
        self.onset_head = nn.Linear(d_model, vocab_size)
        self.medial_head = nn.Linear(d_model, vocab_size)
        self.nucleus_head = nn.Linear(d_model, vocab_size)
        self.coda_head = nn.Linear(d_model, vocab_size)

    def create_padding_mask(self, tensor):
        """Tạo mask cho vị trí padding (Onset == 0)"""
        # tensor: (Batch, Seq_Len, 4) -> Mask: (Batch, Seq_Len)
        return (tensor[..., 0] == 0)

    def forward(self, src, tgt):
        """
        src: (Batch, Src_Len, 4)
        tgt: (Batch, Tgt_Len, 4) - Lưu ý: Đây là Decoder Input (đã bỏ <eos>)
        """
        
        # --- BƯỚC 1: TẠO MASK ---
        # Mask che padding cho Source và Target
        src_padding_mask = self.create_padding_mask(src).to(self.device)
        tgt_padding_mask = self.create_padding_mask(tgt).to(self.device)
        
        # Mask che tương lai cho Target (Causal Mask)
        tgt_seq_len = tgt.shape[1]
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, self.device)
        
        # --- BƯỚC 2: ENCODER ---
        # Embed Source
        src_emb = self.src_embedding(src) # (Batch, Src_Len, D_Model)
        
        # Qua Encoder
        memory = self.transformer_encoder(
            src=src_emb, 
            src_key_padding_mask=src_padding_mask
        )
        
        # --- BƯỚC 3: DECODER ---
        # Embed Target
        tgt_emb = self.tgt_embedding(tgt) # (Batch, Tgt_Len, D_Model)
        
        # Qua Decoder
        # memory là output của encoder
        dec_output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,                   # Che tương lai
            tgt_key_padding_mask=tgt_padding_mask, # Che padding của target
            memory_key_padding_mask=src_padding_mask # Che padding của memory (source)
        )
        
        # --- BƯỚC 4: DỰ ĐOÁN (4 Nhánh) ---
        logits_onset = self.onset_head(dec_output)
        logits_medial = self.medial_head(dec_output)
        logits_nucleus = self.nucleus_head(dec_output)
        logits_coda = self.coda_head(dec_output)
        
        return logits_onset, logits_medial, logits_nucleus, logits_coda

def generate_square_subsequent_mask(sz, device):
    """
    Tạo mask hình tam giác vuông để che các từ tương lai.
    Input: sz (độ dài câu tóm tắt)
    Output: Tensor (sz, sz) chứa 0 và -inf
    """
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

