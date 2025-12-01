import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class Seq2SeqTransformer(nn.Module):
    def __init__(self, config, vocab):
        super(Seq2SeqTransformer, self).__init__()
        
        # Lấy tham số từ config và vocab
        self.pad_idx = vocab.pad_idx
        self.vocab_size = vocab.vocab_size
        
        d_model = getattr(config, 'd_model', 512)
        nhead = getattr(config, 'nhead', 8)
        num_encoder_layers = getattr(config, 'num_encoder_layers', 6)
        num_decoder_layers = getattr(config, 'num_decoder_layers', 6)
        dim_feedforward = getattr(config, 'dim_feedforward', 2048)
        dropout = getattr(config, 'dropout', 0.1)
        
        # 1. Layers Embeddings & Positional Encoding
        self.src_tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_idx)
        self.tgt_tok_emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

        # 2. PyTorch Core Transformer Module
        # batch_first=True: Input/Output sẽ là (Batch, Seq, Feature) thay vì (Seq, Batch, Feature)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )

        # 3. Output Generator
        self.generator = nn.Linear(d_model, self.vocab_size)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        """
        src: (batch_size, src_len)
        tgt: (batch_size, tgt_len)
        """
        # Tạo masks cần thiết cho Transformer
        src_key_padding_mask, tgt_key_padding_mask, tgt_mask = self.create_masks(src, tgt)

        # Embedding + Scale + Positional Encoding
        # (Theo paper gốc thì embedding cần nhân với sqrt(d_model))
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        # Đưa vào nn.Transformer
        # Lưu ý: src_mask thường để None (trừ khi muốn che từ trong Encoder), 
        # quan trọng là tgt_mask (causal mask) và key_padding_mask (padding mask)
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            tgt_mask=tgt_mask, # Che tương lai của decoder
            src_key_padding_mask=src_key_padding_mask, # Che padding của src
            tgt_key_padding_mask=tgt_key_padding_mask, # Che padding của tgt
            memory_key_padding_mask=src_key_padding_mask # Che padding của src khi decoder attend vào
        )

        return self.generator(outs)

    def create_masks(self, src, tgt):
        batch_size, src_len = src.size()
        batch_size, tgt_len = tgt.size()

        # 1. Padding Mask (True ở vị trí là padding token) -> (Batch, Seq)
        src_key_padding_mask = (src == self.pad_idx)
        tgt_key_padding_mask = (tgt == self.pad_idx)

        # 2. Look-ahead Mask (Causal Mask) cho Decoder -> (Seq, Seq)
        # Ma trận tam giác trên chứa -inf để che các từ tương lai
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src.device)

        return src_key_padding_mask, tgt_key_padding_mask, tgt_mask

    def encode(self, src: torch.Tensor):
        """Hàm dùng cho Inference"""
        src_key_padding_mask = (src == self.pad_idx)
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_key_padding_mask=None):
        """Hàm dùng cho Inference"""
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))
        
        return self.transformer.decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
    def predict(self, src: torch.Tensor, max_len: int = 100):
        """
        Thực hiện Greedy Search cho 1 mẫu dữ liệu (Batch size = 1)
        
        Args:
            src: Tensor shape (src_len) hoặc (1, src_len)
            max_len: Độ dài tối đa của câu sinh ra
        """
        self.eval() # Chuyển sang chế độ eval
        device = src.device
        
        # 1. Đảm bảo input có dimension batch (1, src_len)
        if src.dim() == 1:
            src = src.unsqueeze(0)

        # 2. Encoder: Chỉ chạy 1 lần duy nhất để lấy Memory
        # src_key_padding_mask được xử lý bên trong hàm encode
        memory = self.encode(src)
        
        # 3. Khởi tạo Decoder Input với token <BOS>
        # ys sẽ chứa các token được sinh ra theo thời gian
        ys = torch.ones(1, 1).fill_(self.vocab.bos_idx).type(torch.long).to(device)
        
        # 4. Loop sinh từ
        for i in range(max_len):
            # a. Decoder Forward
            # memory_key_padding_mask: Với batch=1 và ko có padding ở src thì có thể để None
            # Tuy nhiên để an toàn ta lấy mask từ src (nếu src có pad)
            src_key_padding_mask = (src == self.pad_idx)
            
            output = self.decode(ys, memory, memory_key_padding_mask=src_key_padding_mask)
            
            # b. Generator: Lấy output tại vị trí cuối cùng
            # output shape: (1, seq_len, d_model) -> lấy token cuối: (1, d_model)
            prob = self.generator(output[:, -1])
            
            # c. Greedy: Chọn từ có xác suất cao nhất
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            # d. Thêm vào chuỗi kết quả
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
            
            # e. Điều kiện dừng: Gặp thẻ <EOS>
            if next_word == self.vocab.eos_idx:
                break
        
        # Trả về tensor chứa các token ID (bỏ token đầu tiên là BOS)
        return ys[0, 1:]
    
    def inference(text: str, model, vocab, max_len=100, device='cpu'):
        """
        Hàm tiện ích để tóm tắt một đoạn văn bản đầu vào.
        """
        model.eval()
        
        # 1. Tiền xử lý & Encode
        # encode_sentence trả về tensor 1D (seq_len)
        input_tensor = vocab.encode_sentence(text).to(device)
        
        # 2. Predict (Greedy Search)
        with torch.no_grad():
            output_tensor = model.predict(input_tensor, max_len)
        
        # 3. Decode về lại text
        # vocab.decode_sentence nhận input (batch, seq_len) nên cần unsqueeze
        decoded_sentence = vocab.decode_sentence(output_tensor.unsqueeze(0), join_words=True)
        
        return decoded_sentence[0]

