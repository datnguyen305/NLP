import torch

class Config:
    # ===========================
    # 1. PATHS (Đường dẫn dữ liệu)
    # ===========================
    BASE_DIR = "D:/NguyenTienDat_23520262/Nam_3/NLP/datasets/Wikilingual-dataset/"
    
    TRAIN = BASE_DIR + "train.json"
    DEV   = BASE_DIR + "dev.json"
    TEST  = BASE_DIR + "test.json"
    
    # Tạo object path giả lập để class Vocab có thể gọi config.path.train
    # (Khớp với code Vocab cũ: json_dirs = [config.path.train, ...])
    path = type('Path', (object,), {
        "train": TRAIN, 
        "dev": DEV, 
        "test": TEST
    })

    # ===========================
    # 2. TOKENIZER & SPECIAL TOKENS
    # ===========================
    # Định nghĩa token
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    
    # Mapping sang chữ thường (để Vocab gọi config.pad_token)
    pad_token = PAD_TOKEN
    bos_token = BOS_TOKEN
    eos_token = EOS_TOKEN
    unk_token = UNK_TOKEN
    
    # Tần suất xuất hiện tối thiểu để đưa vào từ điển
    min_freq = 1 

    # ===========================
    # 3. MODEL PARAMS (Tham số Transformer)
    # ===========================
    # Mapping sang chữ thường để Model gọi (config.d_model, config.nhead...)
    d_model = 512
    nhead = 8 
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    dropout = 0.1
    
    # Độ dài tối đa của câu (dùng cho Positional Encoding hoặc cắt chuỗi)
    max_len = 256 
    
    # ===========================
    # 4. TRAINING PARAMS
    # ===========================
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    
    # Transformer rất nhạy cảm với LR. 0.15 là quá lớn, nên dùng cỡ 1e-4 (0.0001)
    LEARNING_RATE = 0.0001 
    
    # Tự động nhận diện GPU/CPU
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===========================
    # 5. HELPER (Tự động cập nhật Vocab Size)
    # ===========================
    # Giá trị này sẽ được cập nhật trong main.py sau khi build Vocab
    VOCAB_SIZE = 0