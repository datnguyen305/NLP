import torch

class Config:
    # ===========================
    # 1. PATHS
    # ===========================
    # Base folder (để tham khảo nếu cần)
    BASE_DIR = "/kaggle/input/mynlpdataset/datasets/Wikilingual-dataset/"
    
    # Các file dữ liệu cụ thể
    TRAIN = BASE_DIR + "train.json"
    DEV   = BASE_DIR + "dev.json"
    TEST  = BASE_DIR + "test.json"
    JSON_PATH = [TRAIN, DEV, TEST]
    # Biến 'path' dùng để load Dataset hiện tại. 
    # Ban đầu để rỗng, ta sẽ gán giá trị (TRAIN/DEV) trong main.py
    path = "TEST" 
    CHECKPOINT_PATH = "checkpoint_epoch"
    # ===========================
    # 2. TOKENIZER & SPECIAL TOKENS
    # ===========================
    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    TOKENIZER = None

    # ===========================
    # 3. MODEL PARAMS
    # ===========================
    D_MODEL = 512 
    N_HEAD = 8 
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    DIM_FEEDFORWARD = 2048
    DROPOUT = 0.1
    MAX_LEN = 256
    
    # ===========================
    # 4. TRAINING PARAMS
    # ===========================
    BATCH_SIZE = 32
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.15
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")