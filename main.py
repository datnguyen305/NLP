import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import importlib.util
import os

# Import các module của bạn (Hãy điều chỉnh đường dẫn import nếu khác)
from vocab.text_sum_dataset import TextSumDataset      # Class Dataset chuẩn
from collate_fn.collate_fn import Collator         # Class Collator chuẩn
from vocab.vocab import Vocab                   # Class Vocab chuẩn
from models.transformer import Seq2SeqTransformer # Model Transformer chuẩn
from losses.loss import TextSumLoss             # Loss Function chuẩn

# Nếu file config nằm ở configs/config.py
# from configs.config import Config 

def load_config_from_file(config_name: str):
    """Nạp lớp Config từ file Python được chỉ định."""
    file_path = f"configs/{config_name}.py"
    if not os.path.exists(file_path):
         # Fallback nếu để file config cùng cấp
         file_path = f"{config_name}.py"
         
    spec = importlib.util.spec_from_file_location("config_module", file_path)
    if spec is None:
        raise FileNotFoundError(f"Không tìm thấy file config: {file_path}")
        
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.Config()

def initialize_components(config) -> tuple:
    """Khởi tạo Vocab, cập nhật config, và khởi tạo Model."""
    print("🛠 Đang xây dựng từ điển (Vocab)...")
    vocab_obj = Vocab(config)
    
    # Cập nhật VOCAB_SIZE vào config để Model dùng
    config.VOCAB_SIZE = vocab_obj.vocab_size
    print(f"✅ Vocab Size: {config.VOCAB_SIZE}")

    print("🏗 Đang khởi tạo Model Transformer...")
    model = Seq2SeqTransformer(config, vocab_obj).to(config.DEVICE)
    
    # Khởi tạo Loss function (có label smoothing)
    criterion = TextSumLoss(pad_idx=vocab_obj.pad_idx, label_smoothing=0.1).to(config.DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)
    
    return vocab_obj, model, criterion, optimizer

def train_model(config, vocab_obj, model, criterion, optimizer):
    """
    Hàm huấn luyện mô hình.
    """
    # 1. Setup DataLoader
    config.path = config.TRAIN 
    print(f"📂 Đang tải dữ liệu Train từ: {config.path}")
    
    train_dataset = TextSumDataset(config, vocab_obj)
    collator = Collator(pad_idx=vocab_obj.pad_idx)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collator
    )

    print("🚀 Bắt đầu huấn luyện...")
    model.train()
    
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", unit="batch")
        total_loss = 0
        
        for batch in progress_bar:
            # Lấy dữ liệu từ batch
            # Giả sử batch trả về keys: 'input_ids' và 'label'
            src = batch["input_ids"].to(config.DEVICE)   # (Batch, Src_Len)
            trg = batch["label"].to(config.DEVICE)       # (Batch, Trg_Len) -> Gồm <BOS>...<EOS>

            # --- Xử lý Shifted Target cho Transformer ---
            # Decoder Input: Bỏ token cuối (<EOS>) -> [<BOS>, A, B, C]
            tgt_input = trg[:, :-1]
            
            # Target Label: Bỏ token đầu (<BOS>) -> [A, B, C, <EOS>]
            tgt_output = trg[:, 1:]

            optimizer.zero_grad()
            
            # Forward pass
            # Model nhận src và tgt_input
            logits = model(src, tgt_input) # (Batch, Seq_Len, Vocab_Size)
            
            # Tính Loss
            loss = criterion(logits, tgt_output)
            
            # Backward pass
            loss.backward()
            
            # Clip grad norm để tránh bùng nổ gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Update progress bar
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")
        
        # Kết thúc Epoch
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Kết thúc Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Lưu checkpoint
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")
        torch.save(model.state_dict(), f"checkpoints/checkpoint_epoch_{epoch+1}.pt")

def evaluate_model(config, vocab_obj, model, criterion, data_path: str) -> float:
    """
    Hàm đánh giá mô hình trên tập DEV/TEST.
    """
    # Setup DataLoader
    config.path = data_path
    print(f"📂 Đang tải dữ liệu Đánh giá từ: {config.path}")
    
    eval_dataset = TextSumDataset(config, vocab_obj)
    collator = Collator(pad_idx=vocab_obj.pad_idx)
    
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=2, 
        collate_fn=collator
    )

    model.eval() 
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc=f"Evaluating", unit="batch")
        for batch in progress_bar:
            src = batch["input_ids"].to(config.DEVICE)
            trg = batch["label"].to(config.DEVICE)
            
            tgt_input = trg[:, :-1]
            tgt_output = trg[:, 1:]
            
            logits = model(src, tgt_input)
            
            loss = criterion(logits, tgt_output)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(eval_loader)
    print(f"✨ Đánh giá hoàn tất | Average Loss: {avg_loss:.4f}")
    return avg_loss

def generate_summary(config, vocab_obj, model, source_text: str, max_len: int = 100) -> str:
    """
    Hàm sinh tóm tắt sử dụng Greedy Search (Đã đơn giản hóa cho standard Transformer).
    """
    model.eval()
    
    # 1. Mã hóa văn bản
    # encode_sentence trả về tensor 1D, cần thêm batch dim -> (1, Seq_Len)
    src_tensor = vocab_obj.encode_sentence(source_text).unsqueeze(0).to(config.DEVICE)
    
    # 2. Gọi hàm predict của model (Greedy Search)
    # Hàm này trả về tensor token IDs (không bao gồm BOS)
    with torch.no_grad():
        # Lưu ý: Cần đảm bảo class Model của bạn có hàm `predict` như tôi đã cung cấp ở comment trước
        output_tensor = model.predict(src_tensor, max_len=max_len)
    
    # 3. Giải mã về text
    # decode_sentence nhận batch -> unsqueeze(0)
    summary = vocab_obj.decode_sentence(output_tensor.unsqueeze(0), join_words=True)
    
    return summary[0]

def main():
    # Bước 0: Thiết lập Argument Parser
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Transformer Summarization.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config", 
        help="Tên file cấu hình (ví dụ: 'config' cho file config.py)"
    )
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"], help="Chế độ chạy")
    args = parser.parse_args()
    
    try:
        # Nạp Config
        config = load_config_from_file(args.config)
        
        # 1. Khởi tạo components
        vocab_obj, model, criterion, optimizer = initialize_components(config)
        
        if args.mode == "train":
            # 2. Huấn luyện
            train_model(config, vocab_obj, model, criterion, optimizer)
            
            # 3. Đánh giá trên DEV
            print("\n" + "="*50)
            print("🔍 Bắt đầu Đánh giá trên tập DEV")
            evaluate_model(config, vocab_obj, model, criterion, config.DEV)
            print("="*50 + "\n")

            # 4. Test thử 1 câu
            sample_text = "Trí tuệ nhân tạo đang thay đổi thế giới một cách nhanh chóng thông qua các mô hình ngôn ngữ lớn."
            print("📝 Ví dụ Sinh Tóm Tắt (Sau khi train):")
            summary = generate_summary(config, vocab_obj, model, sample_text)
            print(f"Gốc: {sample_text}")
            print(f"Tóm tắt: {summary}")
            
        elif args.mode == "inference":
            # Load checkpoint để test
            checkpoint_path = "checkpoints/checkpoint_epoch_10.pt" # Ví dụ
            if os.path.exists(checkpoint_path):
                print(f"Load checkpoint: {checkpoint_path}")
                model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
            
            text = input("Nhập văn bản cần tóm tắt: ")
            summary = generate_summary(config, vocab_obj, model, text)
            print(f"Tóm tắt: {summary}")

    except FileNotFoundError as e:
        print(f"❌ Lỗi File: {e}")
    except Exception as e:
        print(f"❌ Lỗi không mong muốn: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()