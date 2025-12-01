import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import importlib.util 
# Giả định các import sau là đúng
from vocab.text_sum_dataset_phoneme import ViTextSumDataset
from collate_fn.collate_fn_phoneme import ViCollator
from vocabs.viword_vocab import ViWordVocab 
from configs.phoneme_config import Config 
from models.transformer_phoneneme import ViSeq2SeqTransformer
from losses.phoneneme_loss import PhonemeLoss


def initialize_components(config: Config) -> tuple:
    """Khởi tạo Vocab, cập nhật config, và khởi tạo Model."""
    print("Đang xây dựng từ điển...")
    vocab_obj = ViWordVocab(config)
    
    # Cập nhật VOCAB_SIZE
    config.VOCAB_SIZE = len(vocab_obj.itos)
    print(f"Vocab Size: {config.VOCAB_SIZE}")

    print("Đang khởi tạo Model...")
    model = ViSeq2SeqTransformer(
        vocab_size=config.VOCAB_SIZE, 
        d_model=config.D_MODEL,
        nhead=config.N_HEAD,
        num_encoder_layers=config.NUM_ENCODER_LAYERS,
        num_decoder_layers=config.NUM_DECODER_LAYERS,
        dim_feedforward=config.DIM_FEEDFORWARD,
        max_len=config.MAX_LEN,
        device=config.DEVICE,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    criterion = PhonemeLoss(padding_idx=vocab_obj.padding_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    return vocab_obj, model, criterion, optimizer

def train_model(config: Config, vocab_obj: ViWordVocab, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Hàm huấn luyện mô hình.
    """
    # Cài đặt đường dẫn train và DataLoader
    config.path = config.TRAIN 
    print(f"Đang tải dữ liệu Train từ: {config.path}")
    train_dataset = ViTextSumDataset(config, vocab_obj)
    collator = ViCollator(padding_idx=vocab_obj.padding_idx)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=collator
    )

    # Bắt đầu vòng lặp huấn luyện
    print("Bắt đầu huấn luyện...")
    model.train()
    
    for epoch in range(config.NUM_EPOCHS):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}", unit="batch")
        total_loss = 0
        
        for batch in progress_bar:
            # Chuyển dữ liệu sang GPU/CPU
            src = batch["src"].to(config.DEVICE)
            tgt_input = batch["decoder_input"].to(config.DEVICE) 
            labels = batch["labels"].to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(src, tgt_input) # (B, Tgt_Len, 4, Vocab_Size)
            
            # Tính Loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Cập nhật thông tin
            current_loss = loss.item()
            total_loss += current_loss
            progress_bar.set_postfix(loss=f"{current_loss:.4f}")
        
        # In loss trung bình của cả epoch
        avg_loss = total_loss / len(train_loader)
        print(f"✅ Kết thúc Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")
        
        # Lưu checkpoint
        # Sử dụng biến CHECKPOINT_PATH từ config
        torch.save(model.state_dict(), f"{config.CHECKPOINT_PATH}_{epoch+1}.pt")


def evaluate_model(config: Config, vocab_obj: ViWordVocab, model: nn.Module, criterion: nn.Module, data_path: str) -> float:
    """
    Hàm đánh giá mô hình trên tập DEV/TEST.
    Trả về Average Loss.
    """
    # Cài đặt đường dẫn và DataLoader
    config.path = data_path
    print(f"Đang tải dữ liệu Đánh giá từ: {config.path}")
    eval_dataset = ViTextSumDataset(config, vocab_obj)
    collator = ViCollator(padding_idx=vocab_obj.padding_idx)
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, # Không cần shuffle khi đánh giá
        num_workers=2, 
        collate_fn=collator
    )

    model.eval() # Chuyển sang chế độ đánh giá
    total_loss = 0
    
    with torch.no_grad():
        progress_bar = tqdm(eval_loader, desc=f"Đánh giá ({'DEV' if 'dev' in data_path.lower() else 'TEST'})", unit="batch")
        for batch in progress_bar:
            src = batch["src"].to(config.DEVICE)
            tgt_input = batch["decoder_input"].to(config.DEVICE)
            labels = batch["labels"].to(config.DEVICE)
            
            # Forward pass
            outputs = model(src, tgt_input)
            
            # Tính Loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(eval_loader)
    print(f"✨ Đánh giá hoàn tất | Average Loss: {avg_loss:.4f}")
    return avg_loss

def generate_summary(config: Config, vocab_obj: ViWordVocab, model: nn.Module, source_text: str, max_len: int = 50) -> str:

    model.eval()
    
    # 1. Mã hóa văn bản nguồn thành tensor (Source: List[str] -> Tensor (1, Src_Len, 4))
    words = vocab_obj.preprocess_sentence(source_text)
    src_vec = vocab_obj.encode_caption(words).unsqueeze(0).to(config.DEVICE) # Thêm dimension Batch
    
    # 2. Khởi tạo đầu vào cho Decoder
    # Bắt đầu với token BOS: (1, 1, 4) -> BOS + 3 PAD
    start_token = (vocab_obj.bos_idx, vocab_obj.padding_idx, vocab_obj.padding_idx, vocab_obj.padding_idx)
    tgt_tokens = torch.tensor(start_token).long().unsqueeze(0).unsqueeze(0).to(config.DEVICE) # (1, 1, 4)

    # 3. Vòng lặp sinh câu (Greedy Search)
    for _ in range(max_len):
        # outputs: (1, current_len, 4, Vocab_Size)
        with torch.no_grad():
            outputs = model(src_vec, tgt_tokens)
        
        # Lấy token cuối cùng được dự đoán: (1, 4, Vocab_Size)
        last_prediction = outputs[:, -1, :, :] 
        
        # Tìm index của phoneme có xác suất cao nhất cho 4 thành phần (Onset, Medial, Nucleus, Coda)
        # predicted_phoneme_ids: (1, 4)
        predicted_phoneme_ids = last_prediction.argmax(dim=-1) 
        
        # Nếu thành phần Onset là EOS_ID (tương đương với một từ được dự đoán là EOS) -> Kết thúc
        if predicted_phoneme_ids[0, 0].item() == vocab_obj.eos_idx:
            break
            
        # Nối kết quả dự đoán vào đầu vào của decoder cho bước tiếp theo
        # predicted_phoneme_ids có shape (1, 4), cần reshape thành (1, 1, 4) để concatenate
        tgt_tokens = torch.cat([tgt_tokens, predicted_phoneme_ids.unsqueeze(1)], dim=1)
        
    summary_vec = tgt_tokens.squeeze(0).cpu() # (Tgt_Len, 4)
    summary_text = vocab_obj.decode_caption(summary_vec, join_words=True)
    
    return summary_text


def load_config_from_file(config_name: str):
    """Nạp lớp Config từ file Python được chỉ định (ví dụ: 'config_large')."""
    # 1. Xây dựng đường dẫn file: config_name.py
    spec = importlib.util.spec_from_file_location("config_module", f"{config_name}.py")
    
    if spec is None:
        raise FileNotFoundError(f"Không tìm thấy file config: {config_name}.py")
        
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # 2. Lấy lớp Config từ module đã nạp
    return config_module.Config()

def main():
    # Bước 0: Thiết lập Argument Parser
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình Text Summarization.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/phoneme_config", # Giả định đây là giá trị mặc định đúng của bạn
        help="Tên file cấu hình (bao gồm cả đường dẫn tương đối, không bao gồm phần mở rộng .py). Ví dụ: 'configs/phoneme_config'"
    )
    args = parser.parse_args()
    
    try:
        # Nạp và khởi tạo Config từ tham số dòng lệnh
        config = load_config_from_file(args.config)
        
        # Thêm thuộc tính lưu checkpoint mặc định vào config
        if not hasattr(config, 'CHECKPOINT_PATH'):
            config.CHECKPOINT_PATH = "checkpoint_epoch" 

        # 1. Khởi tạo các thành phần
        vocab_obj, model, criterion, optimizer = initialize_components(config)
        
        # 2. Huấn luyện mô hình
        train_model(config, vocab_obj, model, criterion, optimizer)
        
        # 3. Đánh giá mô hình trên tập DEV
        print("\n" + "="*50)
        print("Bắt đầu Đánh giá trên tập DEV")
        evaluate_model(config, vocab_obj, model, criterion, config.DEV)
        print("="*50 + "\n")

        # 4. Ví dụ sinh tóm tắt
        sample_text = "Hôm nay, thời tiết tại thành phố Hồ Chí Minh rất đẹp, nắng vàng rực rỡ và không khí trong lành, rất thích hợp cho các hoạt động ngoài trời."
        print("🔍 Ví dụ Sinh Tóm Tắt (Inference)")
        
        # Tải checkpoint tốt nhất (hoặc cuối cùng)
        try:
            checkpoint_file = f"{config.CHECKPOINT_PATH}_{config.NUM_EPOCHS}.pt"
            model.load_state_dict(torch.load(checkpoint_file))
            print(f"✅ Đã tải checkpoint: {checkpoint_file}")
        except FileNotFoundError:
            # Sửa lỗi: In ra thông báo rõ ràng khi không tìm thấy file
            print(f"⚠️ Không tìm thấy file checkpoint ({checkpoint_file}). Dùng model cuối cùng trong bộ nhớ.")
        except Exception as e:
            # Sửa lỗi: In ra lỗi nếu có vấn đề khác khi tải state_dict
            print(f"❌ Lỗi khi tải checkpoint: {e}")
        
        # Sửa lỗi: generate_summary không còn bị chặn bởi khối try...except lớn nữa
        summary = generate_summary(config, vocab_obj, model, sample_text)
        print(f"Văn bản gốc: {sample_text}")
        print(f"Tóm tắt: {summary}")
        print("="*50)
        
        # 5. Đánh giá cuối cùng trên tập TEST
        print("\n" + "!"*50)
        print("TIẾN HÀNH ĐÁNH GIÁ CUỐI CÙNG TRÊN TẬP TEST")
        # Sửa lỗi: evaluate_model trên TEST không còn bị chặn nữa
        evaluate_model(config, vocab_obj, model, criterion, config.TEST)
        print("!"*50)

    except FileNotFoundError as e:
        # Sửa lỗi: In ra tên file bị thiếu rõ ràng
        print(f"LỖI KHỞI TẠO: Không tìm thấy file. {e}. Vui lòng kiểm tra tên file config và đường dẫn.")
    except AttributeError as e: 
        # Sửa lỗi: In ra lỗi thuộc tính gốc để gỡ lỗi chính xác
        print(f"LỖI CẤU HÌNH: Thiếu thuộc tính cần thiết trong lớp Config. Lỗi gốc: {e}. Vui lòng kiểm tra file config.")
    except Exception as e:
        print(f"LỖI KHÔNG XÁC ĐỊNH: {e}")


if __name__ == '__main__':
    main()