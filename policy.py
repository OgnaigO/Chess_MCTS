import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import os
from torch.utils.data import DataLoader
from chessmc.data.dataset import ImprovedChessDataset
from chessmc.model import ImprovedChessModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_data_path = 'processed/val.npz'    # Đường dẫn tới tập validation
model_dir     = 'models'               # Thư mục chứa checkpoint
batch_size    = 32                     # Giảm xuống 16 hoặc 8 nếu thiếu bộ nhớ

# Các epoch bạn muốn đánh giá
allowed_epochs = {1, 2, 3, 8, 9, 20}

# 1. Load dataset validation
print(f"Đang load dữ liệu validation từ: {val_data_path}")
val_dataset = ImprovedChessDataset(val_data_path)
print(f"ImprovedChessDataset loaded: inputs {val_dataset.inputs.shape}, targets {val_dataset.targets.shape}")

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 2. Hàm đánh giá một checkpoint
def evaluate_checkpoint(model_path):
    """
    Load một checkpoint (.pth), đánh giá trên tập validation,
    và trả về (policy_accuracy, value_mse).
    """
    # 2.1 Tạo model và load weights
    model = ImprovedChessModel()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    total_correct    = 0
    total_samples    = 0
    total_value_mse  = 0.0
    mse_criterion    = torch.nn.MSELoss(reduction='sum')

    # 2.2 Duyệt toàn bộ val_loader
    with torch.no_grad():
        for boards, policy_tgt, value_tgt in val_loader:
            boards     = boards.to(device)
            policy_tgt = policy_tgt.to(device)
            value_tgt  = value_tgt.to(device)

            value_pred, policy_logits, _ = model(boards)

            # Tính policy accuracy
            preds = torch.argmax(policy_logits, dim=1)
            total_correct += (preds == policy_tgt).sum().item()

            # Tính value MSE (cộng dồn)
            batch_mse = mse_criterion(value_pred.view(-1), value_tgt)
            total_value_mse += batch_mse.item()

            total_samples += value_tgt.size(0)

    policy_accuracy = total_correct / total_samples * 100.0
    value_mse       = total_value_mse / total_samples
    return policy_accuracy, value_mse

# 3. Duyệt qua các checkpoint chỉ định
results = []
print("\nBắt đầu đánh giá các epoch:", sorted(allowed_epochs))
for filename in sorted(os.listdir(model_dir)):
    if filename.startswith("cnn_epoch") and filename.endswith(".pth"):
        epoch_num = int(filename.split("epoch")[1].split(".")[0])
        if epoch_num not in allowed_epochs:
            continue

        model_path = os.path.join(model_dir, filename)
        print(f"→ Đang đánh giá checkpoint epoch {epoch_num}: {filename}")
        acc, mse = evaluate_checkpoint(model_path)
        print(f"    Policy Acc: {acc:.2f}% | Value MSE: {mse:.4f}")
        results.append((epoch_num, acc, mse))

print("Hoàn tất đánh giá các epoch.\n")

# 4. In bảng tổng kết cho các epoch chọn lọc
results.sort(key=lambda x: x[0])
print(f"{'Epoch':>5} | {'Policy Acc (%)':>15} | {'Value MSE':>10}")
print("-" * 40)
for epoch, acc, mse in results:
    print(f"{epoch:>5} | {acc:>15.2f} | {mse:>10.4f}")
