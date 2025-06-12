import os
import numpy as np

def split_selfplay(
    input_path: str,
    output_folder: str,
    train_ratio: float = 0.8,
    seed: int = 2025
):
    """
    Tách file selfplay.npz thành train_sp.npz và val_sp.npz.

    input_path:  đường dẫn đến processed/selfplay.npz (4D đã được flatten ngay trong selfplay_data.py).
    output_folder: thư mục sẽ lưu kết quả (ví dụ "processed").
    train_ratio:  tỉ lệ chia cho train (mặc định 0.8).
    seed:         seed để shuffle.
    """
    data = np.load(input_path)
    X = data["arr_0"]  # shape = (9261, 768)
    Y = data["arr_1"]  # shape = (9261, 3)

    N = X.shape[0]
    np.random.seed(seed)
    perm = np.random.permutation(N)
    split_idx = int(train_ratio * N)

    idx_train = perm[:split_idx]
    idx_val   = perm[split_idx:]

    X_train = X[idx_train]
    Y_train = Y[idx_train]
    X_val   = X[idx_val]
    Y_val   = Y[idx_val]

    os.makedirs(output_folder, exist_ok=True)
    train_path = os.path.join(output_folder, "train_sp.npz")
    val_path   = os.path.join(output_folder, "val_sp.npz")

    np.savez_compressed(train_path, arr_0=X_train, arr_1=Y_train)
    np.savez_compressed(val_path,   arr_0=X_val,   arr_1=Y_val)

    print(f"Saved train: {X_train.shape[0]} samples → {train_path}")
    print(f"Saved  val: {X_val.shape[0]} samples → {val_path}")

if __name__ == "__main__":
    split_selfplay(
        input_path  = "processed/selfplay.npz",
        output_folder = "processed",
        train_ratio = 0.8,
        seed        = 2025
    )
