# ♟️ ChessMC – Neural MCTS-based Chess Engine

**ChessMC** là một công cụ chơi cờ vua sử dụng thuật toán **Monte Carlo Tree Search (MCTS)** được hướng dẫn bởi **mạng nơ-ron học sâu**, thay vì phương pháp alpha-beta truyền thống. Hệ thống được thiết kế theo tinh thần của **AlphaZero**, sử dụng deep learning để đánh giá vị trí bàn cờ thay vì các nước đi ngẫu nhiên.

---

## 🧠 Mục tiêu

- Dự đoán nước đi hợp lý dựa trên cấu trúc bàn cờ.
- Sử dụng mạng nơ-ron CNN hoặc Transformer để ước lượng:
  - **Value** (giá trị lợi thế từ -1 đến +1)
  - **Policy** (phân phối xác suất nước đi)
  - **Class** (thắng/hòa/thua)
- Tích hợp với thuật toán MCTS để nâng cao khả năng tìm kiếm nước đi.

---

## 📂 Dữ liệu

- Nguồn dữ liệu: [PGN Mentor Dataset](https://www.pgnmentor.com/files.html)
- Dữ liệu từ các ván cờ `.pgn` được xử lý thành tensor đầu vào 18×8×8, bao gồm:
  - 12 kênh quân cờ
  - 1 kênh lượt chơi
  - 4 kênh nhập thành
  - 1 kênh en passant

- Nhãn đầu ra:
  - `value`: chuẩn hóa centipawn từ Stockfish
  - `policy`: chỉ số nước đi (0–4095)
  - `class`: kết quả ván cờ (0 thắng, 1 thua, 2 hòa)

---

## 🏗️ Kiến trúc mô hình

### ✅ Improved CNN (Mặc định)
- Input: 18 × 8 × 8
- Residual blocks + Attention
- Dual-head output:
  - `value` ∈ [-1, 1]
  - `policy` ∈ [0, 1]^4096
  - `class` ∈ {0, 1, 2}

### ✅ Transformer (Tùy chọn)
- Mỗi ô cờ là một token
- Positional encoding
- Global pooling đầu ra

### Loss:
- CrossEntropy cho policy
- MSE cho value
- Có hỗ trợ label smoothing và FP16

---

## 🔍 MCTS nâng cao

- Dùng `policy` để hướng dẫn mở rộng node
- **Progressive widening**: giới hạn số node con
- **Dirichlet noise**: tăng độ đa dạng cho root node
- **Virtual loss**: hỗ trợ song song
- Tích hợp mạng nơ-ron để đánh giá `value` và `policy` tại mỗi node

---

## 📊 Đánh giá

- Sử dụng `checkstep.py` để so sánh nước đi AI với nước tốt nhất của **Stockfish** (ở độ sâu 15).
- Tính toán **centipawn loss** cho từng nước đi.
- Vẽ biểu đồ phân bố tổn thất để đánh giá "độ thông minh" của AI.

---

## 🌐 Demo giao diện web

- Flask Web UI hỗ trợ:
  - Reset ván cờ
  - Người đấu AI
  - Chế độ self-play
- Bàn cờ hiển thị bằng **SVG base64**

![demo](https://user-images.githubusercontent.com/54076398/123994421-a7b34980-d9cd-11eb-8ef9-7e2174e5c09f.png)

---

## ▶️ Hướng dẫn sử dụng

```bash
# Bước 1: Chuẩn bị dữ liệu
python convert_pgn_to_npz.py  # Tạo file train.npz, val.npz

# Bước 2: Huấn luyện mô hình
python trainer.py  # Model được lưu trong thư mục /models

# Bước 3: Khởi chạy giao diện web
python main.py  # Truy cập http://localhost:5000
