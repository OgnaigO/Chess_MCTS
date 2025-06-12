import os
import numpy as np
import torch
import random
import chess
from tqdm import trange
from chessmc.state import State
from chessmc.mcts import create_advanced_chess_ai

# --- Thông số cấu hình (tuỳ chỉnh) ---
MODEL_PATH    = r"D:\\Chess\\chessmc\\models\\cnn_best.pth"  # đường dẫn đến model đã train
DEVICE        = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SIMULATIONS = 1000       # số vòng MCTS mỗi lượt
NUM_GAMES     = 100       # số ván self-play cần tạo
MAX_MOVES     = 200       # giới hạn nửa nước (half-moves) mỗi ván
OUTPUT_PATH   = "processed/selfplay.npz"  # nơi lưu dữ liệu đầu ra
# --------------------------------------

def serialize_state(state: State) -> np.ndarray:
    """
    Chuyển state (12×8×8) thành mảng float32 shape (12,8,8)
    từ state.serialize() (768-dim).
    """
    flat = state.serialize()        # mảng 768-dim float32
    return flat.reshape(12, 8, 8)

def policy_dict_to_vector(policy_dict: dict) -> np.ndarray:
    """
    Chuyển dict {move: prob} thành vector 4096-dim float32, đúng mapping:
      idx = from_square*64 + to_square + promotion_offset.
    """
    vec = np.zeros(4096, dtype=np.float32)
    for move, prob in policy_dict.items():
        idx = move.from_square * 64 + move.to_square
        if move.promotion:
            if move.promotion == chess.QUEEN:
                idx += 0
            elif move.promotion == chess.ROOK:
                idx += 1024
            elif move.promotion == chess.BISHOP:
                idx += 2048
            elif move.promotion == chess.KNIGHT:
                idx += 3072
        vec[min(idx, 4095)] = prob
    return vec

def self_play_game(ai, n_simulations: int, max_moves: int):
    """
    Chơi 1 ván giữa 2 phiên bản MCTS cùng model ai và trả về:
      - states_list: List[np.ndarray], mỗi phần tử là (12,8,8)
      - policy_list: List[np.ndarray], mỗi phần tử là (4096,)
      - z_list:      List[float], mỗi phần tử là value target (-1,0,1)
      - result_str:  str, '1-0', '0-1' hoặc '1/2-1/2'
    """
    state = State()
    states_list  = []
    policy_list  = []
    to_move_list = []
    move_count   = 0

    while not state.board.is_game_over() and move_count < max_moves:
        # 1) Gọi MCTS (có Dirichlet-noise tự thêm trong ai.search)
        mv, info = ai.search(state, n_simulations=n_simulations)
        probs_dict = info.get('probs', {})

        # 2) Lưu state và policy
        states_list.append(serialize_state(state))
        policy_list.append(policy_dict_to_vector(probs_dict))
        to_move_list.append(1.0 if state.board.turn == chess.WHITE else -1.0)

        # 3) Thực thi nước đi
        if mv not in state.board.legal_moves:
            mv = random.choice(list(state.board.legal_moves))
        state.board.push(mv)
        move_count += 1

    # 4) Xác định kết quả
    raw_result = state.board.result()  # '1-0', '0-1', '1/2-1/2' hoặc '*'
    if raw_result == '1-0':
        game_value = 1.0
        result_str = '1-0'
    elif raw_result == '0-1':
        game_value = -1.0
        result_str = '0-1'
    else:
        # '1/2-1/2' hoặc '*' (giới hạn max_moves) → hòa
        game_value = 0.0
        result_str = '1/2-1/2'

    # 5) Tạo z_list dựa vào lượt (to_move_list)
    z_list = [game_value * tm for tm in to_move_list]

    return states_list, policy_list, z_list, result_str

def main():
    print("Torch available:", torch.cuda.is_available(), "| Device:", DEVICE)
    # Khởi AI (AdvancedMCTS với Dirichlet-noise)
    ai = create_advanced_chess_ai(MODEL_PATH, DEVICE)

    all_states   = []
    all_policies = []
    all_values   = []
    results      = {'1-0': 0, '0-1': 0, '1/2-1/2': 0}

    for i in trange(NUM_GAMES, desc="Self-play games"):
        states, policies, values, res = self_play_game(ai, N_SIMULATIONS, MAX_MOVES)
        all_states.extend(states)
        all_policies.extend(policies)
        all_values.extend(values)
        results[res] += 1

    print("Kết quả self-play thống kê:", results)

    # 1) Chuyển sang array và flatten arr_0 từ (N,12,8,8) -> (N,768)
    X4 = np.stack(all_states, axis=0)  # shape (N, 12, 8, 8)
    N = X4.shape[0]
    X = X4.reshape(N, 12 * 8 * 8)       # shape (N, 768)

    # 2) Tạo Y shape (N,3): [value, policy_index, class]
    Y = np.zeros((N, 3), dtype=np.float32)
    for idx, (v, pi_vec) in enumerate(zip(all_values, all_policies)):
        Y[idx, 0] = v  # value target
        Y[idx, 1] = float(np.argmax(pi_vec))  # policy index
        if v > 0:
            Y[idx, 2] = 0  # White thắng
        elif v < 0:
            Y[idx, 2] = 1  # Black thắng
        else:
            Y[idx, 2] = 2  # Hòa

    # 3) Lưu file NPZ
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, arr_0=X, arr_1=Y)
    print(f"Saved self-play data: {N} samples → {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
