import copy
import random
import os
import chess.engine
import traceback
import torch
from chessmc.state import State
from chessmc.utils import to_svg
from flask import Flask, request

# Import improved MCTS
try:
    from chessmc.mcts import improved_uct_search, create_advanced_chess_ai
    USE_NEURAL_AI = True
    print("Using Neural Network AI")
except ImportError:
    print("Neural AI not available, using random moves")
    USE_NEURAL_AI = False

app = Flask(__name__)
AI_COLOR    = chess.BLACK
HUMAN_COLOR = chess.WHITE
STATE = State()
engine = chess.engine.SimpleEngine.popen_uci("D:\\tool\\stockfish\\stockfish-windows-x86-64-avx2.exe")

# Tìm model mới nhất
MODEL_DIR = "models"
latest_model = None

if USE_NEURAL_AI and os.path.isdir(MODEL_DIR):
    # ưu tiên load file best.pth (hoặc {model_type}_best.pth)
    best_model_candidates = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pth") and "best" in f
    ]
    if best_model_candidates:
        # nếu có nhiều file có "best" (vd: cnn_best.pth vs transformer_best.pth), lọc lấy cnn_best nếu bạn chỉ train type 'cnn'
        # hoặc chọn file đầu tiên trong danh sách:
        best_model = sorted(best_model_candidates)[-1]
        latest_model = os.path.join(MODEL_DIR, best_model)
        print(f"Using best model: {latest_model}")
    else:
        # nếu không có file best, fallback lấy epoch cao nhất
        epoch_models = [
            f for f in os.listdir(MODEL_DIR)
            if f.endswith(".pth") and "epoch" in f
        ]
        if epoch_models:
            # tìm file có số epoch lớn nhất
            def extract_epoch(fname):
                # giả sử tên kiểu 'cnn_epoch12.pth', 'transformer_epoch5.pth'
                # tách theo "epoch", lấy phần nằm giữa 'epoch' và '.pth'
                try:
                    return int(fname.split("epoch")[1].split(".")[0])
                except:
                    return -1

            best_epoch_file = max(epoch_models, key=extract_epoch)
            latest_model = os.path.join(MODEL_DIR, best_epoch_file)
            print(f"No best.pth found, using latest epoch model: {latest_model}")
        else:
            print("No checkpoint files found in models/")
            latest_model = None

def random_move(state):
    return random.choice([move for move in state.legal_moves])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print('sử dụng device:', DEVICE)

def get_computer_move(state):
    """lấy nước đi của máy sử dụng gpu nếu có"""
    if USE_NEURAL_AI and latest_model and os.path.exists(latest_model):
        print("→ dùng mcts+nn với model:", latest_model, "trên device:", DEVICE)
        try:
            move = improved_uct_search(state, n_simulations=1000, model_path=latest_model, device=DEVICE)
            print(f"[DEBUG main] computer_move returned: {move}")
            return move
        except Exception as e:
            print("neural ai error:", e)
    print("→ fallback random move")
    return random_move(state)

@app.route("/")
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return content.replace('start', STATE.board.fen())

@app.route("/new-game")
def new_game():
    """Reset bàn cờ - nếu có player_color thì đổi màu, không thì giữ nguyên"""
    global AI_COLOR, HUMAN_COLOR
    
    player = request.args.get("player_color", "").lower()
    
    # Nếu có tham số player_color thì cập nhật màu
    if player == "white":
        HUMAN_COLOR = chess.WHITE
        AI_COLOR    = chess.BLACK
        print("Player is WHITE, AI is BLACK")
    elif player == "black":
        HUMAN_COLOR = chess.BLACK
        AI_COLOR    = chess.WHITE
        print("Player is BLACK, AI is WHITE")
    # Nếu không có tham số thì giữ nguyên màu hiện tại
    
    # Reset bàn cờ
    STATE.board.reset()
    
    # Nếu AI đi trước thì cho AI đi
    if STATE.board.turn == AI_COLOR:
        ai_mv = get_computer_move(STATE)
        if ai_mv in STATE.board.legal_moves:
            STATE.board.push(ai_mv)
            print(f"AI made opening move: {ai_mv}")
    
    return STATE.board.fen()

@app.route("/self-play")
def self_play():
    state = State()
    ret = '<html><head></head><body>'

    move_count = 0
    while not state.board.is_game_over() and move_count < 200:  # Limit moves to prevent infinite games
        if USE_NEURAL_AI and latest_model:
            try:
                move = improved_uct_search(state, n_simulations=1000, model_path=latest_model, device=DEVICE)
                print(f"[DEBUG main] computer_move returned: {move}")
                if move in state.board.legal_moves:
                    move_san = state.board.san(move)
                else:
                    move_san = state.board.san(random_move(state))
            except:
                move_san = state.board.san(random_move(state))
        else:
            move_san = state.board.san(random_move(state))
            
        state.board.push_san(move_san)
        ret += f'<p>Move {move_count + 1}: {move_san}</p>'
        ret += '<img width=400 height=400 src="data:image/svg+xml;base64,%s"></img><br/>' % to_svg(state)
        move_count += 1

    result = state.board.result()
    ret += f'<h2>Game Over: {result}</h2></body></html>'
    print(f"Self-play finished. Result: {result}")
    return ret

@app.route("/move")
def move():
    if STATE.board.is_game_over():
        return app.response_class(response=f"Game over! Result: {STATE.board.result()}", status=200)

    # Nếu đến lượt người chơi (cầm màu HUMAN_COLOR) thì đọc param và đẩy move
    if STATE.board.turn == HUMAN_COLOR:
        source    = int(request.args.get('from',  '-1'))
        target    = int(request.args.get('to',    '-1'))
        promotion = request.args.get('promotion') == 'true'
        move = chess.Move(source, target,
                          promotion=chess.QUEEN if promotion else None)
        if move in STATE.board.legal_moves:
            STATE.board.push(move)
            print(f"Human played: {move}")
        # nếu sau khi người chơi đi mà game over thì trả luôn
        if STATE.board.is_game_over():
            return app.response_class(response=f"Game over! Result: {STATE.board.result()}", status=200)

    # Nếu đến lượt AI (cầm màu AI_COLOR) thì gọi MCTS và đẩy
    if STATE.board.turn == AI_COLOR:
        try:
            comp_mv = get_computer_move(STATE)
            if comp_mv in STATE.board.legal_moves:
                # xử lý phong cấp nếu cần
                piece = STATE.board.piece_at(comp_mv.from_square)
                if piece and piece.piece_type == chess.PAWN and (
                   comp_mv.to_square <= 7 or comp_mv.to_square >= 56):
                    comp_mv.promotion = chess.QUEEN
                STATE.board.push(comp_mv)
                print(f"AI played: {comp_mv}")
        except Exception as e:
            print(f"[WARN] MCTS error: {e}, bỏ qua")
        # nếu sau khi AI đi mà game over thì trả
        if STATE.board.is_game_over():
            return app.response_class(response=f"Game over! Result: {STATE.board.result()}", status=200)

    # Cuối cùng trả FEN hiện tại
    return app.response_class(response=STATE.board.fen(), status=200)

if __name__ == '__main__':
    app.run(debug=True, port=5000)