import os
import chess.pgn

DATA_DIR = "Data"  # thư mục chứa các file .pgn

total_games = 0
total_positions = 0
skip_moves = 5

for file in os.listdir(DATA_DIR):
    if not file.endswith(".pgn"):
        continue
    with open(os.path.join(DATA_DIR, file), encoding="utf-8", errors="ignore") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            total_games += 1
            board = game.board()
            for idx, move in enumerate(game.mainline_moves()):
                if idx >= skip_moves:
                    total_positions += 1
                board.push(move)

print(f"Tổng số ván cờ: {total_games}")
print(f"Số vị trí sau khi bỏ 5 nước đầu: {total_positions}")
