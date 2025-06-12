import os
import numpy as np
import chess.pgn
import chess.engine
from multiprocessing import Pool, Manager, cpu_count
from chessmc.state import State

# ==== CẤU HÌNH ====  
STOCKFISH_PATH = r"D:\\tool\\stockfish\\stockfish-windows-x86-64-avx2.exe"  
DATA_DIR = "Data"           
SAVE_DIR = "processed"      
FINAL_SAVE_PATH = os.path.join(SAVE_DIR, "full_dataset.npz")  
CHECKPOINT_INTERVAL = 10_000  
MAX_POSITIONS = 2_500_000     
ENGINE_DEPTH = 12             
VAL_RATIO = 0.2               

# Ánh xạ kết quả ván cờ  
RESULT_MAP = {'1-0': 0, '0-1': 1, '1/2-1/2': 2}


def board_to_input_vector(board):
    return State(board).serialize_cnn().astype(np.uint8)


def move_to_policy_index(move, board):
    """
    Chuyển chess.Move thành policy index (0-4095)
    Sử dụng encoding: from_square * 64 + to_square
    """
    # Basic encoding: from_square * 64 + to_square
    policy_idx = move.from_square * 64 + move.to_square
    
    # Handle promotion (simplified)
    if move.promotion:
        # Add offset for different promotion pieces
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1024, 
            chess.BISHOP: 2048,
            chess.KNIGHT: 3072
        }
        policy_idx += promotion_offset.get(move.promotion, 0)
    
    return min(policy_idx, 4095)  # Ensure within bounds


def normalize_centipawn_score(cp_score, max_cp=1000):
    """
    Cải thiện chuẩn hóa centipawn score
    """
    if cp_score is None:
        return 0.0
    
    # Clamp extreme values
    cp_clamped = max(-max_cp, min(max_cp, cp_score))
    
    # Sigmoid-like normalization
    normalized = 2.0 / (1.0 + np.exp(-cp_clamped / 200.0)) - 1.0
    return float(normalized)


def process_game_positions(game, engine, inputs, values, policies, classes, counter, lock):  
    """Xử lý một ván cờ - IMPROVED VERSION"""  
    board = game.board()  
    game_result = game.headers.get("Result")
    
    if game_result not in RESULT_MAP:
        return
        
    class_target = RESULT_MAP[game_result]
    
    for move_num, mv in enumerate(game.mainline_moves()):  
        if counter.value >= MAX_POSITIONS:  
            break  
            
        # Skip opening moves (first 5 moves) - thường không có much signal
        if move_num < 5:
            board.push(mv)
            continue
            
        legal_moves = list(board.legal_moves)  
        if mv not in legal_moves:  
            board.push(mv)
            continue  
            
        # Get board representation
        vec = board_to_input_vector(board)  
        
        # Get engine evaluation
        try:
            info = engine.analyse(board, chess.engine.Limit(depth=ENGINE_DEPTH))  
            if isinstance(info, list):  
                info = info[0]  
                
            score = info.get("score")  
            if score is None:  
                board.push(mv)
                continue  
                
            cp = score.white().score(mate_score=10000)  
            if cp is None:  
                board.push(mv)
                continue  
        except:
            board.push(mv)
            continue
            
        # Normalize value 
        v = normalize_centipawn_score(cp)
        
        # Adjust value based on turn (important!)
        # if not board.turn: 
        #     v = -v
            
        # Get policy target (FIXED)
        p = move_to_policy_index(mv, board)
        
        # Store data
        inputs.append(vec)  
        values.append(v)  
        policies.append(p)  
        classes.append(class_target)  
        
        # Update counter
        with lock:  
            counter.value += 1  
            
        board.push(mv)


def save_checkpoint(idx, pid, inputs, values, policies, classes):  
    """Lưu checkpoint với format cải thiện"""  
    os.makedirs(SAVE_DIR, exist_ok=True)  
    fname = f"ckpt_p{pid:02d}_{idx:04d}.npz"  
    path = os.path.join(SAVE_DIR, fname)  
    
    X = np.stack(inputs, axis=0)  
    
    # Stack targets properly: [value, policy, class]
    Y = np.column_stack([
        np.array(values, dtype=np.float32),
        np.array(policies, dtype=np.int32), 
        np.array(classes, dtype=np.int32)
    ])
    
    np.savez_compressed(path, arr_0=X, arr_1=Y)  
    print(f"[P{pid}] Saved {fname} ({len(X)} samples)")


def process_file(args):  
    """Worker function với better error handling"""  
    file_path, pid, counter, lock = args  
    inputs, values, policies, classes = [], [], [], []  
    ckpt_idx = 0  
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)  
        print(f"[P{pid}] Start {os.path.basename(file_path)}")  
        
        with open(file_path, encoding='utf-8', errors='ignore') as f:  
            while True:  
                if counter.value >= MAX_POSITIONS:  
                    break  
                    
                try:
                    game = chess.pgn.read_game(f)  
                    if game is None:  
                        break  
                        
                    process_game_positions(game, engine, inputs, values, policies, classes, counter, lock)  
                    
                    if len(inputs) >= CHECKPOINT_INTERVAL:  
                        save_checkpoint(ckpt_idx, pid, inputs, values, policies, classes)  
                        ckpt_idx += 1  
                        inputs.clear()
                        values.clear()
                        policies.clear() 
                        classes.clear()
                        
                except Exception as e:
                    print(f"[P{pid}] Error processing game: {e}")
                    continue
                    
    except Exception as e:  
        print(f"[P{pid}] ERROR processing {file_path}: {e}", flush=True)  
    finally:  
        try:
            engine.quit()  
        except:
            pass
            
        # Save remaining samples
        if inputs:  
            save_checkpoint(ckpt_idx, pid, inputs, values, policies, classes)  
            
        print(f"[P{pid}] Done {os.path.basename(file_path)}")


def merge_and_split():  
    """Gộp và chia dữ liệu với validation"""  
    all_X, all_Y = [], []  
    
    checkpoint_files = [f for f in os.listdir(SAVE_DIR) 
                       if f.startswith('ckpt_') and f.endswith('.npz')]
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    for fn in sorted(checkpoint_files):  
        try:
            data = np.load(os.path.join(SAVE_DIR, fn))  
            all_X.append(data['arr_0'])  
            all_Y.append(data['arr_1'])  
            print(f"Loaded {fn}: X={data['arr_0'].shape}, Y={data['arr_1'].shape}")
        except Exception as e:
            print(f"Error loading {fn}: {e}")
            continue
            
    if not all_X:  
        print("No checkpoints to merge.")
        return  
        
    X = np.concatenate(all_X, axis=0)  
    Y = np.concatenate(all_Y, axis=0)  
    
    print(f"Total samples: {len(X)}")
    print(f"Value range: [{Y[:,0].min():.3f}, {Y[:,0].max():.3f}]")
    print(f"Policy range: [{Y[:,1].min()}, {Y[:,1].max()}]") 
    print(f"Class distribution: {np.bincount(Y[:,2].astype(int))}")
    
    # Save full dataset
    os.makedirs(SAVE_DIR, exist_ok=True)  
    np.savez_compressed(FINAL_SAVE_PATH, arr_0=X, arr_1=Y)  
    print(f"Saved full dataset ({len(X)} samples) to {FINAL_SAVE_PATH}")  
    
    # Stratified split by class
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    split_point = int((1 - VAL_RATIO) * len(X))
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]
    
    # Save splits
    train_path = os.path.join(SAVE_DIR, 'train.npz')  
    val_path = os.path.join(SAVE_DIR, 'val.npz')  
    
    np.savez_compressed(train_path, arr_0=X[train_indices], arr_1=Y[train_indices])  
    np.savez_compressed(val_path, arr_0=X[val_indices], arr_1=Y[val_indices])  
    
    print(f"Saved train ({len(train_indices)}) & val ({len(val_indices)}) sets.")


def main():  
    os.makedirs(SAVE_DIR, exist_ok=True)  
    manager = Manager()  
    counter = manager.Value('i', 0)  
    lock = manager.Lock()  
    
    # Get PGN files
    if not os.path.exists(DATA_DIR):
        print(f"Data directory {DATA_DIR} not found!")
        return
        
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) 
             if f.endswith('.pgn')]
    
    if not files:
        print("No PGN files found!")
        return
        
    print(f"Found {len(files)} PGN files")
    
    args = [(fp, i, counter, lock) for i, fp in enumerate(files)]  
    
    # Use multiprocessing
    with Pool(min(cpu_count(), len(files))) as pool:  
        pool.map(process_file, args)  
        
    merge_and_split()


if __name__ == '__main__':  
    main()