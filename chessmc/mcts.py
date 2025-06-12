import torch
import numpy as np
import chess
import copy
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import random


def move_to_policy_index(move: chess.Move) -> int:
    """
    Encode chess.Move into policy index (0-4095) matching training steps:
    from_square * 64 + to_square, with promotion offsets.
    """
    idx = move.from_square * 64 + move.to_square
    if move.promotion:
        promotion_offset = {
            chess.QUEEN: 0,
            chess.ROOK: 1024,
            chess.BISHOP: 2048,
            chess.KNIGHT: 3072
        }
        idx += promotion_offset.get(move.promotion, 0)
    return min(idx, 4095)


class ImprovedMCTSNode:
    """
    Cải tiến MCTS Node với:
    - Virtual loss để tránh race condition trong parallel search
    - Progressive widening để control exploration
    - Transposition table support
    """
    def __init__(self, state, move=None, parent=None, prior_prob: float = 0.0):
        self.state = state
        self.move = move
        self.parent = parent
        self.prior_prob = prior_prob

        # MCTS statistics
        self.visit_count = 0
        self.total_value = 0.0
        self.virtual_loss = 0

        # Children management
        self.children: Dict[chess.Move, 'ImprovedMCTSNode'] = {}
        self.children_priors: Dict[chess.Move, float] = {}
        self.is_expanded = False

        # Caching
        self.cached_legal_moves = None
        self.cached_is_game_over = None

    @property
    def q_value(self) -> float:
        """Average value of this node"""
        if self.visit_count == 0:
            return 0.0
        return (self.total_value - self.virtual_loss) / (self.visit_count + 1e-8)

    @property
    def u_value(self) -> float:
        """Upper confidence bound exploration term"""
        if self.parent is None:
            return 0.0
        c_puct = 0.6
        sqrt_parent = math.sqrt(self.parent.visit_count + 1)
        return c_puct * self.prior_prob * sqrt_parent / (1 + self.visit_count)

    @property
    def ucb_score(self) -> float:
        """UCB1 score for child selection"""
        return self.q_value + self.u_value

    def is_game_over(self) -> bool:
        """Cached game over state"""
        if self.cached_is_game_over is None:
            self.cached_is_game_over = self.state.board.is_game_over()
        return self.cached_is_game_over

    def get_legal_moves(self) -> List[chess.Move]:
        """Cached legal moves"""
        if self.cached_legal_moves is None:
            self.cached_legal_moves = list(self.state.board.legal_moves)
        return self.cached_legal_moves

    def select_child(self) -> 'ImprovedMCTSNode':
        """Select best child based on UCB"""
        return max(self.children.values(), key=lambda n: n.ucb_score)

    def expand(self, policy_probs: Dict[chess.Move, float]) -> None:
        """Expand node with top-k moves from policy"""
        if self.is_expanded or self.is_game_over():
            return
        legal_moves = self.get_legal_moves()
          # nếu là root thì mở hết, còn node con mới giới hạn
        if self.parent is None:
            k = len(legal_moves)
        else:
            k = min(len(legal_moves), max(4, int(math.sqrt(len(legal_moves)))))
        sorted_moves = sorted(legal_moves, key=lambda m: policy_probs.get(m, 0.0), reverse=True)
        for move in sorted_moves[:k]:
            prob = policy_probs.get(move, 0.0)
            child_state = copy.deepcopy(self.state)
            child_state.board.push(move)
            child = ImprovedMCTSNode(child_state, move, self, prob)
            self.children[move] = child
            self.children_priors[move] = prob
        self.is_expanded = True

    def backup(self, value: float) -> None:
        """Backpropagate value through ancestors"""
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backup(-value)

    def add_virtual_loss(self) -> None:
        """Add virtual loss for parallel threads"""
        self.virtual_loss += 1
        if self.parent:
            self.parent.add_virtual_loss()

    def remove_virtual_loss(self) -> None:
        """Remove virtual loss after simulation"""
        self.virtual_loss -= 1
        if self.parent:
            self.parent.remove_virtual_loss()

    def get_visit_counts(self) -> Dict[chess.Move, int]:
        """Visits per move"""
        return {move: node.visit_count for move, node in self.children.items()}

    def get_action_probs(self, temperature: float = 1.0) -> Dict[chess.Move, float]:
        """Action probabilities based on visit counts"""
        counts = self.get_visit_counts()
        if not counts:
            legal = self.get_legal_moves()
            return {m: 1/len(legal) for m in legal} if legal else {}
        moves, visits = zip(*counts.items())
        visits = np.array(visits, dtype=float)
        if temperature == 0:
            best = moves[np.argmax(visits)]
            return {m: 1.0 if m==best else 0.0 for m in moves}
        visits = visits ** (1/temperature)
        probs = visits / visits.sum()
        return {m: float(p) for m,p in zip(moves, probs)}


class NeuralNetworkEvaluator:
    """Interface neural network integration"""
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        from chessmc.model import ImprovedChessModel
        self.model = ImprovedChessModel()
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get('model_state_dict',
                       checkpoint.get('state_dict', checkpoint))
        self.model.load_state_dict(state_dict)
        

        self.model.to(self.device)
        self.model.eval()

    def state_to_tensor(self, state) -> torch.Tensor:
        data = state.serialize_cnn()
        return torch.FloatTensor(data).reshape(18,8,8)

    def evaluate_batch(self, states: List) -> Tuple[List[float], List[Dict[chess.Move, float]]]:
        if not states:
            return [], []
        tensors, all_moves = [], []
        for s in states:
            tensors.append(self.state_to_tensor(s))
            all_moves.append(list(s.board.legal_moves))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            values, policies, _ = self.model(batch)
        vals = values.cpu().numpy().flatten().tolist()
        policies = policies.cpu().numpy()
        policy_dicts = []
        for i, moves in enumerate(all_moves):
            policy_dicts.append(self.policy_to_move_dict(policies[i], moves))
        return vals, policy_dicts

    def policy_to_move_dict(self, logits: np.ndarray, legal_moves: List[chess.Move]) -> Dict[chess.Move, float]:
        shifted = logits - np.max(logits)     # shift để ổn định
        exp_logits = np.exp(shifted)  
        probs = exp_logits / exp_logits.sum()
        return {m: float(probs[move_to_policy_index(m)]) for m in legal_moves}


import time
import copy
import numpy as np
import chess
import random
from typing import Optional, Dict, List, Tuple



class AdvancedMCTS:
    """Advanced MCTS with network guidance và Dirichlet‐noise tại root."""
    def __init__(
        self,
        evaluator: NeuralNetworkEvaluator,
        c_puct: float = 0.6,
        n_threads: int = 1,
        use_transposition_table: bool = True,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25
    ):
        """
        Parameters:
            evaluator: Neural network evaluator, có method evaluate_batch([state]) → (values, priors).
            c_puct: hệ số khám phá UCT.
            n_threads: số luồng (chưa dùng trong ví dụ này).
            use_transposition_table: nếu True, dùng bảng chuyển vị (hiện chưa tích hợp).
            dirichlet_alpha: tham số alpha cho Dirichlet‐noise.
            dirichlet_epsilon: tỉ lệ epsilon để trộn noise vào priors gốc tại root.
        """
        self.evaluator = evaluator
        self.c_puct = c_puct
        self.n_threads = n_threads
        self.use_tt = use_transposition_table
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.ttable: Dict[int, ImprovedMCTSNode] = {}
        self.stats = {'nodes': 0, 'evals': 0, 'time': 0.0}

    def search(
        self,
        root_state,
        n_simulations: int = 1000,
        time_limit: Optional[float] = None
    ):
        """
        Thực hiện MCTS guided by neural network, với Dirichlet‐noise tại root.

        Parameters:
            root_state: một instance của State (có attribute .board).
            n_simulations: số vòng simulation tối đa.
            time_limit: nếu được truyền (giây), sẽ dừng khi vượt thời gian.
        Returns:
            best_move: nước đi có visit_count cao nhất sau khi hoàn thành simulations.
            info: dict chứa 'stats' và 'probs' (phân phối visit_counts tại root).
        """
        start = time.time()
        root = ImprovedMCTSNode(root_state)

        # 1) initial expand root bằng network, lấy giá trị và priors gốc
        vals, priors = self.evaluator.evaluate_batch([root_state])
        priors_root = priors[0]  # dict {move: prior_prob}
        if priors_root:
            # tìm move với xác suất prior cao nhất
            best_prior_move, best_prior_prob = max(
                priors_root.items(), key=lambda item: item[1]
            )
            print(f"[DEBUG] best prior move: {best_prior_move}  |  prior = {best_prior_prob:.4f}")
        else:
            print("[DEBUG] priors_root trống, không có move nào.")
        # 2) Bỏ Dirichlet‐noise vào dùng priors_root
        

        root.expand(priors_root)
        self.stats['evals'] += 1  # Đếm 1 lần network evaluate cho root

        sims = 0
        while sims < n_simulations:
            if time_limit and (time.time() - start) > time_limit:
                break

            node = root
            state = copy.deepcopy(root_state)

            # 3) Selection: đi xuống đến node chưa mở
            while node.is_expanded and not node.is_game_over():
                node.add_virtual_loss()
                node = node.select_child()     # <— sửa ở đây: bỏ tham số c_puct
                state.board.push(node.move)

            # 4) Evaluation/Expansion
            if node.is_game_over():
                # Nếu đã terminal, tính giá trị cuối
                value = self._get_terminal_value(node)
            else:
                vals_batch, priors_batch = self.evaluator.evaluate_batch([state])
                value = vals_batch[0]
                priors_new = priors_batch[0]
                node.expand(priors_new)
                self.stats['evals'] += 1

            node.remove_virtual_loss()
            # 5) Backup: cập nhật giá trị ngược lại gốc
            node.backup(value)
            sims += 1
            self.stats['nodes'] += 1

        self.stats['time'] = time.time() - start

        # 6) Chọn best_move từ root.children theo visit_count cao nhất
        best_move = max(root.children.items(), key=lambda x: x[1].visit_count)[0]
        info = {
            'stats': self.stats,
            'probs': root.get_action_probs()  # dict {move: visit_count/sum_visits}
        }
        return best_move, info

    def _get_terminal_value(self, node: ImprovedMCTSNode) -> float:
        """
        Nếu node đã game over, trả về:
          +1  nếu người tới lượt tại node thắng,
          -1  nếu người tới lượt thua,
           0 nếu hòa.
        """
        result = node.state.board.result()
        if result == '1-0':
            return 1.0 if node.state.board.turn == chess.WHITE else -1.0
        elif result == '0-1':
            return 1.0 if node.state.board.turn == chess.BLACK else -1.0
        else:
            return 0.0

    def clear(self):
        """Xoá bảng chuyển vị (nếu sử dụng)."""
        self.ttable.clear()


def create_advanced_chess_ai(model_path: str, device: str = 'cpu') -> AdvancedMCTS:
    """
    Hàm tiện ích để khởi AdvancedMCTS với NeuralNetworkEvaluator.
    """
    evaluator = NeuralNetworkEvaluator(model_path, device)
    return AdvancedMCTS(evaluator)



def improved_uct_search(state, n_simulations=1000, model_path=None, device='cpu'):
    if model_path is None:
        model_path = 'models/improved-chess-model.pth'
    try:
        # khởi tạo mcts với device (cpu hoặc cuda)
        mcts = create_advanced_chess_ai(model_path, device)
        move, info = mcts.search(state, n_simulations)
        return move
    except Exception as e:
        print(f"mcts error: {e}")
        return random.choice(list(state.board.legal_moves))

