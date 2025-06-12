import chess
import numpy as np


class State(object):
    def __init__(self, board=None):
        self.board = chess.Board() if board is None else board

    @property
    def legal_moves(self):
        return list(self.board.legal_moves)

    def serialize(self):
        """
        Serialize board to 768-dimensional vector (12 * 64)
        FIXED VERSION: Proper piece representation
        """
        # 12 piece types: P,N,B,R,Q,K,p,n,b,r,q,k
        state = np.zeros(1152, dtype=np.float32)
        
        piece_to_index = {
            'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
            'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
        }
        
        # Fill piece positions
        piece_map = self.board.piece_map()
        for square, piece in piece_map.items():
            piece_type_idx = piece_to_index[piece.symbol()]
            idx = piece_type_idx * 64 + square
            state[idx] = 1.0
            
        return state
    
    def serialize_cnn(self):
        planes = np.zeros((18, 8, 8), dtype=np.float32)

        # 1) quân cờ (12 kênh cũ)
        piece_to_idx = { 'P':0,'N':1,'B':2,'R':3,'Q':4,'K':5,
                         'p':6,'n':7,'b':8,'r':9,'q':10,'k':11 }
        for sq, pc in self.board.piece_map().items():
            r, c = divmod(sq, 8)
            planes[piece_to_idx[pc.symbol()], r, c] = 1.0

        # 2) side-to-move
        planes[12, :, :] = 1.0 if self.board.turn == chess.WHITE else 0.0

        # 3) castling rights
        planes[13, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[14, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[15, :, :] = 1.0 if self.board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[16, :, :] = 1.0 if self.board.has_queenside_castling_rights(chess.BLACK) else 0.0

        # 4) en-passant file
        if self.board.ep_square is not None:
            r, c = divmod(self.board.ep_square, 8)
            planes[17, r, c] = 1.0

        return planes

    
    def get_game_result(self):
        """Get game result from current position"""
        if self.board.is_checkmate():
            return 1.0 if self.board.turn == chess.BLACK else -1.0  # Win for non-moving side
        elif self.board.is_stalemate() or self.board.is_insufficient_material():
            return 0.0  # Draw
        else:
            return None  # Game not over
    
    def copy(self):
        """Create a copy of the state"""
        return State(self.board.copy())
    
    def make_move(self, move):
        """Make a move and return new state"""
        new_board = self.board.copy()
        new_board.push(move)
        return State(new_board)
    
    def get_legal_move_indices(self):
        """Get policy indices for all legal moves"""
        indices = []
        for move in self.legal_moves:
            idx = move.from_square * 64 + move.to_square
            if move.promotion:
                promotion_offset = {
                    chess.QUEEN: 0,
                    chess.ROOK: 1024,
                    chess.BISHOP: 2048, 
                    chess.KNIGHT: 3072
                }
                idx += promotion_offset.get(move.promotion, 0)
            indices.append(min(idx, 4095))
        return indices