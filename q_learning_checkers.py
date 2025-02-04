import pygame
import sys
import numpy as np
import pickle
import random

# Inicializar pygame
pygame.init()

# Dimensiones de la ventana y los colores
WIDTH, HEIGHT = 400, 400
ROWS, COLS = 4, 4
SQUARE_SIZE = WIDTH // COLS

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GOLD = (255, 215, 0)

# Parámetros de Q-Learning
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPSILON = 0.1  # Para exploración
TRAINING_EPISODES = 5 #1000

# Clase para manejar el Q-Learning
class QLearningAgent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = EPSILON
        
    def get_state_key(self, board):
        # Convertir el tablero en una tupla inmutable para usar como clave
        return tuple(tuple(row) for row in board)
    
    def get_q_value(self, state, action):
        # Método auxiliar para obtener valores Q con valor predeterminado 0
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def get_action(self, board, valid_moves, training=False):
        state = self.get_state_key(board)
        
        if training and random.random() < self.epsilon:
            return random.choice(valid_moves) if valid_moves else None
            
        best_value = float('-inf')
        best_move = None
        
        for move in valid_moves:
            action_key = (move[0], move[1])
            q_value = self.get_q_value(state, action_key)
            
            if q_value > best_value:
                best_value = q_value
                best_move = move
                
        return best_move if best_move else (random.choice(valid_moves) if valid_moves else None)
    
    def update(self, state, action, next_state, reward):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        action_key = (action[0], action[1])
        
        # Obtener Q(s,a) - El valor actual
        current_q = self.get_q_value(state_key, action_key)
        
        # Obtener max Q(s',a') - El máximo valor Q del siguiente estado
        next_q_values = [self.get_q_value(next_state_key, (move[0], move[1])) 
                        for move in get_all_valid_moves("black")]
        next_max = max(next_q_values) if next_q_values else 0
        
        # Fórmula Q-Learning TD:
        # Q(s,a) = Q(s,a) + α[R + γ*max Q(s',a') - Q(s,a)]
        # donde:
        # α (LEARNING_RATE) = tasa de aprendizaje
        # γ (DISCOUNT_FACTOR) = factor de descuento
        # R = reward (recompensa inmediata)
        new_value = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max)
        
        # Actualizar el valor en la tabla Q
        if state_key not in self.q_table:
            self.q_table[state_key] = {}
        self.q_table[state_key][action_key] = new_value
    
    def save_q_table(self, filename='q_table.pkl'):
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self.q_table, f)
            print("Q-table guardada exitosamente")
        except Exception as e:
            print(f"Error al guardar Q-table: {e}")
    
    def load_q_table(self, filename='q_table.pkl'):
        try:
            with open(filename, 'rb') as f:
                loaded_data = pickle.load(f)
                if isinstance(loaded_data, dict):
                    self.q_table = loaded_data
                else:
                    print("Archivo Q-table corrupto, creando nueva tabla")
                    self.q_table = {}
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            print("No se pudo cargar Q-table, creando nueva tabla")
            self.q_table = {}

# Crear ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Damas con Q-Learning")

# Estado del tablero
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
selected_piece = None
current_turn = "black"  # Turno inicial (agente)
moves_without_capture = 0

# Inicializar piezas
for col in range(0, COLS, 2):
    board[0][col] = "black"
    board[3][col + 1] = "white"

# Crear agente Q-Learning
agent = QLearningAgent()

def get_reward(board, is_terminal, winner):
    if is_terminal:
        if winner == 1:  # Agente gana
            return 100
        elif winner == -1:  # Agente pierde
            return -100
        return 0  # Empate
    
    # Recompensa basada en la diferencia de piezas
    black_pieces = sum(cell == "black" or cell == "BLACK" for row in board for cell in row)
    white_pieces = sum(cell == "white" or cell == "WHITE" for row in board for cell in row)
    return (black_pieces - white_pieces) * 10

def is_queen(piece, row):
    return (piece == "white" and row == 0) or (piece == "black" and row == ROWS - 1)

def draw_board():
    for row in range(ROWS):
        for col in range(COLS):
            color = BLACK if (row + col) % 2 == 0 else WHITE
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces():
    radius = SQUARE_SIZE // 3
    for row in range(ROWS):
        for col in range(COLS):
            piece = board[row][col]
            if piece == "black":
                pygame.draw.circle(screen, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
                pygame.draw.circle(screen, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius, 2)
            elif piece == "white":
                pygame.draw.circle(screen, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
            elif piece == "BLACK":
                pygame.draw.circle(screen, BLACK, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
                pygame.draw.circle(screen, GOLD, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius, 2)
            elif piece == "WHITE":
                pygame.draw.circle(screen, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
                pygame.draw.circle(screen, GOLD, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius, 2)

def get_square_under_mouse():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    col = mouse_x // SQUARE_SIZE
    row = mouse_y // SQUARE_SIZE
    return row, col

def is_within_bounds(row, col):
    return 0 <= row < ROWS and 0 <= col < COLS

def get_all_valid_moves(player):
    moves = []
    capture_moves = []

    for row in range(len(board)):
        for col in range(len(board[row])):
            if board[row][col] is not None and board[row][col].lower() == player.lower():
                new_moves, new_capture_moves = get_piece_moves(row, col)
                moves.extend(new_moves)
                capture_moves.extend(new_capture_moves)
                
    return capture_moves if capture_moves else moves

def is_valid_capture(piece_row, piece_col, target_row, target_col):
    if abs(target_row - piece_row) == 2:  # Movimiento de captura
        middle_row = (piece_row + target_row) // 2
        middle_col = (piece_col + target_col) // 2

        if is_within_bounds(middle_row, middle_col) and is_within_bounds(target_row, target_col):
            if board[target_row][target_col] is not None:
                return False

            return (
                board[middle_row][middle_col] is not None
                and board[middle_row][middle_col].lower() != board[piece_row][piece_col].lower()
            )
    return False

def is_at_edge(row, col):
    return row == 0 or row == ROWS - 1 or col == 0 or col == COLS - 1

def get_piece_moves(piece_row, piece_col):
    moves = []
    capture_moves = []

    piece = board[piece_row][piece_col]
    directions = []

    # Determinar direcciones permitidas
    if piece.lower() == "white":
        directions = [(-1, 1), (-1, -1)]  # Sólo moverse hacia arriba
    elif piece.lower() == "black":
        directions = [(1, 1), (1, -1)]  # Sólo moverse hacia abajo

    # Permitir movimientos en ambas direcciones para reinas
    if piece == "WHITE" or piece == "BLACK":
        directions = [
            (1, 1), (1, -1),  # Diagonales hacia abajo
            (-1, 1), (-1, -1)  # Diagonales hacia arriba
        ]

    for dr, dc in directions:
        # Movimiento normal (una casilla en la dirección)
        target_row, target_col = piece_row + dr, piece_col + dc
        if is_valid_move(piece_row, piece_col, target_row, target_col, capture=False):
            moves.append(((piece_row, piece_col), (target_row, target_col)))

        # Movimiento de captura (dos casillas en la dirección)
        capture_row, capture_col = piece_row + 2 * dr, piece_col + 2 * dc
        if is_valid_capture(piece_row, piece_col, capture_row, capture_col):
            capture_moves.append(((piece_row, piece_col), (capture_row, capture_col)))
    
    return moves, capture_moves

def is_valid_move(piece_row, piece_col, target_row, target_col, capture=True):
    if not is_within_bounds(target_row, target_col):
        return False

    piece = board[piece_row][piece_col]
    if piece is None:
        return False

    direction = -1 if piece.lower() == "white" else 1

    # Movimiento diagonal básico
    if abs(target_row - piece_row) == 1 and abs(target_col - piece_col) == 1:
        if piece == "white" and target_row > piece_row:
            return False
        
        if board[target_row][target_col] is None:
            return True

    # Comer una pieza
    if capture and abs(target_row - piece_row) == 2 and abs(target_col - piece_col) == 2:
        middle_row = (piece_row + target_row) // 2
        middle_col = (piece_col + target_col) // 2
        if (
            is_within_bounds(middle_row, middle_col)
            and board[middle_row][middle_col] is not None
            and board[middle_row][middle_col].lower() != piece.lower()
        ):
            return True

    return False

def make_move(piece_row, piece_col, target_row, target_col):
    global moves_without_capture
    
    piece = board[piece_row][piece_col]
    board[piece_row][piece_col] = None
    board[target_row][target_col] = piece.upper() if is_queen(piece, target_row) else piece
    
    if abs(target_row - piece_row) == 2:
        middle_row = (piece_row + target_row) // 2
        middle_col = (piece_col + target_col) // 2
        board[middle_row][middle_col] = None
        moves_without_capture = 0
    else:
        moves_without_capture += 1
    
    return board

def is_terminal():
    if moves_without_capture >= 64:
        return True, 0  # Empate

    white_exists = any(cell == "white" or cell == "WHITE" for row in board for cell in row)
    black_exists = any(cell == "black" or cell == "BLACK" for row in board for cell in row)

    if not white_exists:
        return True, 1  # Gana el agente
    if not black_exists:
        return True, -1  # Pierde el agente

    return False, None

def agent_move(training=False):
    valid_moves = get_all_valid_moves("black")
    if not valid_moves:
        return False
    
    # Obtener acción del agente
    move = agent.get_action(board, valid_moves, training)
    if move:
        piece_row, piece_col = move[0]
        target_row, target_col = move[1]
        
        # Guardar estado actual
        old_state = [row[:] for row in board]
        
        # Realizar movimiento
        make_move(piece_row, piece_col, target_row, target_col)
        
        # Obtener recompensa
        is_term, winner = is_terminal()
        reward = get_reward(board, is_term, winner)
        
        # Actualizar Q-table si estamos entrenando
        if training:
            agent.update(old_state, move, board, reward)
        
        return True
    return False

def train_agent():
    print("Entrenando agente...")
    for episode in range(TRAINING_EPISODES):
        # Reiniciar tablero
        for row in range(ROWS):
            for col in range(COLS):
                board[row][col] = None
        
        for col in range(0, COLS, 2):
            board[0][col] = "black"
            board[3][col + 1] = "white"
        
        moves_without_capture = 0
        current_turn = "black"
        
        while True:
            if current_turn == "black":
                if not agent_move(training=True):
                    break
                current_turn = "white"
            else:
                # Simular movimiento del oponente (aleatorio)
                valid_moves = get_all_valid_moves("white")
                if valid_moves:
                    move = random.choice(valid_moves)
                    make_move(*move[0], *move[1])
                current_turn = "black"
            
            is_terminal_state, winner = is_terminal()
            if is_terminal_state:
                break
        
        if episode % 100 == 0:
            print(f"Episodio {episode}/{TRAINING_EPISODES}")
    
    print("Entrenamiento completado")
    agent.save_q_table()


def visualize_q_table_content():
    try:
        with open('q_table.pkl', 'rb') as f:
            q_table = pickle.load(f)
            
        print("\nContenido del archivo q_table.pkl:")
        print("===================================")
        print(f"Número total de estados almacenados: {len(q_table)}")
        
        # Mostrar algunos ejemplos de estados y sus valores Q
        print("\nEjemplos de estados y sus valores Q:")
        for i, (state, actions) in enumerate(q_table.items()):
            if i >= 3:  # Mostrar solo los primeros 3 estados como ejemplo
                break
                
            print("\nEstado del tablero:")
            for row in state:  # state es una tupla de tuplas que representa el tablero
                print(row)
            
            print("\nAcciones posibles y sus valores Q:")
            for action, q_value in actions.items():
                # action es una tupla ((from_row, from_col), (to_row, to_col))
                from_pos, to_pos = action
                print(f"Mover de {from_pos} a {to_pos}: {q_value:.2f}")
            print("-" * 50)
            
    except FileNotFoundError:
        print("No se encontró el archivo q_table.pkl")
    except Exception as e:
        print(f"Error al leer el archivo: {e}")

def main():
    global selected_piece, current_turn
    
    # Cargar Q-table si existe o entrenar si no
    agent.load_q_table()
    
    if not agent.q_table:
        train_agent()

    # Visualizar el contenido de la tabla Q
    visualize_q_table_content()
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if current_turn == "white" and event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_square_under_mouse()
                if selected_piece:
                    piece_row, piece_col = selected_piece
                    if board[row][col] is None and is_valid_move(piece_row, piece_col, row, col):
                        make_move(piece_row, piece_col, row, col)
                        selected_piece = None
                        current_turn = "black"
                else:
                    if board[row][col] is not None and board[row][col].lower() == "white":
                        selected_piece = (row, col)
        
        if current_turn == "black":
            if agent_move(training=False):
                current_turn = "white"
        
        draw_board()
        draw_pieces()
        pygame.display.flip()
        clock.tick(60)
        
        # Verificar si el juego ha terminado
        is_terminal_state, winner = is_terminal()
        if is_terminal_state:
            if winner == 1:
                print("¡Gana el agente!")
            elif winner == -1:
                print("¡Gana el jugador!")
            else:
                print("¡Empate!")
            pygame.time.wait(2000)
            running = False
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()