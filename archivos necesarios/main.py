import pygame
import sys

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
NEW_WHITE = (200, 200, 200)

# Crear ventana
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tablero de Damas 4x4")

# Estado del tablero
board = [[None for _ in range(COLS)] for _ in range(ROWS)]
selected_piece = None
current_turn = "white"  # Turno inicial

def is_queen(piece, row):
    return (piece == "white" and row == 0) or (piece == "black" and row == ROWS - 1)

# Inicializar piezas
for col in range(0, COLS, 2):
    board[0][col] = "black"
    board[3][col + 1] = "white"

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
                pygame.draw.circle(screen, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius, 2)
            elif piece == "WHITE":
                pygame.draw.circle(screen, WHITE, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius)
                pygame.draw.circle(screen, RED, (col * SQUARE_SIZE + SQUARE_SIZE // 2, row * SQUARE_SIZE + SQUARE_SIZE // 2), radius, 2)

def get_square_under_mouse():
    mouse_x, mouse_y = pygame.mouse.get_pos()
    col = mouse_x // SQUARE_SIZE
    row = mouse_y // SQUARE_SIZE
    return row, col

def is_valid_move(piece_row, piece_col, target_row, target_col):
    piece = board[piece_row][piece_col]
    if piece is None:
        return False

    direction = -1 if piece.lower() == "white" else 1

    # Movimiento diagonal básico
    if abs(target_row - piece_row) == 1 and abs(target_col - piece_col) == 1:
        if target_row - piece_row == direction or piece.isupper():
            return True

    # Comer una pieza
    if abs(target_row - piece_row) == 2 and abs(target_col - piece_col) == 2:
        middle_row = (piece_row + target_row) // 2
        middle_col = (piece_col + target_col) // 2
        if board[middle_row][middle_col] is not None and board[middle_row][middle_col].lower() != piece.lower():
            board[middle_row][middle_col] = None
            return True

    # Movimiento de reina en múltiples casillas
    if piece.isupper() and abs(target_row - piece_row) == abs(target_col - piece_col):
        step_row = 1 if target_row > piece_row else -1
        step_col = 1 if target_col > piece_col else -1
        for i in range(1, abs(target_row - piece_row)):
            if board[piece_row + i * step_row][piece_col + i * step_col] is not None:
                return False
        return True

    return False

def can_piece_move(piece_row, piece_col):
    piece = board[piece_row][piece_col]
    if piece is None:
        return False

    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    if piece.isupper():  # Reina puede moverse más lejos
        for dr, dc in directions:
            for dist in range(1, ROWS):
                target_row = piece_row + dr * dist
                target_col = piece_col + dc * dist
                if 0 <= target_row < ROWS and 0 <= target_col < COLS and board[target_row][target_col] is None:
                    return True
    else:
        direction = -1 if piece == "white" else 1
        for dr, dc in directions[:2]:
            target_row = piece_row + dr * direction
            target_col = piece_col + dc
            if 0 <= target_row < ROWS and 0 <= target_col < COLS and board[target_row][target_col] is None:
                return True

    return False

def switch_turn():
    global current_turn
    for row in range(ROWS):
        for col in range(COLS):
            if board[row][col] is not None and board[row][col].lower() == current_turn:
                if can_piece_move(row, col):
                    current_turn = "black" if current_turn == "white" else "white"
                    return

    current_turn = "black" if current_turn == "white" else "white"

def main():
    global selected_piece, current_turn
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                row, col = get_square_under_mouse()
                if selected_piece:
                    piece_row, piece_col = selected_piece
                    # Verificar si el movimiento es válido
                    if board[row][col] is None and (row + col) % 2 == 0 and is_valid_move(piece_row, piece_col, row, col):
                        piece = board[piece_row][piece_col]
                        if piece.lower() == current_turn:  # Verificar turno
                            board[row][col] = piece.upper() if is_queen(piece, row) else piece
                            board[piece_row][piece_col] = None
                            selected_piece = None
                            switch_turn()
                    else:
                        selected_piece = None  # Deseleccionar si no es válido
                else:
                    # Seleccionar pieza
                    if board[row][col] is not None and board[row][col].lower() == current_turn:
                        selected_piece = (row, col)

        draw_board()
        draw_pieces()
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
