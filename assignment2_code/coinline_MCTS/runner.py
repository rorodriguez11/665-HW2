# Authors: S. El Alaoui and ChatGPT 5.2
# runner.py

import pygame
import sys
import random
import time
import coinline as cl

# Pygame Setup  ----------------
pygame.init()
WIDTH, HEIGHT = 1000, 400
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Coin Line Game")
FONT = pygame.font.SysFont("arial", 24)
BIG_FONT = pygame.font.SysFont("arial", 40)
CLOCK = pygame.time.Clock()

# Coin Details ----------------
NUM_COINS = 15
GAP = 20
COIN_RADIUS = (WIDTH - GAP*(NUM_COINS+2))//(NUM_COINS*2)
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 50
BUTTON_Y = HEIGHT - BUTTON_HEIGHT - 20
BUTTON_COLOR = (100, 100, 255)
BUTTON_HOVER_COLOR = (150, 150, 255)
BUTTON_TEXT_COLOR = (255, 255, 255)

# Buttons (kept for display only)
buttons = {
    "L1": pygame.Rect(100, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
    "L2": pygame.Rect(275, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
    "R1": pygame.Rect(550, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
    "R2": pygame.Rect(725, BUTTON_Y, BUTTON_WIDTH, BUTTON_HEIGHT),
}

# --- MCTS settings for "player" ---
PLAYER_BUDGET = 2000
PLAYER_REWARD_MODE = "winloss"   # or "scorediff"
PLAYER_C = 2 ** 0.5              # sqrt(2)

def draw_game(state, message=""):
    SCREEN.fill((30, 30, 30))
    coins = state.coins

    # Draw coins
    x = (WIDTH - ((COIN_RADIUS * 2 + GAP) * NUM_COINS - GAP)) // 2
    y = HEIGHT // 2 - 50
    for value in coins:
        pygame.draw.circle(SCREEN, (200, 200, 0), (x + COIN_RADIUS, y), COIN_RADIUS)
        text = FONT.render(str(value), True, (0, 0, 0))
        text_rect = text.get_rect(center=(x + COIN_RADIUS, y))
        SCREEN.blit(text, text_rect)
        x += COIN_RADIUS * 2 + GAP

    # Scores
    p_text = FONT.render(f"You (MCTS): {state.pScore}", True, (255, 255, 255))
    a_text = FONT.render(f"AI (Minimax): {state.aiScore}", True, (255, 255, 255))
    t_text = FONT.render(f"Turn: {state.turn.upper()}", True, (200, 200, 255))
    SCREEN.blit(p_text, (20, 20))
    SCREEN.blit(a_text, (20, 50))
    SCREEN.blit(t_text, (20, 80))

    # Buttons (display only)
    for label, rect in buttons.items():
        is_hovered = rect.collidepoint(pygame.mouse.get_pos())
        color = BUTTON_HOVER_COLOR if is_hovered else BUTTON_COLOR
        pygame.draw.rect(SCREEN, color, rect)
        pygame.draw.rect(SCREEN, (255, 255, 255), rect, 2)
        btn_text = FONT.render(label, True, BUTTON_TEXT_COLOR)
        SCREEN.blit(btn_text, btn_text.get_rect(center=rect.center))

    if message:
        msg_text = BIG_FONT.render(message, True, (255, 100, 100))
        SCREEN.blit(msg_text, msg_text.get_rect(center=(WIDTH // 2, HEIGHT - 100)))

    pygame.display.flip()

# --- Main Game Loop ---
def main():
    initial_coins = [random.randint(1, 15) for _ in range(NUM_COINS)]
    state = cl.State(initial_coins)

    game_over = False
    result_message = ""

    while True:
        CLOCK.tick(30)

        if cl.terminal(state) and not game_over:
            win = cl.winner(state)
            game_over = True
            if str(win).lower() == "player":
                result_message = "PLAYER (MCTS) Wins!"
            elif str(win).lower() == "ai":
                result_message = "AI (Minimax) Wins!"
            else:
                result_message = "It's a Tie!"

        draw_game(state, result_message)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            # Start new game if game is over and SPACE is pressed
            if game_over and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                initial_coins = [random.randint(1, 15) for _ in range(NUM_COINS)]
                state = cl.State(initial_coins)
                game_over = False
                result_message = ""

        # --- Automated "player" move using MCTS (replaces clicks) ---
        if not game_over and cl.player(state) == "player":
            pygame.time.delay(100)
            t0 = time.perf_counter()
            action = cl.mcts(state, budget=PLAYER_BUDGET, reward_mode=PLAYER_REWARD_MODE, c=PLAYER_C)
            t1 = time.perf_counter()
            print(f"PLAYER (MCTS) action={action} time={(t1 - t0)*1000:.2f}ms")
            if action:
                state = cl.succ(state, action)

        # --- AI Move (minimax baseline) ---
        if not game_over and cl.player(state) == "ai":
            pygame.time.delay(100)
            _, action = cl.minimax(state, is_maximizing=True)
            print(f"AI (Minimax) action={action}")
            if action:
                state = cl.succ(state, action)

if __name__ == "__main__":
    main()