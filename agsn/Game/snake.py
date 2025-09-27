import cv2
import pygame
import sys
import random
import mediapipe as mp
import numpy as np

# --- SNAKE GAME SETTINGS ---
WINDOW_SIZE = 800  # Area game snake
WEBCAM_WIDTH = 600  # Perbesar area webcam
WEBCAM_HEIGHT = 600
GRID_SIZE = 20
SPEED = 5 # Snake statis dan lambat

# --- FACE DETECTION SETUP ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def find_working_camera(device_index=2):
    # Pakai device index yang diinginkan user (misal: 2 untuk /dev/video2)
    cap = cv2.VideoCapture(device_index)
    if cap.isOpened():
        # Warming up: buang 10 frame pertama
        for _ in range(10):
            cap.read()
        ret, frame = cap.read()
        if ret and frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            print(f'Webcam aktif di device {device_index}')
            return cap
    cap.release()
    return None

# --- SNAKE GAME LOGIC ---
def draw_snake(screen, snake):
    for pos in snake:
        pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(pos[0], pos[1], GRID_SIZE, GRID_SIZE))

def draw_food(screen, food):
    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(food[0], food[1], GRID_SIZE, GRID_SIZE))

def get_random_food():
    return [random.randrange(0, WINDOW_SIZE, GRID_SIZE), random.randrange(0, WINDOW_SIZE, GRID_SIZE)]

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE + WEBCAM_WIDTH, max(WINDOW_SIZE, WEBCAM_HEIGHT)))
    pygame.display.set_caption('Snake Game with Hand Movement Tracking')
    clock = pygame.time.Clock()

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

    def reset_game():
        return [[100, 100]], 'RIGHT', get_random_food(), 0

    # --- Game state ---
    game_running = False
    snake, direction, food, score = reset_game()
    speed = SPEED
    
    # Palm tracking variables
    center_palm_x = None
    center_palm_y = None
    movement_threshold = 0.15  # Sensitivitas pergerakan telapak tangan

    cap = find_working_camera(device_index=2)
    if cap is None:
        print('Tidak ada webcam yang terdeteksi di device 2!')
        font = pygame.font.SysFont(None, 36)
        screen.fill((0,0,0))
        msg = font.render('Webcam tidak ditemukan!', True, (255,0,0))
        screen.blit(msg, (50, WINDOW_SIZE//2))
        pygame.display.flip()
        pygame.time.wait(3000)
        pygame.quit()
        sys.exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

    frame_count = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release()
                pygame.quit()
                sys.exit()
            # Kontrol arah manual (opsional)
            if event.type == pygame.KEYDOWN:
                if game_running:
                    if event.key == pygame.K_1:
                        direction = 'UP'
                    elif event.key == pygame.K_2:
                        direction = 'RIGHT'
                    elif event.key == pygame.K_3:
                        direction = 'DOWN'
                    elif event.key == pygame.K_4:
                        direction = 'LEFT'
            # Mouse click untuk mulai/restart game
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not game_running:
                    snake, direction, food, score = reset_game()
                    game_running = True
                    center_palm_x = None
                    center_palm_y = None

        ret, frame = cap.read()
        if not ret or frame is None or frame.shape[0] == 0 or frame.shape[1] == 0:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        frame_draw = frame_rgb.copy()
        h, w, _ = frame.shape
        
        # --- Index finger movement control ---
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Index finger tip: landmark 8
            index_tip = hand_landmarks.landmark[8]
            finger_x = index_tip.x
            finger_y = index_tip.y
            
            # Set posisi awal finger saat game baru mulai
            if game_running and center_palm_x is None and center_palm_y is None:
                center_palm_x = finger_x
                center_palm_y = finger_y
                
            # Kontrol arah berdasarkan pergerakan index finger
            if game_running and center_palm_x is not None and center_palm_y is not None:
                dx = finger_x - center_palm_x
                dy = finger_y - center_palm_y

                if abs(dx) > abs(dy):
                    if dx > movement_threshold:
                        direction = 'DOWN'
                    elif dx < -movement_threshold:
                        direction = 'UP'
                else:
                    if dy > movement_threshold:
                        direction = 'RIGHT'
                    elif dy < -movement_threshold:
                        direction = 'LEFT'
            
            # Gambar hand tracking di frame
            for lm in hand_landmarks.landmark:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame_draw, (px, py), 3, (0,255,0), -1)
            for c1, c2 in mp_hands.HAND_CONNECTIONS:
                x1, y1 = int(hand_landmarks.landmark[c1].x * w), int(hand_landmarks.landmark[c1].y * h)
                x2, y2 = int(hand_landmarks.landmark[c2].x * w), int(hand_landmarks.landmark[c2].y * h)
                cv2.line(frame_draw, (x1, y1), (x2, y2), (0,255,255), 2)
            
            # Highlight index finger tip dengan circle besar
            finger_px, finger_py = int(finger_x * w), int(finger_y * h)
            cv2.circle(frame_draw, (finger_px, finger_py), 12, (255,0,0), -1)
            cv2.circle(frame_draw, (finger_px, finger_py), 15, (255,255,255), 2)
            
            # Debug info: tampilkan finger position
            debug_text = f'Finger: ({finger_x:.2f}, {finger_y:.2f})'
            cv2.putText(frame_draw, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            # Tampilkan arah movement jika ada
            if game_running and center_palm_x is not None:
                movement_text = f'Direction: {direction}'
                cv2.putText(frame_draw, movement_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # --- UI: jika game belum mulai ---
        if not game_running:
            screen.fill((0, 0, 0))
            font = pygame.font.SysFont(None, 48)
            text = font.render('Klik pada layar untuk mulai!', True, (255, 255, 0))
            screen.blit(text, (WINDOW_SIZE//2 - text.get_width()//2, WINDOW_SIZE//2 - text.get_height()//2))
            # Tampilkan webcam
            frame_surface = pygame.surfarray.make_surface(frame_draw)
            frame_surface = pygame.transform.scale(frame_surface, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
            screen.blit(frame_surface, (WINDOW_SIZE, 0))
            pygame.display.flip()
            clock.tick(15)
            continue

        # --- Move snake ---
        head = snake[0].copy()
        if direction == 'UP':
            head[1] -= GRID_SIZE
        elif direction == 'DOWN':
            head[1] += GRID_SIZE
        elif direction == 'LEFT':
            head[0] -= GRID_SIZE
        elif direction == 'RIGHT':
            head[0] += GRID_SIZE
        snake.insert(0, head)

        # --- Check food collision ---
        if head == food:
            score += 1
            food = get_random_food()
        else:
            snake.pop()

        # --- Check collision with wall or self ---
        if (head[0] < 0 or head[0] >= WINDOW_SIZE or head[1] < 0 or head[1] >= WINDOW_SIZE or head in snake[1:]):
            print(f'Game Over! Score: {score}')
            game_running = False
            continue

        # --- Draw everything ---
        screen.fill((0, 0, 0))
        draw_snake(screen, snake)
        draw_food(screen, food)
        font = pygame.font.SysFont(None, 32)
        score_text = font.render(f'Score: {score}', True, (255, 255, 255))
        screen.blit(score_text, (10, 10))
        # Tampilkan webcam di samping kanan game
        frame_surface = pygame.surfarray.make_surface(frame_draw)
        frame_surface = pygame.transform.scale(frame_surface, (WEBCAM_WIDTH, WEBCAM_HEIGHT))
        screen.blit(frame_surface, (WINDOW_SIZE, 0))
        # Tampilkan hint kontrol di bawah
        hint_font = pygame.font.SysFont(None, 28)
        hint_text = hint_font.render('Kontrol arah: Gerakkan jari telunjuk | Klik layar untuk mulai/restart', True, (200, 200, 0))
        screen.blit(hint_text, (10, WINDOW_SIZE - 40))
        pygame.display.flip()
        clock.tick(speed)

if __name__ == "__main__":
    main()
