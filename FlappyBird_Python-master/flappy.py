import pygame, sys, random, numpy as np, os
import pandas as pd

# === CONFIGURATION FLAGS ===
RESET = False
SAVE = False
VISUALIZE_AFTER = True
SAVE_FILE = "best_bird.txt"
DATA_FILE = "training_data.csv"

# === DETERMINE START GENERATION ===
if not RESET and os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        start_gen = int(df['generation'].iloc[-1]) + 1
    except:
        start_gen = 1
else:
    start_gen = 1

# === NEURAL NETWORK CLASS ===
class NeuralNetwork:
    def __init__(self):
        self.input_size = 5
        self.hidden_size = 24
        self.output_size = 1
        self.w1 = np.random.randn(self.hidden_size, self.input_size)
        self.b1 = np.random.randn(self.hidden_size)
        self.w2 = np.random.randn(self.output_size, self.hidden_size)
        self.b2 = np.random.randn(self.output_size)

    def forward(self, x):
        z1 = np.dot(self.w1, x) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(self.w2, a1) + self.b2
        return z2

    def save_to_txt(self, filename):
        with open(filename, "w") as f:
            for array in [self.w1, self.b1, self.w2, self.b2]:
                flat = array.flatten()
                f.write(",".join(map(str, flat)) + "\n")

    def load_from_txt(self, filename):
        with open(filename, "r") as f:
            lines = f.readlines()
            self.w1 = np.array(list(map(float, lines[0].split(",")))).reshape(self.hidden_size, self.input_size)
            self.b1 = np.array(list(map(float, lines[1].split(","))))
            self.w2 = np.array(list(map(float, lines[2].split(",")))).reshape(self.output_size, self.hidden_size)
            self.b2 = np.array(list(map(float, lines[3].split(","))))

def crossover(p1, p2):
    child = NeuralNetwork()
    for attr in ['w1', 'b1', 'w2', 'b2']:
        gene1 = getattr(p1, attr)
        gene2 = getattr(p2, attr)
        mask = np.random.rand(*gene1.shape) > 0.5
        setattr(child, attr, np.where(mask, gene1, gene2))
    return child

def mutate(nn, rate=0.1):
    for attr in ['w1', 'b1', 'w2', 'b2']:
        param = getattr(nn, attr)
        noise = np.random.randn(*param.shape) * rate
        mutated = param + noise
        setattr(nn, attr, np.clip(mutated, -5, 5))

# === PYGAME SETUP ===
pygame.init()
screen = pygame.display.set_mode((576, 1024))
clock = pygame.time.Clock()
game_font = pygame.font.Font('04B_19.ttf', 40)

# === ASSETS ===
gravity = 0.25
bird_movement = 0
game_active = True
score = 0
high_score = 0
death_timer = 0

bg_surface = pygame.transform.scale2x(pygame.image.load('assets/background-day.png').convert())
floor_surface = pygame.transform.scale2x(pygame.image.load('assets/base.png').convert())
floor_x_pos = 0

bird_downflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-downflap.png').convert_alpha())
bird_midflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-midflap.png').convert_alpha())
bird_upflap = pygame.transform.scale2x(pygame.image.load('assets/bluebird-upflap.png').convert_alpha())
bird_frames = [bird_downflap, bird_midflap, bird_upflap]
bird_index = 0
bird_surface = bird_frames[bird_index]
bird_rect = bird_surface.get_rect(center=(100, 512))

pipe_surface = pygame.transform.scale2x(pygame.image.load('assets/pipe-green.png'))
pipe_list = []
pipe_height = [400, 600, 800]

game_over_surface = pygame.transform.scale2x(pygame.image.load('assets/message.png').convert_alpha())
game_over_rect = game_over_surface.get_rect(center=(288, 512))

# === GAME FUNCTIONS ===
def draw_floor():
    screen.blit(floor_surface, (floor_x_pos, 900))
    screen.blit(floor_surface, (floor_x_pos + 576, 900))

def create_pipe():
    pos = random.choice(pipe_height)
    GAP_SIZE = 300
    return pipe_surface.get_rect(midtop=(700, pos)), pipe_surface.get_rect(midbottom=(700, pos - GAP_SIZE))

def move_pipes(pipes):
    for pipe in pipes:
        pipe.centerx -= 5
    return pipes

def remove_pipes(pipes):
    return [p for p in pipes if p.right > -50]

def draw_pipes(pipes):
    for pipe in pipes:
        if pipe.bottom >= 1024:
            screen.blit(pipe_surface, pipe)
        else:
            flip = pygame.transform.flip(pipe_surface, False, True)
            screen.blit(flip, pipe)

def check_collision(pipes):
    for pipe in pipes:
        if bird_rect.colliderect(pipe):
            return False
    if bird_rect.top <= -100 or bird_rect.bottom >= 900:
        return False
    return True

def init_game():
    global bird_rect, bird_movement, pipe_list, score, game_active, death_timer
    bird_rect.center = (100, 512)
    bird_movement = 0
    pipe_list.clear()
    score = 0
    game_active = True
    death_timer = 0

def simulate(nn, render=True):
    init_game()
    frame = 0
    scored_pipes = set()
    while True:
        if render:
            screen.blit(bg_surface, (0, 0))

        global bird_movement, score, game_active, death_timer
        if game_active:
            frame += 1
            bird_movement += gravity
            bird_rect.centery += bird_movement
            if frame % 75 == 0:
                pipe_list.extend(create_pipe())
            pipe_list[:] = move_pipes(pipe_list)
            pipe_list[:] = remove_pipes(pipe_list)
            if render:
                draw_pipes(pipe_list)

            if pipe_list:
                next_pipe = [p for p in pipe_list if p.centerx > bird_rect.centerx]
                pipe_x, pipe_y = (next_pipe[0].centerx, next_pipe[0].centery) if next_pipe else (700, 512)
            else:
                pipe_x, pipe_y = 700, 512

            gap_top = pipe_y - 150
            gap_bottom = pipe_y + 150

            inputs = np.array([
                bird_rect.centery / 1024,
                bird_movement / 20.0,
                (bird_rect.top - gap_top) / 512,
                (gap_bottom - bird_rect.bottom) / 512,
                (pipe_x - bird_rect.centerx) / 576
            ])

            if np.tanh(nn.forward(inputs)) > 0.2:
                bird_movement = -9

            game_active = check_collision(pipe_list)
            score += 1

            for pipe in pipe_list:
                pipe_id = id(pipe)
                if pipe.centerx < bird_rect.centerx and pipe_id not in scored_pipes:
                    score += 300
                    scored_pipes.add(pipe_id)

                    if abs(bird_rect.top - gap_top) < 10 or abs(bird_rect.bottom - gap_bottom) < 10:
                        score -= 50

            if render:
                screen.blit(bird_surface, bird_rect)
                draw_floor()
                pygame.display.update()
                clock.tick(120)
        else:
            death_timer += 1
            if death_timer > 60:
                break
    return score

def evolve(population, retain=0.2, mutate_rate=0.1):
    global gen
    scored = [(simulate(nn, render=False), nn) for nn in population]
    scored.sort(key=lambda x: x[0], reverse=True)
    print(f"Best score this generation: {scored[0][0]:.2f}")

    if SAVE:
        best_score = scored[0][0]
        with open(DATA_FILE, "a") as f:
            f.write(f"{gen},{best_score}\n")

    retain_len = int(len(scored) * retain)
    parents = [x[1] for x in scored[:retain_len]]
    children = []
    while len(children) + len(parents) < len(population):
        p1, p2 = random.sample(parents, 2)
        child = crossover(p1, p2)
        mutate(child, mutate_rate)
        children.append(child)
    return parents + children, scored[0][1]

# === SETUP LOGIC ===
if RESET:
    print("🧹 Reset flag enabled. Starting from scratch.")
    population = [NeuralNetwork() for _ in range(20)]
    best = None
    if SAVE and os.path.exists(SAVE_FILE):
        os.remove(SAVE_FILE)
    if SAVE and os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
else:
    if os.path.exists(SAVE_FILE):
        print("📂 Loading from save file.")
        best = NeuralNetwork()
        best.load_from_txt(SAVE_FILE)
        population = [best] + [crossover(best, NeuralNetwork()) for _ in range(19)]
    else:
        print("✨ No save found, starting fresh.")
        population = [NeuralNetwork() for _ in range(20)]
        best = None

# === MAIN TRAINING LOOP ===
total_generations = 3000
render_every = 50

for gen in range(start_gen, total_generations + 1):
    print(f"\n--- Generation {gen} ---")
    population, best = evolve(population)

    if SAVE:
        best.save_to_txt(SAVE_FILE)

    if VISUALIZE_AFTER and gen % render_every == 0:
        print(f"🎥 Visualizing best bird from generation {gen}")
        simulate(best, render=True)
