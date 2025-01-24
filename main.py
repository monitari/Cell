import pygame
import random
import math
from datetime import datetime
from pygame.math import Vector2
import heapq

# 게임 화면 설정
WIDTH, HEIGHT = 1200, 800
MAX_CELLS = 1000
MAX_PREDATORS = 25
SAFE_ZONE_RADIUS = 30
SPLIT_INTERVAL = 1.0
PREDATOR_SPAWN_DELAY = 6
PREDATOR_MAX_SPEED = 200
GRID_SIZE = 100
EXTINCTION_THRESHOLD = 50  # 멸종 위기 임계값

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 폰트 및 색상 정의
font = pygame.font.Font(None, 36)
font_small = pygame.font.Font(None, 24)
COLORS = {
    "background": (15, 15, 25),
    "safe_zone": (100, 200, 100),
    "hud": (200, 220, 240),
    "tooltip_bg": (30, 30, 50, 200),
    "graph_increase": (255, 50, 50),
    "graph_decrease": (50, 150, 255),
    "graph_line": (100, 100, 100),
    "graph_bg": (30, 30, 50, 150),
    "predator": (255, 255, 255)
}

class ParticlePool:
    def __init__(self, max_particles=1000):
        self.pool = [Particle((0,0), (0,0,0)) for _ in range(max_particles)]
        self.free = list(range(max_particles))
        heapq.heapify(self.free)
        
    def add(self, pos, color, speed_multiplier=1.0):
        if self.free:
            idx = heapq.heappop(self.free)
            self.pool[idx].reset(pos, color, speed_multiplier)
            return True
        return False
    
    def update(self, dt):
        for idx, p in enumerate(self.pool):
            if p.active:
                if not p.update(dt):
                    p.active = False
                    heapq.heappush(self.free, idx)
    
    def draw(self, surface):
        for p in self.pool:
            if p.active: 
                p.draw(surface)

class Particle:
    def __init__(self, pos, color):
        self.reset(pos, color)
        
    def reset(self, pos, color, speed_multiplier=1.0):
        self.pos = Vector2(pos)
        self.vel = Vector2(
            random.uniform(-300, 300) * speed_multiplier,
            random.uniform(-300, 300) * speed_multiplier
        )
        self.color = color
        self.lifespan = random.uniform(0.1, 0.5)
        self.age = 0.0
        self.size = 1
        self.active = True

    def update(self, dt):
        self.age += dt
        self.pos += self.vel * dt
        self.vel *= 0.97
        return self.age < self.lifespan

    def draw(self, surface):
        alpha = 255 * (1 - self.age/self.lifespan)
        pygame.draw.circle(surface, self.color + (int(alpha),), (int(self.pos.x), int(self.pos.y)), self.size)

class SpatialGrid:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
        
    def clear(self):
        self.grid.clear()
        
    def add(self, obj):
        x, y = int(obj.pos.x//self.cell_size), int(obj.pos.y//self.cell_size)
        if (x, y) not in self.grid: 
            self.grid[(x, y)] = []
        self.grid[(x, y)].append(obj)
        
    def get_nearby(self, obj, radius):
        cells = set()
        cx = int(obj.pos.x // self.cell_size)
        cy = int(obj.pos.y // self.cell_size)
        search_range = int(math.ceil(radius / self.cell_size))
        
        for x in range(cx - search_range, cx + search_range + 1):
            for y in range(cy - search_range, cy + search_range + 1):
                if (x, y) in self.grid: 
                    cells.update(self.grid[(x, y)])
        return cells

def hsl_to_rgb(h, s, l):
    h = h % 360
    s = max(0, min(100, s)) / 100.0
    l = max(0, min(100, l)) / 100.0
    
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = l - c/2
    
    if 0 <= h < 60: r, g, b = c, x, 0
    elif 60 <= h < 120: r, g, b = x, c, 0
    elif 120 <= h < 180: r, g, b = 0, c, x
    elif 180 <= h < 240: r, g, b = 0, x, c
    elif 240 <= h < 300: r, g, b = x, 0, c
    else: r, g, b = c, 0, x
    
    return (
        round((r + m) * 255),
        round((g + m) * 255),
        round((b + m) * 255)
    )

class Cell:
    def __init__(self, generation=0, genes=None, pos=None):
        self.generation = generation
        self.pos = pos if pos else Vector2(WIDTH//2, HEIGHT//2)
        self.genes = genes or self._init_genes()
        self.direction = Vector2(0, -1).rotate(random.uniform(0, 360))
        self.dir_timer = 0.0
        self.rotated_image_cache = {}
        self._apply_genes()

    def _init_genes(self):
        return {
            'size': {
                'length': {
                    'value': random.uniform(45, 65),  # 기본 길이 (45~65px)
                    'mutation_rate': 1.8  # ±1.8% 세대당 변동
                },
                'width_ratio': {
                    'value': random.uniform(0.25, 0.6),  # 길이 대비 너비 비율 
                    'mutation_rate': 0.03  # ±0.03% 변동 (안정적 형태 유지)
                }
            },
            'speed': {
                'value': random.uniform(160, 220),  # 초기 속도 (160~220px/s)
                'mutation_rate': 6.0  # ±6.0% 변동 (적극적인 속도 진화)
            },
            'color': {
                'hue': {
                    'value': random.uniform(0, 360),  # 0~360도 색상각
                    'mutation_rate': 3.5  # ±3.5도 변동 (시각적 다양성)
                },
                'saturation': {  # 채도 고정
                    'value': 80, 
                    'mutation_rate': 0.0
                },
                'lightness': {  # 명도 고정
                    'value': 50, 
                    'mutation_rate': 0.0
                }
            },
            'direction_change': {
                'interval': {
                    'value': random.uniform(0.8, 1.5),  # 방향전환 주기 (0.8~1.5초)
                    'mutation_rate': 0.12  # ±0.12초 변동 (적당한 행동 변화)
                }
            },
            'mutation_rate': {  # 변이율 자체의 변이 설정
                'value': random.uniform(0.15, 0.25),  # 초기 변이율 (15~25%)
                'mutation_rate': 0.008  # ±0.008% 변동 (변이율 급변 방지)
            }
        }

    def _apply_genes(self):
        genes = self.genes
        self.length = max(3, genes['size']['length']['value'])
        self.width = max(3, self.length * genes['size']['width_ratio']['value'])
        self.speed = abs(genes['speed']['value'])
        h = genes['color']['hue']['value']
        s = genes['color']['saturation']['value']
        l = genes['color']['lightness']['value']
        self.color = hsl_to_rgb(h, s, l)
        self.collision_radius = self.length * 0.6
        self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90
        genes['direction_change']['interval']['value'] = max(0.01, genes['direction_change']['interval']['value'])

    def mutate_gene(self, gene):
        if isinstance(gene, dict):
            mutation = gene['mutation_rate'] * random.uniform(-1, 1)
            return gene['value'] + mutation
        return gene

    def create_child_genes(self):
        new_genes = {}
        for category, genes in self.genes.items():
            if isinstance(genes, dict) and 'value' in genes:
                new_genes[category] = {
                    'value': self.mutate_gene(genes),
                    'mutation_rate': genes['mutation_rate']
                }
            else:
                new_genes[category] = {}
                for sub_gene, params in genes.items():
                    new_genes[category][sub_gene] = {
                        'value': self.mutate_gene(params),
                        'mutation_rate': params['mutation_rate']
                    }
        return new_genes

    def update(self, dt):
        self.dir_timer += dt
        if self.dir_timer >= self.genes['direction_change']['interval']['value']:
            angle = random.uniform(-30, 30)
            self.direction = self.direction.rotate(angle)
            self.dir_timer = 0
            self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90

        next_pos = self.pos + self.direction * self.speed * dt
        if next_pos.x - self.collision_radius < 0:
            next_pos.x = self.collision_radius
            self.direction.x *= -1
            self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90
        elif next_pos.x + self.collision_radius > WIDTH:
            next_pos.x = WIDTH - self.collision_radius
            self.direction.x *= -1
            self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90

        if next_pos.y - self.collision_radius < 0:
            next_pos.y = self.collision_radius
            self.direction.y *= -1
            self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90
        elif next_pos.y + self.collision_radius > HEIGHT:
            next_pos.y = HEIGHT - self.collision_radius
            self.direction.y *= -1
            self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90

        self.pos = Vector2(
            max(self.collision_radius, min(WIDTH - self.collision_radius, next_pos.x)),
            max(self.collision_radius, min(HEIGHT - self.collision_radius, next_pos.y))
        )

    def draw(self, surface):
        width = int(self.width)
        length = int(self.length)
        angle = round(self.angle * 2) / 2

        if angle not in self.rotated_image_cache:
            base = pygame.Surface((width, length), pygame.SRCALPHA)
            pygame.draw.rect(base, self.color, (0, 0, width, length))
            self.rotated_image_cache[angle] = pygame.transform.rotate(base, angle)
        
        image = self.rotated_image_cache[angle]
        rect = image.get_rect(center=self.pos)
        surface.blit(image, rect.topleft)

    def split(self, mode="self"):
        children = []
        count = 2 if mode == "death" else 1
        offset_range = 5  # 분열 시 위치 오프셋 범위

        for _ in range(count):
            child = Cell(
                generation=self.generation + 1,
                genes=self.create_child_genes(),
                pos=self.pos + Vector2(
                    random.uniform(-offset_range, offset_range),
                    random.uniform(-offset_range, offset_range)
                )
            )
            child.pos.x = max(child.collision_radius, min(WIDTH - child.collision_radius, child.pos.x))
            child.pos.y = max(child.collision_radius, min(HEIGHT - child.collision_radius, child.pos.y))
            children.append(child)
        return children

    def get_info(self):
        return [
            f"Generation: {self.generation}",
            f"Length: {self.length:.2f}px (Δ{self.genes['size']['length']['mutation_rate']*100:.1f}%)",
            f"Width: {self.width:.2f}px ({self.genes['size']['width_ratio']['value']*100:.1f}%)",
            f"Speed: {self.speed:.2f}px/s (Δ{self.genes['speed']['mutation_rate']*100:.1f}%)",
            f"Dir Change: {self.genes['direction_change']['interval']['value']:.2f}s (Δ{self.genes['direction_change']['interval']['mutation_rate']*100:.1f}%)",
            f"Color HSL: ({self.genes['color']['hue']['value']:.0f}°, "
            f"{self.genes['color']['saturation']['value']}%, "
            f"{self.genes['color']['lightness']['value']}%)",
            f"Mutation Rate: {self.genes['mutation_rate']['value']*100:.1f}%",
            f"Collision Radius: {self.collision_radius:.1f}px"
        ]

class Predator:
    __slots__ = ['pos', 'vel', 'size', 'color', 'collision_radius', 
                'safe_zone_pos', 'safe_zone_radius', 'angle', 'rotated_image_cache',
                'direction', 'dir_timer', 'direction_change_interval']

    def __init__(self, pos):
        self.pos = Vector2(pos)
        self.direction = Vector2(0, -1).rotate(random.uniform(0, 360)).normalize()
        self.vel = self.direction * PREDATOR_MAX_SPEED
        self.size = 35
        self.color = COLORS["predator"]
        self.collision_radius = self.size * 0.6
        self.safe_zone_pos = Vector2(WIDTH//2, HEIGHT//2)
        self.safe_zone_radius = SAFE_ZONE_RADIUS
        self.rotated_image_cache = {}
        self.angle = 0.0
        self.dir_timer = 0.0
        self.direction_change_interval = random.uniform(1.0, 3.0)  # 1~3초 간격으로 방향 변경

    def update(self, dt):
        # 주기적 방향 변경
        self.dir_timer += dt
        if self.dir_timer >= self.direction_change_interval:
            self.direction = self.direction.rotate(random.uniform(-30, 30))  # -30~30도 회전
            self.vel = self.direction * PREDATOR_MAX_SPEED
            self.dir_timer = 0
            self.direction_change_interval = random.uniform(0.0, 3.0)  # 0~3초 간격으로 방향 변경

        # 안전 구역 반발
        safe_dist = self.pos.distance_to(self.safe_zone_pos)
        min_allowed_dist = self.safe_zone_radius + self.collision_radius

        if safe_dist < min_allowed_dist:
            # 반발 방향 계산 및 위치 보정
            repel_dir = (self.pos - self.safe_zone_pos).normalize()
            self.direction = self.direction.reflect(repel_dir)
            self.vel = self.direction * PREDATOR_MAX_SPEED * 1.2
            
            # 위치 강제 조정 (중첩 방지)
            overlap = min_allowed_dist - safe_dist
            self.pos += repel_dir * (overlap + 5)  # 5px 추가 여유

        # 화면 순환 이동 (벽 통과 구현)
        self.pos += self.vel * dt
        
        # X축 순환
        if self.pos.x < -self.collision_radius: self.pos.x = WIDTH + self.collision_radius
        elif self.pos.x > WIDTH + self.collision_radius: self.pos.x = -self.collision_radius

        # Y축 순환
        if self.pos.y < -self.collision_radius: self.pos.y = HEIGHT + self.collision_radius
        elif self.pos.y > HEIGHT + self.collision_radius: self.pos.y = -self.collision_radius

        # 화면 경계 내 위치 보정 (선택적)
        self.pos.x = max(-self.collision_radius, min(WIDTH + self.collision_radius, self.pos.x))
        self.pos.y = max(-self.collision_radius, min(HEIGHT + self.collision_radius, self.pos.y))

    def draw(self, surface):
        self.angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90
        angle = round(self.angle * 2) / 2
        
        if angle not in self.rotated_image_cache:
            base = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.rect(base, self.color, (0, 0, self.size, self.size))
            self.rotated_image_cache[angle] = pygame.transform.rotate(base, angle)
        
        image = self.rotated_image_cache[angle]
        rect = image.get_rect(center=self.pos)
        surface.blit(image, rect.topleft)

def main():
    cells = [Cell()]
    predators = []
    particle_pool = ParticlePool(max_particles=5000)
    start_time = datetime.now()
    game_start_time = pygame.time.get_ticks() / 1000
    split_timer = 0.0
    graph_data = []
    graph_timer = 0.0
    graph_delay = 0.5  # 그래프 업데이트 주기 (0.5초)
    max_graph_points = 100  # 최대 표시 데이터 포인트 수
    graph_rect = pygame.Rect(WIDTH-320, 10, 300, 150)  # 그래프 위치 및 크기    
    grid = SpatialGrid(GRID_SIZE)

    running = True
    while running:
        dt = clock.tick() / 1000
        current_time = pygame.time.get_ticks() / 1000
        graph_timer += dt

        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = False

        # 분열 로직 (일반 분열)
        split_timer += dt
        if split_timer >= SPLIT_INTERVAL:
            split_timer = 0
            current_count = len(cells)
            if current_count < MAX_CELLS:
                available = MAX_CELLS - current_count
                split_candidates = random.sample(cells, min(len(cells), available))
                new_children = []
                
                for cell in split_candidates:
                    new_children.extend(cell.split(mode="self"))
                    if len(new_children) >= available:
                        break
                
                cells.extend(new_children[:available])

        # 세대 기반 분열 (death 모드)
        if cells:
            max_gen = max(c.generation for c in cells)
            split_cells = [c for c in cells if c.generation + 10 < max_gen] # 10세대 초과 세포 추출
            available = MAX_CELLS - len(cells)
            max_splits = min(len(split_cells), available)
            
            for cell in split_cells[:max_splits]:
                new_children = cell.split(mode="death")
                cells.extend(new_children)
                cells.remove(cell)
                if len(cells) >= MAX_CELLS: break

        # 포식자 생성
        if current_time - game_start_time >= PREDATOR_SPAWN_DELAY:
            if len(predators) < MAX_PREDATORS and random.random() < 0.5:
                spawn_pos = Vector2(
                    random.randint(WIDTH-200, WIDTH-50),
                    random.randint(HEIGHT-200, HEIGHT-50)
                )
                predators.append(Predator(spawn_pos))

        # 세포 업데이트
        for cell in cells[:]:
            cell.update(dt)
            if (cell.pos.x < -cell.collision_radius or 
                cell.pos.x > WIDTH + cell.collision_radius or
                cell.pos.y < -cell.collision_radius or 
                cell.pos.y > HEIGHT + cell.collision_radius):
                cells.remove(cell)

        # 포식자 업데이트
        for predator in predators:
            predator.update(dt)

        # 충돌 처리
        dead_cells = []
        grid.clear()
        for cell in cells: grid.add(cell)
        
        for predator in predators:
            nearby = grid.get_nearby(predator, predator.collision_radius + 50)
            for cell in nearby:
                if cell.pos.distance_to(predator.pos) <= cell.collision_radius + predator.collision_radius:
                    if cell not in dead_cells:
                        dead_cells.append(cell)
                        for _ in range(5):
                            particle_pool.add(cell.pos, cell.color)
        
        cells = [c for c in cells if c not in dead_cells]
        cells = cells[:MAX_CELLS]  # 최종 안전장치
        
        # 마우스 위치 추적 및 툴팁 대상 세포 찾기
        mouse_pos = Vector2(pygame.mouse.get_pos())
        hovered_cell = None
        for cell in cells:
            if cell.pos.distance_to(mouse_pos) < cell.collision_radius:
                hovered_cell = cell
                break
        
        # 렌더링
        screen.fill(COLORS["background"]) # 배경 초기화
        pygame.draw.circle(screen, COLORS["safe_zone"], (WIDTH//2, HEIGHT//2), SAFE_ZONE_RADIUS, 3)

        # 세포 & 포식자 렌더링
        for cell in cells: cell.draw(screen)
        for predator in predators: predator.draw(screen)
        particle_pool.update(dt)
        particle_pool.draw(screen)

        # 멸종 위기 경고 메시지
        if len(cells) < EXTINCTION_THRESHOLD:
            warning_text = font.render("WARNING: EXTINCTION RISK!", True, (255, 50, 50))
            if len(cells) == 0:
                warning_text = font.render("EXTINCTION! GAME OVER!", True, (255, 50, 50))
            warning_rect = warning_text.get_rect(center=(WIDTH // 2, 30))
            screen.blit(warning_text, warning_rect)

        # 그래프 데이터 업데이트
        if graph_timer >= graph_delay:
            graph_timer = 0
            graph_data.append(len(cells))
            if len(graph_data) > max_graph_points: graph_data = graph_data[-max_graph_points:]
        
        # === 그래프 렌더링 ===
        graph_surface = pygame.Surface((graph_rect.width, graph_rect.height), pygame.SRCALPHA)
        graph_surface.fill(COLORS["graph_bg"])
        
        # 그리드 라인
        max_y = MAX_CELLS * 1.2
        grid_steps = list(range(0, int(max_y), 200))
        for y in grid_steps:
            y_pos = graph_rect.height - (y/max_y * graph_rect.height)
            pygame.draw.line(graph_surface, COLORS["graph_line"], (0, y_pos), (graph_rect.width, y_pos), 1)

        # 데이터 라인
        if len(graph_data) >= 2:
            x_step = graph_rect.width / (len(graph_data)-1)
            points = []
            for i, v in enumerate(graph_data):
                x = i * x_step
                y = graph_rect.height - (v/max_y * graph_rect.height)
                points.append((x, y))

            for i in range(1, len(points)):
                prev = points[i-1]
                curr = points[i]
                color = COLORS["graph_increase"] if graph_data[i] > graph_data[i-1] else COLORS["graph_decrease"]
                pygame.draw.line(graph_surface, color, prev, curr, 2)

        # 현재 값 텍스트
        value_text = font_small.render(f"Cells: {len(cells)}/{MAX_CELLS}", True, COLORS["hud"])
        graph_surface.blit(value_text, (10, 5))
        
        # 메인 화면에 그래프 적용
        screen.blit(graph_surface, graph_rect.topleft)
        pygame.draw.rect(screen, COLORS["hud"], graph_rect, 1, border_radius=3)

        # 툴팁 렌더링
        if hovered_cell:
            info_lines = hovered_cell.get_info()
            tooltip_width = 270
            tooltip_height = 15 + len(info_lines)*20
            tooltip_surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
            tooltip_surface.fill(COLORS["tooltip_bg"])
            
            # 삼각형 포인터
            pygame.draw.polygon(tooltip_surface, COLORS["tooltip_bg"], [(15, 5), (25, 15), (5, 15)])
            
            # 텍스트 렌더링
            for i, line in enumerate(info_lines):
                text = font_small.render(line, True, COLORS["hud"])
                tooltip_surface.blit(text, (10, 10 + i*20))
            
            # 화면에 배치
            tooltip_pos = mouse_pos + Vector2(20, 20)
            tooltip_pos.x = min(tooltip_pos.x, WIDTH - tooltip_width)
            tooltip_pos.y = min(tooltip_pos.y, HEIGHT - tooltip_height)
            screen.blit(tooltip_surface, tooltip_pos)

        # HUD 표시
        elapsed = datetime.now() - start_time
        total_seconds = int(elapsed.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        gen_range = (
            f"{min(c.generation for c in cells)}~{max(c.generation for c in cells)}"
            if cells else "0~0"
        )
        hud_text = [
            f"Time: {hours:02}:{minutes:02}:{seconds:02}",
            f'FPS: {int(clock.get_fps())}',
            f"Cells: {len(cells)}/{MAX_CELLS}",
            f"Predators: {len(predators)}/{MAX_PREDATORS}",
            f"Generation: {gen_range}"
        ]
        
        hud_bg = pygame.Surface((300, 160), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 150))
        screen.blit(hud_bg, (10, 0))
        
        for i, text in enumerate(hud_text):
            color = COLORS["graph_increase"] if "Cells" in text and len(cells) <= EXTINCTION_THRESHOLD else COLORS["hud"]
            screen.blit(font.render(text, True, color), (20, 10 + i*30))

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()