import pygame
import random
import math
from datetime import datetime
from pygame.math import Vector2

# 게임 화면 설정
WIDTH, HEIGHT = 1200, 800  # 화면 크기
FPS = 60  # 초당 프레임 수
MAX_CELLS = 1000  # 최대 세포 수
MAX_PREDATORS = 20  # 최대 포식자 수
SAFE_ZONE_RADIUS = 30  # 안전지대 반경
SPLIT_INTERVAL = 1.0  # 세포 분열 주기(초)
PREDATOR_SPAWN_DELAY = 5  # 포식자 생성 주기(초)
PREDATOR_MAX_SPEED = 200 # 포식자 이동 속도

# Pygame 초기화
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# 폰트 및 색상 정의
font = pygame.font.Font(None, 36)  # 기본 폰트
font_small = pygame.font.Font(None, 24)  # 작은 폰트
COLORS = {
    "background": (15, 15, 25),  # 배경색
    "safe_zone": (100, 200, 100),  # 안전지대 색상
    "hud": (200, 220, 240),  # HUD 텍스트 색상
    "tooltip_bg": (30, 30, 50, 200),  # 툴팁 배경
    "graph_increase": (255, 50, 50),  # 그래프 상승선
    "graph_decrease": (50, 150, 255),  # 그래프 하락선
    "graph_line": (100, 100, 100),  # 그래프 격자선
    "graph_bg": (30, 30, 50, 150),  # 그래프 배경
    "predator": (255, 255, 255)  # 포식자 색상
}

class Particle:
    """파티클 효과 관리 클래스"""
    def __init__(self, pos, color):
        self.pos = Vector2(pos)  # 초기 위치
        self.vel = Vector2(random.uniform(-5,5), random.uniform(-5,5))  # 이동 벡터
        self.color = color  # 파티클 색상
        self.lifespan = random.uniform(0.3, 0.6)  # 수명(초)
        self.age = 0.0  # 현재 연령
        self.size = 1  # 크기

    def update(self, dt):
        """파티클 상태 업데이트"""
        self.age += dt  # 시간 누적
        self.pos += self.vel * dt * 60  # 위치 업데이트
        self.vel *= 0.90  # 점점 느려짐
        return self.age < self.lifespan  # 수명 여부 반환

    def draw(self, surface):
        """파티클 그리기"""
        alpha = 255 * (1 - self.age/self.lifespan)  # 투명도 계산
        pygame.draw.circle(surface, self.color + (int(alpha),), 
                         self.pos, self.size)

class Cell:
    """세포 객체 클래스"""
    def __init__(self, generation=0, genes=None, pos=None):
        self.generation = generation  # 세대 수
        self.pos = pos if pos else Vector2(WIDTH//2, HEIGHT//2)  # 초기 위치
        self.genes = genes or self._init_genes()  # 유전자 정보
        self.direction = Vector2(0, -1).rotate(random.uniform(0, 360))  # 이동 방향
        self.dir_timer = 0.0  # 방향 변경 타이머
        self._apply_genes()  # 유전자 적용

    def _init_genes(self):
        """초기 유전자 생성"""
        return {
            'size': {
                'length': {'value': random.uniform(20, 30), 'mutation_rate': 0.5},
                'width_ratio': {'value': random.uniform(0.3, 0.6), 'mutation_rate': 0.05}
            },
            'speed': {'value': random.uniform(1.5, 3.0), 'mutation_rate': 0.15},
            'color': {
                'r': {'value': random.randint(0, 255), 'mutation_rate': 2.5},
                'g': {'value': random.randint(0, 255), 'mutation_rate': 2.5},
                'b': {'value': random.randint(0, 255), 'mutation_rate': 2.5},
            },
            'direction_change': {
                'interval': {'value': random.uniform(0.5, 2.0), 'mutation_rate': 0.1}
            },
            'mutation_rate': {'value': random.uniform(0.1, 0.3), 'mutation_rate': 0.05}
        }

    def _apply_genes(self):
        """유전자 정보 적용"""
        genes = self.genes
        self.length = genes['size']['length']['value']  # 세포 길이
        self.width = self.length * genes['size']['width_ratio']['value']  # 세포 너비
        self.speed = genes['speed']['value']  # 이동 속도
        self.color = (  # 색상 값 적용
            int(genes['color']['r']['value']),
            int(genes['color']['g']['value']),
            int(genes['color']['b']['value'])
        )
        self.collision_radius = self.length * 0.6  # 충돌 판정 반경

    def mutate_gene(self, gene):
        """유전자 변이 처리"""
        mutation = gene['mutation_rate'] * random.uniform(-2, 2) # 변이 값
        return gene['value'] + mutation

    def create_child_genes(self):
        """자식 세포 유전자 생성"""
        new_genes = {}
        for category, genes in self.genes.items():
            if isinstance(genes, dict) and 'value' in genes:  # 단일 값 유전자
                new_value = self.mutate_gene(genes)
                if category == 'mutation_rate':
                    new_value = max(0.05, min(0.5, new_value))  # 변이율 범위 제한
                new_genes[category] = {
                    'value': new_value,
                    'mutation_rate': genes['mutation_rate']
                }
            else:  # 복합 구조 유전자
                new_genes[category] = {}
                for sub_gene, params in genes.items():
                    mutated_value = self.mutate_gene(params)
                    # 값 범위 제한
                    if category == 'color':
                        mutated_value = max(0, min(255, mutated_value))
                    elif category == 'size' and sub_gene == 'width_ratio':
                        mutated_value = max(0.1, min(1.0, mutated_value))
                    new_genes[category][sub_gene] = {
                        'value': mutated_value,
                        'mutation_rate': params['mutation_rate']
                    }
        return new_genes

    def update(self, dt):
        """세포 상태 업데이트"""
        self.dir_timer += dt
        # 주기적으로 방향 변경
        if self.dir_timer >= self.genes['direction_change']['interval']['value']:
            angle = random.uniform(-30, 30)
            self.direction = self.direction.rotate(angle)
            self.dir_timer = 0

        # 벽 충돌 처리
        next_pos = self.pos + self.direction * self.speed
        if next_pos.x - self.collision_radius < 0 or next_pos.x + self.collision_radius > WIDTH:
            normal = Vector2(1, 0) if next_pos.x < WIDTH/2 else Vector2(-1, 0)
            self.direction = self.direction.reflect(normal)
        if next_pos.y - self.collision_radius < 0 or next_pos.y + self.collision_radius > HEIGHT:
            normal = Vector2(0, 1) if next_pos.y < HEIGHT/2 else Vector2(0, -1)
            self.direction = self.direction.reflect(normal)

        # 위치 업데이트
        self.pos += self.direction * self.speed
        # 화면 경계 제한
        self.pos.x = max(self.collision_radius, min(WIDTH - self.collision_radius, self.pos.x))
        self.pos.y = max(self.collision_radius, min(HEIGHT - self.collision_radius, self.pos.y))

    def draw(self, surface):
        """세포 렌더링"""
        width_int = max(10, int(self.width))
        length_int = max(10, int(self.length))
        base_image = pygame.Surface((width_int, length_int), pygame.SRCALPHA)
        pygame.draw.rect(base_image, self.color, (0, 0, self.width, self.length))
        # 이동 방향에 따른 회전
        target_angle = math.degrees(math.atan2(-self.direction.y, self.direction.x)) - 90
        rotated_image = pygame.transform.rotate(base_image, target_angle)
        rotated_rect = rotated_image.get_rect(center=self.pos)
        surface.blit(rotated_image, rotated_rect.topleft)

    def split(self):
        """세포 분열 처리"""
        return [Cell(
            generation=self.generation + 1,
            genes=self.create_child_genes(),
            pos=self.pos.copy()
        ) for _ in range(2)]

    def get_info(self):
        """툴팁 정보 생성"""
        return [
            f"Generation: {self.generation}",
            f"Length: {self.length:.1f}px (Δ{self.genes['size']['length']['mutation_rate']*100:.1f}%)",
            f"Width: {self.width:.1f}px ({self.genes['size']['width_ratio']['value']*100:.1f}%)",
            f"Speed: {self.speed:.1f}px/s (Δ{self.genes['speed']['mutation_rate']*100:.1f}%)",
            f"Dir Change: {self.genes['direction_change']['interval']['value']:.1f}s",
            f"Color RGB: ({self.genes['color']['r']['value']:.0f}, "
                      f"{self.genes['color']['g']['value']:.0f}, "
                      f"{self.genes['color']['b']['value']:.0f})",
            f"Mutation Rate: {self.genes['mutation_rate']['value']*100:.1f}%",
            f"Collision Radius: {self.collision_radius:.1f}px"
        ]

class Predator:
    """포식자 객체 클래스"""
    def __init__(self, pos):
        self.pos = Vector2(pos)  # 초기 위치
        self.vel = Vector2(0, -1).rotate(random.uniform(0, 360)).normalize() * PREDATOR_MAX_SPEED  # 이동 벡터
        self.size = 35  # 정사각형 크기
        self.color = COLORS["predator"]  # 색상
        self.collision_radius = self.size * 0.6  # 충돌 판정 반경
        self.safe_zone_pos = Vector2(WIDTH//2, HEIGHT//2)  # 안전지대 위치
        self.safe_zone_radius = SAFE_ZONE_RADIUS  # 안전지대 반경

    def update(self, dt):
        """포식자 상태 업데이트"""
        # 안전지대 충돌 검사
        safe_dist = self.pos.distance_to(self.safe_zone_pos)
        repel_dir = None
        
        if safe_dist < self.safe_zone_radius + self.collision_radius:
            repel_dir = (self.pos - self.safe_zone_pos).normalize()
            self.vel = self.vel.reflect(repel_dir) * 1.2  # 반사 + 가속

        # 포식자 속도 제한
        if self.vel.length() > PREDATOR_MAX_SPEED:
            self.vel.scale_to_length(PREDATOR_MAX_SPEED)

        # 벽 충돌 처리
        next_pos = self.pos + self.vel * dt  # 프레임 영향 제거
        if next_pos.x - self.collision_radius < 0 or next_pos.x + self.collision_radius > WIDTH:
            normal = Vector2(1, 0) if next_pos.x < WIDTH/2 else Vector2(-1, 0)
            self.vel = self.vel.reflect(normal)
        if next_pos.y - self.collision_radius < 0 or next_pos.y + self.collision_radius > HEIGHT:
            normal = Vector2(0, 1) if next_pos.y < HEIGHT/2 else Vector2(0, -1)
            self.vel = self.vel.reflect(normal)

        # 포식자 속도 제한
        if self.vel.length() > PREDATOR_MAX_SPEED:
            self.vel.scale_to_length(PREDATOR_MAX_SPEED)

        # 위치 업데이트
        self.pos += self.vel * dt  # 프레임 영향 제거
        
        # 안전지대 내부 진입 방지
        current_safe_dist = self.pos.distance_to(self.safe_zone_pos)
        if current_safe_dist < self.safe_zone_radius + self.collision_radius:
            if repel_dir is None:  # 반사 방향 재계산
                repel_dir = (self.pos - self.safe_zone_pos).normalize()
            self.pos += repel_dir * 2  # 밀어내기

        # 화면 경계 제한
        self.pos.x = max(self.collision_radius, min(WIDTH - self.collision_radius, self.pos.x))
        self.pos.y = max(self.collision_radius, min(HEIGHT - self.collision_radius, self.pos.y))

    def draw(self, surface):
        """포식자 렌더링"""
        base_image = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
        pygame.draw.rect(base_image, self.color, (0, 0, self.size, self.size))
        # 이동 방향에 따른 회전
        target_angle = math.degrees(math.atan2(-self.vel.y, self.vel.x)) - 90
        rotated_image = pygame.transform.rotate(base_image, target_angle)
        rotated_rect = rotated_image.get_rect(center=self.pos)
        surface.blit(rotated_image, rotated_rect.topleft)

def main():
    """메인 게임 루프"""
    cells = [Cell()]  # 초기 세포
    predators = []  # 포식자 리스트
    particles = []  # 파티클 리스트
    safe_zone_pos = Vector2(WIDTH//2, HEIGHT//2)  # 안전지대 위치
    start_time = datetime.now()  # 게임 시작 시간
    game_start_time = pygame.time.get_ticks() / 1000  # 타이머 시작 시간
    split_timer = 0.0  # 분열 타이머
    hovered_cell = None  # 호버링된 세포

    # 그래프 관련 설정
    graph_data = []
    max_graph_points = 100
    graph_rect = pygame.Rect(WIDTH-320, 10, 300, 150)

    running = True
    while running:
        # 시간 관리
        current_time = pygame.time.get_ticks() / 1000
        dt = clock.tick(FPS) / 1000  # 델타타임 계산
        
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # 화면 초기화
        screen.fill(COLORS["background"])

        # 안전지대 렌더링
        pygame.draw.circle(screen, COLORS["safe_zone"], 
                         (int(safe_zone_pos.x), int(safe_zone_pos.y)), 
                         SAFE_ZONE_RADIUS, 3)

        # 세포 분열 처리
        split_timer += dt
        if split_timer >= SPLIT_INTERVAL:
            split_timer = 0
            current_count = len(cells)
            if 0 < current_count < MAX_CELLS:
                split_count = min(MAX_CELLS - current_count, current_count)
                split_candidates = random.sample(cells, split_count)
                new_cells = []
                dead_cells = []
                for cell in split_candidates:
                    new_cells.extend(cell.split())
                    dead_cells.append(cell)
                    # 파티클 생성
                    particles.extend([Particle(cell.pos, cell.color) for _ in range(3)])
                # 세포 리스트 업데이트
                cells = [c for c in cells if c not in dead_cells] + new_cells

        # 포식자 생성
        if current_time - game_start_time >= PREDATOR_SPAWN_DELAY:
            if len(predators) < MAX_PREDATORS and random.random() < 0.5:
                spawn_pos = Vector2(
                    random.randint(WIDTH-200, WIDTH-50),
                    random.randint(HEIGHT-200, HEIGHT-50)
                )
                predators.append(Predator(pos=spawn_pos))

        # 세포 업데이트 및 렌더링
        mouse_pos = Vector2(pygame.mouse.get_pos())
        hovered_cell = None
        for cell in cells:
            cell.update(dt)
            # 마우스 호버링 검사
            if cell.pos.distance_to(mouse_pos) < cell.collision_radius:
                hovered_cell = cell
            cell.draw(screen)

        # 포식자 업데이트 및 렌더링
        for predator in predators:
            predator.update(dt)
            predator.draw(screen)

        # 충돌 검사
        dead_cells = []
        for predator in predators:
            # 포식자 충돌 영역 계산
            pred_rect = pygame.Rect(
                int(predator.pos.x - predator.size/2),
                int(predator.pos.y - predator.size/2),
                predator.size,
                predator.size
            )
            
            for cell in cells[:]:
                # 가장 가까운 점 계산
                closest_x = max(pred_rect.left, min(cell.pos.x, pred_rect.right))
                closest_y = max(pred_rect.top, min(cell.pos.y, pred_rect.bottom))
                distance = math.hypot(cell.pos.x - closest_x, cell.pos.y - closest_y)
                
                if distance <= cell.collision_radius:
                    dead_cells.append(cell)
                    particles.extend([Particle(cell.pos, cell.color) for _ in range(5)])
        cells = [c for c in cells if c not in dead_cells]

        # 파티클 처리
        particles = [p for p in particles if p.update(dt)]
        for p in particles:
            p.draw(screen)

        # 툴팁 렌더링
        if hovered_cell:
            info_lines = hovered_cell.get_info()
            tooltip_width = 300
            tooltip_height = 20 + len(info_lines) * 18
            tooltip_surface = pygame.Surface((tooltip_width, tooltip_height), pygame.SRCALPHA)
            tooltip_surface.fill(COLORS["tooltip_bg"])
            
            # 화살표 그리기
            pygame.draw.polygon(tooltip_surface, COLORS["tooltip_bg"], 
                              [(10, 0), (20, 10), (0, 10)])
            
            # 텍스트 렌더링
            for i, line in enumerate(info_lines):
                text = font_small.render(line, True, COLORS["hud"])
                tooltip_surface.blit(text, (10, 5 + i * 18))
            
            # 위치 조정
            pos = mouse_pos + Vector2(20, 20)
            pos.x = min(pos.x, WIDTH - tooltip_width)
            pos.y = min(pos.y, HEIGHT - tooltip_height)
            screen.blit(tooltip_surface, pos)

        # HUD 렌더링
        elapsed = datetime.now() - start_time
        hud_text = [
            f"Time: {elapsed.seconds//60:02}:{elapsed.seconds%60:02}",
            f"FPS: {int(clock.get_fps())}",
            f"Cells: {len(cells)}/{MAX_CELLS}",
            f"Predators: {len(predators)}/{MAX_PREDATORS}",
            f"Gen(Max): {max(c.generation for c in cells) if cells else 0}"
        ]
        hud_surface = pygame.Surface((200, 160), pygame.SRCALPHA)
        hud_surface.fill((0, 0, 0, 150))
        screen.blit(hud_surface, (0, 0))
        for i, text in enumerate(hud_text):
            color = COLORS["hud"]
            if len(cells) <= 5 and (i == 2 or i == 4):  # Cells와 Gen 텍스트 색상 변경
                color = (255, 0, 0)
            screen.blit(font.render(text, True, color), (10, 10 + i * 30))

        # 그래프 렌더링
        graph_data.append(len(cells))
        if len(graph_data) > max_graph_points:
            graph_data = graph_data[-max_graph_points:]
        
        graph_surface = pygame.Surface((graph_rect.width, graph_rect.height), pygame.SRCALPHA)
        graph_surface.fill(COLORS["graph_bg"])
        max_y = MAX_CELLS * 1.2
        
        if len(graph_data) >= 2:
            x_step = graph_rect.width / (len(graph_data)-1)
            # 격자선 그리기
            grid_steps = list(range(0, int(max_y), 200)) + [int(max_y)]
            for y in grid_steps:
                y_pos = graph_rect.height - (y / max_y * graph_rect.height)
                pygame.draw.line(graph_surface, COLORS["graph_line"], 
                               (0, y_pos), (graph_rect.width, y_pos), 1)
            
            # 데이터 포인트 생성
            points = [(i * x_step, graph_rect.height - (v / max_y * graph_rect.height)) 
                     for i, v in enumerate(graph_data)]
            
            # 선분 그리기
            for i in range(1, len(points)):
                color = COLORS["graph_increase"] if graph_data[i] > graph_data[i-1] else COLORS["graph_decrease"]
                pygame.draw.line(graph_surface, color, points[i-1], points[i], 2)

        # 현재 값 표시
        value_text = font_small.render(f"Cells: {len(cells)}/{MAX_CELLS}", True, COLORS["hud"])
        graph_surface.blit(value_text, (10, 5))
        screen.blit(graph_surface, graph_rect.topleft)
        pygame.draw.rect(screen, COLORS["hud"], graph_rect, 1)

        # 화면 업데이트
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()