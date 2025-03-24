import numpy as np
from scipy.ndimage import convolve

class GridBasedModel:
    """격자 기반 포식자-피식자 모델 (반응-확산 시스템)"""
    
    def __init__(self, config):
        """
        Parameters:
        config: 설정 딕셔너리
        """
        # 공간 파라미터
        self.L = config['L']  # 실제 공간 크기
        self.n_grid = config['n_grid']  # 시각화용 격자선
        
        # 이동 관련 설정
        self.prey_move_percent = config['prey_move_percent']
        self.predator_move_percent = config['predator_move_percent']
        self.boundary_condition = config['boundary_condition']
        self.movement_pattern = config['movement_pattern']
        
        # Lotka-Volterra 파라미터
        self.K = config['K']  # 환경수용력
        self.beta = config['beta']  # 포식률
        
        # 격자 초기화 (2D arrays) - L x L 크기로 초기화
        self.prey_density = np.zeros((self.L, self.L))
        self.predator_density = np.zeros((self.L, self.L))
        
        # 초기 밀도 설정
        self._initialize_densities(config)
        
        self.current_step = 0
    
    def _initialize_densities(self, config):
        """초기 밀도 분포 설정"""
        n_prey = config['n_prey']
        n_predator = config['n_predator']
        n_prey_location = config['n_prey_location']
        n_predator_location = config['n_predator_location']
        
        # 피식자 초기 분포: n_prey_location개의 무작위 위치에 균등 분포
        prey_per_location = n_prey // n_prey_location
        remaining_prey = n_prey % n_prey_location
        
        # 무작위 위치 선택 (L x L 공간에서)
        prey_locations = []
        while len(prey_locations) < n_prey_location:
            cx = np.random.randint(0, self.L)
            cy = np.random.randint(0, self.L)
            if (cx, cy) not in prey_locations:
                prey_locations.append((cx, cy))
                # 마지막 위치에는 남은 개체도 추가
                if len(prey_locations) == n_prey_location:
                    self.prey_density[cx, cy] = prey_per_location + remaining_prey
                else:
                    self.prey_density[cx, cy] = prey_per_location
        
        # 포식자 초기 분포: n_predator_location개의 무작위 위치에 균등 분포
        predator_per_location = n_predator // n_predator_location
        remaining_predator = n_predator % n_predator_location
        
        # 무작위 위치 선택 (L x L 공간에서)
        predator_locations = []
        while len(predator_locations) < n_predator_location:
            cx = np.random.randint(0, self.L)
            cy = np.random.randint(0, self.L)
            if (cx, cy) not in predator_locations:
                predator_locations.append((cx, cy))
                # 마지막 위치에는 남은 개체도 추가
                if len(predator_locations) == n_predator_location:
                    self.predator_density[cx, cy] = predator_per_location + remaining_predator
                else:
                    self.predator_density[cx, cy] = predator_per_location

        # 디버그 출력 추가
        print(f"Initial setup:")
        print(f"Prey - Total: {np.sum(self.prey_density)}, Expected: {n_prey}")
        print(f"Predator - Total: {np.sum(self.predator_density)}, Expected: {n_predator}")
        print(f"Max prey density: {np.max(self.prey_density)}")
        print(f"Max predator density: {np.max(self.predator_density)}")
    
    def get_movement_choices_and_probs(self, i, j, move_percent, movement_pattern):
        """이동 방향과 확률 계산
        
        Parameters:
        i, j: 현재 위치
        move_percent: 이동 확률
        movement_pattern: 이동 패턴 ('4dir', '8dir', '8dir_weighted')
        
        Returns:
        choices: 가능한 이동 위치 리스트
        probs: 각 위치로 이동할 확률 리스트
        """
        if movement_pattern == '4dir':
            choices = [(i, j), (i-1, j), (i, j+1), (i+1, j), (i, j-1)]
            probs = [1-move_percent] + [move_percent/4] * 4
            
        elif movement_pattern == '8dir':
            choices = [(i, j), 
                      (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
                      (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1)]
            probs = [1-move_percent] + [move_percent/8] * 8
            
        else:  # 8dir_weighted
            choices = [(i, j), 
                      (i-1, j), (i-1, j+1), (i, j+1), (i+1, j+1),
                      (i+1, j), (i+1, j-1), (i, j-1), (i-1, j-1)]
            straight = move_percent / (4 + 4/np.sqrt(2))
            diagonal = straight / np.sqrt(2)
            probs = [1-move_percent] + [straight, diagonal, straight, diagonal,
                                      straight, diagonal, straight, diagonal]
        
        return choices, probs
    
    def apply_boundary_condition(self, pos):
        """경계 조건 적용
        
        Parameters:
        pos: (i, j) 위치 튜플
        
        Returns:
        new_i, new_j: 경계 조건이 적용된 새로운 위치. None, None은 개체가 사라짐을 의미
        """
        i, j = pos
        
        if self.boundary_condition == 'periodic':
            return i % self.L, j % self.L
            
        elif self.boundary_condition == 'neumann':
            return np.clip(i, 0, self.L-1), np.clip(j, 0, self.L-1)
            
        else:  # dirichlet
            if 0 <= i < self.L and 0 <= j < self.L:
                return i, j
            return None, None
    
    def _compute_derivatives(self, u, v):
        """Lotka-Volterra 방정식의 미분값 계산
        u̇ = u - u²/K - (βu/(u+v))v     # 로지스틱 성장 - 포식
        v̇ = (βu/(u+v))v - v            # 포식을 통한 성장 - 자연사
        """
        # 수치적 안정성을 위한 작은 상수
        eps = 1e-10
        
        # 로지스틱 성장과 포식 상호작용
        du = u - (u * u) / self.K - (self.beta * u / (u + v + eps)) * v
        dv = (self.beta * u / (u + v + eps)) * v - v
        
        return du, dv

    def rk4_step(self, u, v):
        """4차 Runge-Kutta method로 다음 상태 계산 (step 단위)"""
        # k1 계산
        k1_u, k1_v = self._compute_derivatives(u, v)
        
        # k2 계산 (step의 1/2 지점)
        u2 = u + k1_u/2
        v2 = v + k1_v/2
        k2_u, k2_v = self._compute_derivatives(u2, v2)
        
        # k3 계산 (step의 1/2 지점)
        u3 = u + k2_u/2
        v3 = v + k2_v/2
        k3_u, k3_v = self._compute_derivatives(u3, v3)
        
        # k4 계산 (step의 끝 지점)
        u4 = u + k3_u
        v4 = v + k3_v
        k4_u, k4_v = self._compute_derivatives(u4, v4)
        
        # 다음 상태 계산
        u_next = u + (k1_u + 2*k2_u + 2*k3_u + k4_u) / 6
        v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
        
        return u_next, v_next

    def _apply_boundary_movement(self, stayed, moves):
        """경계 조건에 따른 이동 처리
        
        Parameters:
        stayed: 현재 위치에 남은 개체
        moves: 각 방향으로 이동하는 개체 리스트 [(up, right, down, left)] 또는 [(up, upright, right, downright, down, downleft, left, upleft)]
        
        Returns:
        new_population: 이동 후 개체군 분포
        """
        if self.boundary_condition == 'periodic':
            new_population = stayed.copy()
            if len(moves) == 4:  # 4방향
                up, right, down, left = moves
                new_population += (np.roll(up, 1, axis=0) +
                                 np.roll(right, -1, axis=1) +
                                 np.roll(down, -1, axis=0) +
                                 np.roll(left, 1, axis=1))
            else:  # 8방향
                up, upright, right, downright, down, downleft, left, upleft = moves
                new_population += (np.roll(up, 1, axis=0) +
                                 np.roll(upright, (1, -1), axis=(0, 1)) +
                                 np.roll(right, -1, axis=1) +
                                 np.roll(downright, (-1, -1), axis=(0, 1)) +
                                 np.roll(down, -1, axis=0) +
                                 np.roll(downleft, (-1, 1), axis=(0, 1)) +
                                 np.roll(left, 1, axis=1) +
                                 np.roll(upleft, (1, 1), axis=(0, 1)))
                
        elif self.boundary_condition == 'neumann':
            new_population = stayed.copy()
            if len(moves) == 4:  # 4방향
                up, right, down, left = moves
                # 상하 이동
                new_population[1:, :] += up[:-1, :]
                new_population[0, :] += up[0, :]  # 반사
                new_population[:-1, :] += down[1:, :]
                new_population[-1, :] += down[-1, :]  # 반사
                # 좌우 이동
                new_population[:, :-1] += right[:, 1:]
                new_population[:, -1] += right[:, -1]  # 반사
                new_population[:, 1:] += left[:, :-1]
                new_population[:, 0] += left[:, 0]  # 반사
            else:  # 8방향
                up, upright, right, downright, down, downleft, left, upleft = moves
                # 상하좌우 이동
                new_population[1:, :] += up[:-1, :]
                new_population[0, :] += up[0, :]
                new_population[:-1, :] += down[1:, :]
                new_population[-1, :] += down[-1, :]
                new_population[:, :-1] += right[:, 1:]
                new_population[:, -1] += right[:, -1]
                new_population[:, 1:] += left[:, :-1]
                new_population[:, 0] += left[:, 0]
                # 대각선 이동
                new_population[1:, :-1] += upright[:-1, 1:]
                new_population[1:, 1:] += upleft[:-1, :-1]
                new_population[:-1, :-1] += downright[1:, 1:]
                new_population[:-1, 1:] += downleft[1:, :-1]
                # 경계에서의 대각선 반사
                new_population[0, :] += (upright[0, 1:] + upleft[0, :-1])
                new_population[-1, :] += (downright[-1, 1:] + downleft[-1, :-1])
                new_population[:, 0] += (upleft[:-1, 0] + downleft[1:, 0])
                new_population[:, -1] += (upright[:-1, -1] + downright[1:, -1])
                
        else:  # dirichlet
            new_population = stayed.copy()
            if len(moves) == 4:  # 4방향
                up, right, down, left = moves
                # 경계 밖으로 나가면 사라짐
                new_population[1:, :] += up[:-1, :]
                new_population[:-1, :] += down[1:, :]
                new_population[:, :-1] += right[:, 1:]
                new_population[:, 1:] += left[:, :-1]
            else:  # 8방향
                up, upright, right, downright, down, downleft, left, upleft = moves
                # 상하좌우
                new_population[1:, :] += up[:-1, :]
                new_population[:-1, :] += down[1:, :]
                new_population[:, :-1] += right[:, 1:]
                new_population[:, 1:] += left[:, :-1]
                # 대각선
                new_population[1:, :-1] += upright[:-1, 1:]
                new_population[1:, 1:] += upleft[:-1, :-1]
                new_population[:-1, :-1] += downright[1:, 1:]
                new_population[:-1, 1:] += downleft[1:, :-1]
                
        return new_population

    def _move_population(self, population, move_percent):
        """개체군 이동 처리
        
        Parameters:
        population: 현재 개체군 분포 (2D array)
        move_percent: 이동하는 비율
        
        Returns:
        new_population: 이동 후 개체군 분포
        """
        # 현재 위치에 남는 개체와 이동하는 개체 계산
        stayed = np.random.binomial(population, 1 - move_percent)
        moving = population - stayed
        
        # 각 위치에서 이동하는 개체들의 방향 결정
        if self.movement_pattern == '4dir':
            # 4방향의 균등한 확률
            probs = np.array([0.25, 0.25, 0.25, 0.25])  # up, right, down, left
            moves = [np.zeros_like(moving) for _ in range(4)]
            
            # 각 위치에서 이동하는 개체들의 방향을 multinomial로 결정
            for i in range(self.L):
                for j in range(self.L):
                    if moving[i, j] > 0:
                        move_counts = np.random.multinomial(moving[i, j], probs)
                        moves[0][i, j] = move_counts[0]  # up
                        moves[1][i, j] = move_counts[1]  # right
                        moves[2][i, j] = move_counts[2]  # down
                        moves[3][i, j] = move_counts[3]  # left
            
        elif self.movement_pattern == '8dir':
            # 8방향의 균등한 확률
            probs = np.array([0.125] * 8)  # up, upright, right, downright, down, downleft, left, upleft
            moves = [np.zeros_like(moving) for _ in range(8)]
            
            for i in range(self.L):
                for j in range(self.L):
                    if moving[i, j] > 0:
                        move_counts = np.random.multinomial(moving[i, j], probs)
                        for k in range(8):
                            moves[k][i, j] = move_counts[k]
            
        else:  # 8dir_weighted
            # 8방향 가중치 이동
            straight = 1 / (4 + 4/np.sqrt(2))
            diagonal = straight / np.sqrt(2)
            probs = np.array([straight, diagonal, straight, diagonal,
                            straight, diagonal, straight, diagonal])
            probs = probs / probs.sum()
            moves = [np.zeros_like(moving) for _ in range(8)]
            
            for i in range(self.L):
                for j in range(self.L):
                    if moving[i, j] > 0:
                        move_counts = np.random.multinomial(moving[i, j], probs)
                        for k in range(8):
                            moves[k][i, j] = move_counts[k]
        
        return self._apply_boundary_movement(stayed, moves)

    def step(self):
        """한 스텝 단위 시뮬레이션 진행"""
        # 1. Random walk
        prey = self.prey_density.astype(int)
        pred = self.predator_density.astype(int)
        
        # 피식자와 포식자 이동
        self.prey_density = self._move_population(prey, self.prey_move_percent)
        self.predator_density = self._move_population(pred, self.predator_move_percent)
        
        # 2. Lotka-Volterra 방정식 (RK4로 계산)
        mask = (self.prey_density > 0) | (self.predator_density > 0)
        if np.any(mask):
            u_next, v_next = self.rk4_step(
                self.prey_density[mask],
                self.predator_density[mask]
            )
            self.prey_density[mask] = u_next
            self.predator_density[mask] = v_next
        
        # 음수 방지
        self.prey_density = np.maximum(0, self.prey_density)
        self.predator_density = np.maximum(0, self.predator_density)
        
        # 매 10 스텝마다 디버그 출력
        if self.current_step % 10 == 0:
            print(f"\nStep {self.current_step}:")
            print(f"Prey - Total: {np.sum(self.prey_density):.2f}, Max: {np.max(self.prey_density):.2f}")
            print(f"Predator - Total: {np.sum(self.predator_density):.2f}, Max: {np.max(self.predator_density):.2f}")
        
        self.current_step += 1
        return self.get_stats()
    
    def get_stats(self):
        """현재 상태 통계"""
        return {
            'time': self.current_step,
            'prey_density': self.prey_density,
            'predator_density': self.predator_density,
            'total_prey': np.sum(self.prey_density),
            'total_predator': np.sum(self.predator_density),
            'grid_size': self.L
        }
    
    def get_grid_stats(self, x, y):
        """특정 격자의 상세 정보"""
        grid_x = int(x / self.dx)
        grid_y = int(y / self.dx)
        return {
            'prey_density': self.prey_density[grid_x, grid_y],
            'predator_density': self.predator_density[grid_x, grid_y]
        } 