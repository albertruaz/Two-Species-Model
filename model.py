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
        self.L = config['L']
        self.n_grid = self.L * 2  # 실제 시뮬레이션 격자 크기는 L*2
        
        # 이동 관련 설정
        self.prey_move_percent = config['prey_move_percent']
        self.predator_move_percent = config['predator_move_percent']
        
        # 반응 파라미터 (Lotka-Volterra)
        self.birth_rate_prey = config['birth_rate_prey']
        self.birth_rate_predator = config['birth_rate_predator']  # 포식자 번식률 추가
        self.death_rate_predator = config['death_rate_predator']
        self.interaction_rate = config['interaction_rate']
        
        # 환경수용력
        self.carrying_capacity = config['carrying_capacity']
        
        # 경계 조건 설정
        self.boundary_condition = config['boundary_condition']
        self.movement_pattern = config['movement_pattern']
        
        # 격자 초기화 (2D arrays)
        self.prey_density = np.zeros((self.n_grid, self.n_grid))
        self.predator_density = np.zeros((self.n_grid, self.n_grid))
        
        # 초기 밀도 설정 (지정된 위치에 균등 분포)
        # 피식자 초기 분포
        n_prey = config['n_prey']
        n_prey_location = config['n_prey_location']
        prey_per_location = n_prey // n_prey_location
        
        # 무작위로 위치 선택
        prey_locations = []
        while len(prey_locations) < n_prey_location:
            i = np.random.randint(0, self.n_grid)
            j = np.random.randint(0, self.n_grid)
            if (i, j) not in prey_locations:
                prey_locations.append((i, j))
                self.prey_density[i, j] = prey_per_location
        
        # 남은 개체 무작위 배치
        remaining_prey = n_prey % n_prey_location
        if remaining_prey > 0:
            i = np.random.randint(0, self.n_grid)
            j = np.random.randint(0, self.n_grid)
            self.prey_density[i, j] += remaining_prey
            
        # 포식자 초기 분포
        n_predator = config['n_predator']
        n_predator_location = config['n_predator_location']
        predator_per_location = n_predator // n_predator_location
        
        # 무작위로 위치 선택
        predator_locations = []
        while len(predator_locations) < n_predator_location:
            i = np.random.randint(0, self.n_grid)
            j = np.random.randint(0, self.n_grid)
            if (i, j) not in predator_locations:
                predator_locations.append((i, j))
                self.predator_density[i, j] = predator_per_location
        
        # 남은 개체 무작위 배치
        remaining_predator = n_predator % n_predator_location
        if remaining_predator > 0:
            i = np.random.randint(0, self.n_grid)
            j = np.random.randint(0, self.n_grid)
            self.predator_density[i, j] += remaining_predator
        
        self.current_step = 0
    
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
            return i % self.n_grid, j % self.n_grid
            
        elif self.boundary_condition == 'neumann':
            return np.clip(i, 0, self.n_grid-1), np.clip(j, 0, self.n_grid-1)
            
        else:  # dirichlet
            if 0 <= i < self.n_grid and 0 <= j < self.n_grid:
                return i, j
            return None, None
    
    def step(self):
        """RK4 method를 사용한 시뮬레이션 한 단계 진행"""
        dt = self.dt
        
        # 현재 상태
        P = self.prey_density
        Q = self.predator_density
        
        # RK4 계수 계산
        k1_P, k1_Q = self._compute_derivatives(P, Q)
        k2_P, k2_Q = self._compute_derivatives(P + dt/2 * k1_P, Q + dt/2 * k1_Q)
        k3_P, k3_Q = self._compute_derivatives(P + dt/2 * k2_P, Q + dt/2 * k2_Q)
        k4_P, k4_Q = self._compute_derivatives(P + dt * k3_P, Q + dt * k3_Q)
        
        # 다음 상태 계산
        self.prey_density = P + dt/6 * (k1_P + 2*k2_P + 2*k3_P + k4_P)
        self.predator_density = Q + dt/6 * (k1_Q + 2*k2_Q + 2*k3_Q + k4_Q)
        
        # 음수 방지
        self.prey_density = np.maximum(0, self.prey_density)
        self.predator_density = np.maximum(0, self.predator_density)
        
        self.time += dt
        return self.get_stats()
    
    def _compute_derivatives(self, P, Q):
        """미분 방정식의 우변 계산"""
        # 확산항 계산 (Laplacian)
        P_laplacian = self._compute_laplacian(P)
        Q_laplacian = self._compute_laplacian(Q)
        
        # 반응항 계산
        P_reaction = self.a * P * (1 - P/self.K) - self.b * P * Q
        Q_reaction = self.c * P * Q - self.d * Q
        
        # 확산항과 반응항 결합
        dP_dt = self.D_p * P_laplacian + P_reaction
        dQ_dt = self.D_q * Q_laplacian + Q_reaction
        
        return dP_dt, dQ_dt
    
    def get_stats(self):
        """현재 상태 통계"""
        return {
            'time': self.current_step,
            'prey_density': self.prey_density,
            'predator_density': self.predator_density,
            'total_prey': np.sum(self.prey_density),
            'total_predator': np.sum(self.predator_density),
            'grid_size': self.n_grid
        }
    
    def get_grid_stats(self, x, y):
        """특정 격자의 상세 정보"""
        grid_x = int(x / self.dx)
        grid_y = int(y / self.dx)
        return {
            'prey_density': self.prey_density[grid_x, grid_y],
            'predator_density': self.predator_density[grid_x, grid_y]
        } 