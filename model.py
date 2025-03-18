import random
import numpy as np
from scipy.ndimage import laplace

class Agent:
    """
    모든 개체(피식자/Prey, 포식자/Predator)를 포괄하는 기본 클래스.
    species: "Prey" 또는 "Predator"
    x, y: 2D 상에서의 위치
    alive: 생존 여부
    """
    def __init__(self, species, x, y):
        self.species = species
        self.x = x
        self.y = y
        self.alive = True
        self.age = 0
        self.reproduction_timer = 0  # 번식 지연을 위한 타이머

class ReactionDiffusionModel:
    """
    반응-확산 방정식을 사용한 2D 포식자-피식자 시뮬레이션
    """
    def __init__(self, width, height, config):
        """
        Parameters:
        width, height: 시뮬레이션 공간 크기
        config: 파라미터 설정 (딕셔너리)
            - du: 피식자 확산 계수
            - dv: 포식자 확산 계수
            - K: 피식자의 carrying capacity
            - cu: 피식자 수확 계수
            - cv: 포식자 수확 계수
            - dx: 공간 간격
            - T: 총 시뮬레이션 시간
        """
        self.width = width
        self.height = height
        self.config = config
        
        # 격자 간격과 시간 간격
        self.dx = config.get("dx", 0.2)
        self.du = config.get("du", 0.25)
        self.dv = config.get("dv", 0.05)
        self.K = config.get("K", 8)
        self.cu = config.get("cu", 0.2)
        self.cv = config.get("cv", 0.05)
        self.T = config.get("T", 150)  # 총 시뮬레이션 시간
        
        # 안정성을 위한 시간 간격 계산
        self.dt = self.dx**2 / max(self.du, self.dv) / 4
        
        # 공간 격자점 생성
        self.x = np.linspace(0, width * self.dx, width)
        self.t = np.arange(0, self.T, self.dt)
        
        # 2D 격자 초기화 (랜덤 초기 상태)
        self.u = np.random.rand(height, width) * 2  # 피식자
        self.v = np.random.rand(height, width) * 2  # 포식자
        
        # 결과 저장용
        self.time = 0
        self.history = {"time": [], "u": [], "v": []}
        
    def step(self):
        """반응-확산 방정식을 한 단계 진행"""
        # Laplacian 계산 (확산항)
        laplace_u = laplace(self.u, mode='mirror')
        laplace_v = laplace(self.v, mode='mirror')
        
        # 반응-확산 방정식 적용
        nu = np.maximum(0, self.u + self.dt * (self.du * laplace_u + 
                                             self.u - self.u**2/self.K - 
                                             self.v * self.u - self.cu))
        
        nv = np.maximum(0, self.v + self.dt * (self.dv * laplace_v + 
                                             self.v * self.u - 
                                             self.v - self.cv))
        
        # Neumann 경계 조건 적용
        nu[0, :] = nu[2, :]; nu[-1, :] = nu[-3, :]
        nu[:, 0] = nu[:, 2]; nu[:, -1] = nu[:, -3]
        nv[0, :] = nv[2, :]; nv[-1, :] = nv[-3, :]
        nv[:, 0] = nv[:, 2]; nv[:, -1] = nv[:, -3]
        
        # 상태 업데이트
        self.u = nu
        self.v = nv
        
        # 시간 업데이트
        self.time += self.dt
        
        # 주기적으로 결과 저장
        if len(self.history["time"]) == 0 or self.time - self.history["time"][-1] >= 0.5:
            self.history["time"].append(self.time)
            self.history["u"].append(self.u.copy())
            self.history["v"].append(self.v.copy())
        
    def get_state(self):
        """현재 상태 반환"""
        return {
            "time": self.time,
            "u": self.u,
            "v": self.v
        }
        
    def get_stats(self):
        """시스템 통계 계산"""
        return {
            "time": self.time,
            "total_prey": np.sum(self.u),
            "total_predator": np.sum(self.v),
            "max_prey": np.max(self.u),
            "max_predator": np.max(self.v)
        }

class RandomWalkSimulation:
    """
    - 2D 평면 상에서 개체(Agent)들이 무작위 이동하며 상호작용하는 시뮬레이션 클래스
    - 포식, 번식, 사망 등의 이벤트를 확률적으로 처리
    - 번식 지연 기능 추가 (알 상태에서 부화까지 시간 지연)
    """
    def __init__(self, width, height, config):
        """
        width, height: 시뮬레이션 공간 크기
        config: 파라미터 설정 (딕셔너리)
            - prob_eat: 포식 확률
            - prob_reproduce_predator: 포식자 번식 확률
            - prob_reproduce_prey: 피식자 번식 확률
            - prob_death: 자연사 확률
            - delay_predator: 포식자 부화 지연 시간 (단계 수)
            - delay_prey: 피식자 부화 지연 시간 (단계 수)
        """
        self.width = width
        self.height = height
        self.config = config
        self.agents = []
        self.eggs = []  # 부화를 기다리는 알(egg) 리스트
        self.time = 0

    def add_agent(self, agent):
        self.agents.append(agent)

    def step(self):
        """
        1 스텝 동안
        1) 무작위 이동
        2) 포식/번식/사망 이벤트 처리
        3) 알 부화 처리
        """
        # 시간 증가
        self.time += 1
        
        # 모든 에이전트의 나이 증가
        for agent in self.agents:
            agent.age += 1

        # 1) 무작위 이동
        self._random_walk_all()

        # 2) 상호작용 이벤트
        self._interactions()

        # 3) 알 부화 처리 (지연 번식)
        self._process_eggs()

        # 스텝이 끝난 뒤, 실제로 사망 처리된 개체 제거
        self.agents = [a for a in self.agents if a.alive]

    def _random_walk_all(self):
        """ 모든 개체를 무작위로 한 칸(또는 한 단위) 이동 """
        for agent in self.agents:
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            # 위치 업데이트 (경계를 넘어가면 wrap-around 또는 반사 등 처리 가능)
            agent.x = (agent.x + dx) % self.width
            agent.y = (agent.y + dy) % self.height

    def _interactions(self):
        """
        동일 위치(또는 근방)에 있는 개체들 간 포식/번식/사망 처리
        여기서는 간단히 같은 정수 좌표에 있는 경우를 충돌로 간주
        """
        # 위치별로 개체를 그룹화
        position_dict = {}
        for agent in self.agents:
            pos = (agent.x, agent.y)
            if pos not in position_dict:
                position_dict[pos] = []
            position_dict[pos].append(agent)

        # 각 위치에 대해 포식자(Predator)와 피식자(Prey)가 함께 있으면 상호작용
        for pos, agents_in_cell in position_dict.items():
            if len(agents_in_cell) < 2:
                continue

            predators = [a for a in agents_in_cell if a.species == "Predator" and a.alive]
            preys = [a for a in agents_in_cell if a.species == "Prey" and a.alive]

            if predators and preys:
                # 포식 확률로 사냥, 번식 확률 등 처리
                for predator in predators:
                    for prey in preys:
                        if random.random() < self.config["prob_eat"]:
                            prey.alive = False  # 피식자가 잡아먹힘
                            # 포식자가 번식할 확률
                            if random.random() < self.config["prob_reproduce_predator"]:
                                # 지연 번식 - 포식자 알 생성
                                delay = self.config.get("delay_predator", 0)
                                if delay > 0:
                                    self._add_egg("Predator", predator.x, predator.y, delay)
                                else:
                                    self._spawn_predator(predator.x, predator.y)

        # 모든 개체에 대해 자연사 확률 적용 (포식자/피식자 공통)
        for agent in self.agents:
            if agent.alive:
                # 피식자(Prey) 번식 확률 
                if agent.species == "Prey":
                    # 번식 충분히 나이를 먹었을 때만 번식 가능 (선택적)
                    min_reproduction_age = self.config.get("min_reproduction_age_prey", 0)
                    if agent.age >= min_reproduction_age:
                        if random.random() < self.config["prob_reproduce_prey"]:
                            # 지연 번식 - 피식자 알 생성
                            delay = self.config.get("delay_prey", 0)
                            if delay > 0:
                                self._add_egg("Prey", agent.x, agent.y, delay)
                            else:
                                self._spawn_prey(agent.x, agent.y)

                # 자연사 처리
                if random.random() < self.config["prob_death"]:
                    agent.alive = False
                
                # 노화에 의한 사망 (선택적)
                max_age = self.config.get("max_age", 1000)  # 기본값은 아주 큰 값으로
                if agent.age > max_age:
                    agent.alive = False

    def _process_eggs(self):
        """알 부화 처리 (지연 번식)"""
        hatched_eggs = []
        remaining_eggs = []
        
        for egg in self.eggs:
            egg["timer"] -= 1
            if egg["timer"] <= 0:
                # 부화 시간이 되면 새 에이전트 생성
                if egg["species"] == "Predator":
                    self._spawn_predator(egg["x"], egg["y"])
                else:
                    self._spawn_prey(egg["x"], egg["y"])
                hatched_eggs.append(egg)
            else:
                remaining_eggs.append(egg)
                
        # 부화한 알은 제거
        self.eggs = remaining_eggs

    def _add_egg(self, species, x, y, delay):
        """번식 지연을 위한 알 추가"""
        self.eggs.append({
            "species": species,
            "x": x,
            "y": y,
            "timer": delay
        })

    def _spawn_predator(self, x, y):
        """ 포식자 생성 """
        new_agent = Agent("Predator", x, y)
        self.agents.append(new_agent)

    def _spawn_prey(self, x, y):
        """ 피식자 생성 """
        new_agent = Agent("Prey", x, y)
        self.agents.append(new_agent)
        
    def get_stats(self):
        """시스템 통계 계산"""
        num_prey = sum(a.species == "Prey" and a.alive for a in self.agents)
        num_predator = sum(a.species == "Predator" and a.alive for a in self.agents)
        num_prey_eggs = sum(egg["species"] == "Prey" for egg in self.eggs)
        num_predator_eggs = sum(egg["species"] == "Predator" for egg in self.eggs)
        
        return {
            "time": self.time,
            "num_prey": num_prey,
            "num_predator": num_predator,
            "num_prey_eggs": num_prey_eggs,
            "num_predator_eggs": num_predator_eggs,
            "total_prey": num_prey + num_prey_eggs,
            "total_predator": num_predator + num_predator_eggs
        } 