# Predator-Prey Simulation Model

이 프로젝트는 격자 기반의 포식자-피식자 모델을 구현한 시뮬레이션입니다. Lotka-Volterra 방정식을 기반으로 하며, 공간적 분포와 이동을 고려한 확장된 모델을 제공합니다.

## 주요 특징

1. **Lotka-Volterra 방정식**

   ```math
   \dot{u} = u - \frac{u^2}{K} - \frac{\beta u}{u+v}v
   \quad
   \dot{v} = \frac{\beta u}{u+v}v - v
   ```

   - u: 피식자 밀도
   - v: 포식자 밀도
   - K: 환경수용력
   - β: 포식률

2. **공간 이동 패턴**

   - `4dir`: 상하좌우 4방향 이동
   - `8dir`: 8방향 균등 이동
   - `8dir_weighted`: 대각선 이동에 가중치 적용

3. **경계 조건**
   - `periodic`: 주기적 경계 조건
   - `neumann`: 반사 경계 조건
   - `dirichlet`: 흡수 경계 조건

## 설정 파라미터

```python
config = {
   # 공간 파라미터
   'L': 100,                    # 공간 크기
   'n_grid': 10,               # 시각화용 격자 크기
   'boundary_condition': 'periodic',

   # 이동 설정
   'movement_pattern': '8dir',
   'prey_move_percent': 0.6,    # 피식자 이동 확률
   'predator_move_percent': 0.6, # 포식자 이동 확률

   # 초기 분포 설정
   'prey_location_rate': 0.2,     # 전체 공간 중 10%에 분포
   'prey_density': 10,            # 각 위치당 10배의 개체수
   'predator_location_rate': 0.2, # 전체 공간 중 10%에 분포
   'predator_density': 10,        # 각 위치당 10배의 개체수

   # Lotka-Volterra 파라미터
   'K': 50,                    # 환경수용력
   'beta': 2,                  # 포식률

   # 시뮬레이션 설정
   'total_steps': 100,         # 총 시뮬레이션 스텝
   'plot_interval': 1,         # 그래프 갱신 간격

   # 시각화 설정
   'figsize': (15, 5),
   'prey_color': 'YlGn',
   'predator_color': 'OrRd',

   # 비디오 설정
   'record_video': True,
   'video_fps': 30,
}
```

## 실행 방법

1. 필요한 패키지 설치:

```bash
pip install numpy scipy matplotlib
```

2. 시뮬레이션 실행:

```bash
python main.py
```

## 시각화

1. 공간상의 밀도 분포 (컬러맵)
2. 시간에 따른 전체 개체수 변화
3. 특정 지역의 개체수 변화

## 구현

- RK4(Runge-Kutta 4차) 방법을 사용한 수치해석
- NumPy를 활용한 효율적인 행렬 연산
- 다양한 경계 조건 처리
- Matplotlib을 이용한 실시간 시각화
- 시뮬레이션 결과 비디오 저장 기능

## 설명

- 초기 개체수와 위치 설정이 시뮬레이션 결과에 큰 영향
- 수치적 안정성을 위해 적절한 파라미터 설정 중요
