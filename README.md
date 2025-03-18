# Two-Species Predator-Prey Model

이 프로젝트는 포식자(Predator)와 피식자(Prey)의 상호작용을 시뮬레이션하는 두 가지 모델을 구현합니다:

1. **반응-확산 모델(Reaction-Diffusion Model)**: 미분 방정식 기반의 연속적인 모델
2. **개체 기반 모델(Individual-Based Model)**: 무작위 이동과 상호작용을 하는 이산적 개체 모델

두 모델 모두 번식 지연(reproduction delay) 메커니즘을 포함하고 있습니다.

## 기능

- **반응-확산 모델**:

  - 2D 공간에서의 포식자-피식자 밀도 시뮬레이션
  - Laplacian 확산과 상호작용 항을 포함
  - 초기 랜덤 밀도 분포에서 시작하여 패턴 형성 관찰

- **개체 기반 모델**:

  - 개체들의 무작위 이동(Random Walk)
  - 포식, 번식, 자연사 처리
  - 지연 번식 메커니즘 (알을 낳고 부화까지 시간이 걸리는 과정)

- **공통 기능**:
  - 실시간 시각화
  - WandB를 통한 실험 로깅
  - 비디오 저장 기능
  - 다양한 설정 비교 실험

## 설치 방법

필요한 패키지 설치:

```bash
pip install numpy matplotlib scipy wandb
```

비디오 저장 기능을 사용하려면 ffmpeg가 필요합니다:

```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt-get install ffmpeg
```

## 사용 방법

### 기본 시뮬레이션 실행

```bash
# 반응-확산 모델(기본값)
python main.py

# 개체 기반 모델
python main.py --model random_walk
```

### 명령줄 옵션

```bash
python main.py --help
```

주요 옵션:

- `--model`: 시뮬레이션 모델 선택 (`reaction_diffusion` 또는 `random_walk`)
- `--experiment`: 실험 종류 (`delay`: 지연 실험, `comparison`: 파라미터 비교)
- `--steps`: 시뮬레이션 스텝 수
- `--video`: 비디오 저장 여부
- `--width`, `--height`: 시뮬레이션 공간 크기

### 실험 예시

1. **지연 효과 실험**:

```bash
python main.py --model random_walk --experiment delay
```

2. **파라미터 비교 실험**:

```bash
python main.py --experiment comparison
```

3. **비디오 저장**:

```bash
python main.py --video
```

## 모델 커스터마이징

### 반응-확산 모델 파라미터

`main.py`의 `run_reaction_diffusion_simulation` 함수에서 설정:

- `du`: 피식자 확산 계수
- `dv`: 포식자 확산 계수
- `K`: 피식자의 carrying capacity
- `cu`: 피식자 수확 계수
- `cv`: 포식자 수확 계수

### 개체 기반 모델 파라미터

`main.py`의 `run_individual_based_simulation` 함수에서 설정:

- `prob_eat`: 포식 확률
- `prob_reproduce_predator`: 포식자 번식 확률
- `prob_reproduce_prey`: 피식자 번식 확률
- `delay_prey`, `delay_predator`: 번식 지연 단계 수
- `max_age`: 개체 최대 수명

## 파일 구조

- `model.py`: 모델 클래스 (Agent, RandomWalkSimulation, ReactionDiffusionModel) 정의
- `simulation.py`: 시뮬레이션 실행, 시각화, 로깅 담당 함수
- `main.py`: 설정값과 실행 흐름을 관리
