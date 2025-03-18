import argparse
from environment import Environment

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='Predator-Prey Simulation')
    parser.add_argument('--model', type=str, default='reaction_diffusion',
                      choices=['reaction_diffusion', 'random_walk'],
                      help='시뮬레이션 모델 선택')
    parser.add_argument('--width', type=int, default=100, help='시뮬레이션 공간 너비')
    parser.add_argument('--height', type=int, default=100, help='시뮬레이션 공간 높이')
    parser.add_argument('--steps', type=int, default=1000, help='시뮬레이션 단계 수')
    parser.add_argument('--plot-interval', type=int, default=1, help='시각화 간격')
    parser.add_argument('--video', action='store_true', help='비디오 저장 여부')
    args = parser.parse_args()
    
    # 모델별 기본 설정
    if args.model == 'reaction_diffusion':
        config = {
            "dx": 0.2,          # 공간 간격
            "du": 0.25,         # 피식자 확산 계수
            "dv": 0.05,         # 포식자 확산 계수
            "K": 8,             # 피식자의 carrying capacity
            "cu": 0.2,          # 피식자 수확 계수
            "cv": 0.05,         # 포식자 수확 계수
            "T": 150            # 총 시뮬레이션 시간
        }
    else:  # random_walk
        config = {
            "prob_eat": 0.5,            # 포식 확률
            "prob_reproduce_predator": 0.3,  # 포식자 번식 확률
            "prob_reproduce_prey": 0.3,      # 피식자 번식 확률
            "prob_death": 0.1,          # 자연사 확률
            "delay_predator": 5,        # 포식자 부화 지연
            "delay_prey": 3,            # 피식자 부화 지연
            "num_init_prey": 100,       # 초기 피식자 수
            "num_init_predator": 20     # 초기 포식자 수
        }
    
    # 환경 생성 및 시뮬레이션 실행
    env = Environment(args.model, args.width, args.height, config)
    env.run(total_steps=args.steps,
           plot_interval=args.plot_interval,
           record_video=args.video)

if __name__ == "__main__":
    main() 