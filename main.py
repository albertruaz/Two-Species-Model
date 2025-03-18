from model import Agent, RandomWalkSimulation, ReactionDiffusionModel
from simulation import run_simulation, run_reaction_diffusion_comparison, run_delay_experiment
import numpy as np
import random
import argparse
import os

def main():
    # 커맨드 라인 인수 파싱
    parser = argparse.ArgumentParser(description='Two-Species Predator-Prey Model Simulation')
    parser.add_argument('--model', type=str, default='reaction_diffusion', 
                      choices=['reaction_diffusion', 'random_walk'],
                      help='시뮬레이션 모델 선택 (reaction_diffusion 또는 random_walk)')
    parser.add_argument('--experiment', type=str, default=None,
                      choices=[None, 'delay', 'comparison'],
                      help='실험 종류 (delay: 지연 실험, comparison: 파라미터 비교)')
    parser.add_argument('--steps', type=int, default=200, help='시뮬레이션 스텝 수')
    parser.add_argument('--video', action='store_true', help='비디오 저장 여부')
    parser.add_argument('--width', type=int, default=50, help='시뮬레이션 폭')
    parser.add_argument('--height', type=int, default=50, help='시뮬레이션 높이')
    
    args = parser.parse_args()
    
    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    
    if args.model == 'reaction_diffusion':
        run_reaction_diffusion_simulation(args)
    else:
        run_individual_based_simulation(args)

def run_reaction_diffusion_simulation(args):
    """반응-확산 모델 시뮬레이션 실행"""
    width = args.width
    height = args.height
    
    # 기본 설정값
    config = {
        "du": 0.25,         # 피식자 확산 계수
        "dv": 0.05,         # 포식자 확산 계수
        "K": 8,             # 피식자의 carrying capacity
        "cu": 0.2,          # 피식자 수확 계수
        "cv": 0.05,         # 포식자 수확 계수
        "dx": 0.2           # 공간 간격
    }
    
    if args.experiment == 'comparison':
        # 여러 파라미터 설정으로 비교 실행
        configs = [
            # 기본 설정
            config.copy(),
            
            # 확산계수가 다른 설정
            {**config, "du": 0.5, "dv": 0.1},
            
            # Carrying capacity가 다른 설정
            {**config, "K": 4},
            
            # 수확 계수가 다른 설정
            {**config, "cu": 0.1, "cv": 0.02}
        ]
        
        run_reaction_diffusion_comparison(width, height, configs, args.steps, 
                                          plot_interval=5, 
                                          project_name="ReactionDiffusionComparison")
    else:
        # 단일 시뮬레이션 실행
        sim = ReactionDiffusionModel(width, height, config)
        
        video_filename = "reaction_diffusion.mp4" if args.video else None
        
        run_simulation(sim, args.steps, plot_interval=5, 
                       project_name="ReactionDiffusion", 
                       record_video=args.video,
                       video_filename=video_filename)

def run_individual_based_simulation(args):
    """개체 기반 모델 시뮬레이션 실행"""
    width = args.width
    height = args.height
    
    # 기본 설정값
    config = {
        "prob_eat": 0.3,                # 포식 확률
        "prob_reproduce_predator": 0.1, # 포식자 번식 확률
        "prob_reproduce_prey": 0.01,    # 피식자 번식 확률
        "prob_death": 0.002,            # 자연사 확률
        "delay_prey": 0,                # 피식자 번식 지연 (기본값: 0)
        "delay_predator": 0,            # 포식자 번식 지연 (기본값: 0)
        "max_age": 100,                 # 최대 수명
        "min_reproduction_age_prey": 5, # 피식자 최소 번식 나이
        "num_init_prey": 100,           # 초기 피식자 수
        "num_init_predator": 20         # 초기 포식자 수
    }
    
    if args.experiment == 'delay':
        # 지연 효과 실험
        delay_values = [
            (0, 0),      # 지연 없음
            (5, 0),      # 피식자만 지연
            (0, 5),      # 포식자만 지연
            (5, 5),      # 둘 다 같은 지연
            (10, 5),     # 피식자가 더 긴 지연
            (5, 10)      # 포식자가 더 긴 지연
        ]
        
        run_delay_experiment(width, height, config, delay_values, args.steps, 
                             plot_interval=5, project_name="DelayExperiment")
    else:
        # 단일 시뮬레이션 실행
        # 시뮬레이션 초기화
        sim = RandomWalkSimulation(width, height, config)
        
        # 초기 개체수 설정 (랜덤 위치 배치)
        for _ in range(config["num_init_prey"]):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            sim.add_agent(Agent("Prey", x, y))
        
        for _ in range(config["num_init_predator"]):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            sim.add_agent(Agent("Predator", x, y))
        
        video_filename = "random_walk.mp4" if args.video else None
        
        # 시뮬레이션 실행
        run_simulation(sim, args.steps, plot_interval=5, 
                       project_name="RandomWalk", 
                       record_video=args.video,
                       video_filename=video_filename)

if __name__ == "__main__":
    main() 