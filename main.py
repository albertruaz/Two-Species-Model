import numpy as np
from environment import Environment

def main():
    # 시뮬레이션 설정값
    config = {
        # 공간 파라미터
        'L': 50,
        'boundary_condition': 'periodic',  # 'periodic', 'neumann', 'dirichlet'
        
        # 이동 관련 설정
        'movement_pattern': '8dir',  # '4dir', '8dir', '8dir_weighted'
        'prey_move_percent': 0.888,
        'predator_move_percent': 0.888,
        
        # 초기 분포 설정
        'prey_location_rate': 0.2,     # 전체 공간 중 10%에 분포
        'prey_density': 10,            # 각 위치당 10배의 개체수
        'predator_location_rate': 0.2, # 전체 공간 중 10%에 분포
        'predator_density': 10,        # 각 위치당 10배의 개체수
        
        # Lotka-Volterra 파라미터
        'K': 50,  # 환경수용력
        'beta': 2,  # 포식률
        
        # 시뮬레이션 기본 설정
        'total_steps': 200,   # 총 시뮬레이션 스텝 수
        'prey_speed': 1.0,      # 피식자 이동 속도
        'predator_speed': 1.0,  # 포식자 이동 속도
        
        # 시각화 관련 설정
        'n_grid': 10,
        'plot_interval': 1,    # 그래프 갱신 간격 (단위: 스텝)
        'figsize': (15, 5),    # 그래프 크기
        
        # 색상 설정
        'prey_color': 'YlGn',       # 피식자 컬러맵
        'predator_color': 'OrRd',   # 포식자 컬러맵
        'grid_color': 'gray',       # 격자 선 색상
        'grid_alpha': 0.2,          # 격자 선 투명도
        
        # 그래프 스타일 설정
        'plot_style': 'dark_background',  # 그래프 스타일
        'line_width': 2,                  # 선 굵기
        'grid_alpha_pop': 0.3,            # 개체수 그래프 격자 투명도
        'pause_interval': 0.01,           # 그래프 갱신 간격 (초)
        
        # 비디오 설정
        'record_video': False,  # 비디오 저장 여부
        'video_fps': 15,        # 비디오 프레임률

        # WandB 설정
        'use_wandb': False,     # WandB 사용 여부
        'wandb_project': "predator-prey-simulation",  # WandB 프로젝트 이름
        'wandb_entity': None    # WandB 엔티티 (None이면 기본값 사용)
    }

    # 환경 생성 및 시뮬레이션 실행
    env = Environment(config)
    env.run(
        total_steps=config['total_steps'],
        plot_interval=config['plot_interval'],
        record_video=config['record_video']
    )

if __name__ == "__main__":
    main() 