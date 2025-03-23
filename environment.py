import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
import wandb
from model import GridBasedModel

class Environment:
    """시뮬레이션 환경을 관리하는 클래스"""
    def __init__(self, config):
        """
        Parameters:
        config: 모델 설정 딕셔너리
        """
        # 난수 시드 설정
        seed = config.get('random_seed', 42)
        np.random.seed(seed)
        
        self.config = config
        self.model = GridBasedModel(config)
        
        # 시각화 설정
        plt.style.use(config['plot_style'])
        self.fig = plt.figure(figsize=(20, 5))
        # 왼쪽부터 순서대로: 밀도 맵, 전체 개체수, 1x1 영역, 5x5 영역, 10x10 영역
        self.ax1 = plt.subplot(151, aspect='equal')  # 밀도 맵
        self.ax2 = plt.subplot(152)  # 전체 개체수
        self.ax3 = plt.subplot(153)  # 1x1 영역
        self.ax4 = plt.subplot(154)  # 5x5 영역
        self.ax5 = plt.subplot(155)  # 10x10 영역
        
        # 시간에 따른 개체수 데이터
        self.times = []
        # 전체 공간의 개체수
        self.total_preys = []
        self.total_predators = []
        # 1x1 영역의 개체수
        self.local_preys_1 = []
        self.local_predators_1 = []
        # 5x5 영역의 개체수
        self.local_preys_5 = []
        self.local_predators_5 = []
        # 10x10 영역의 개체수
        self.local_preys_10 = []
        self.local_predators_10 = []
        
        # 초기 데이터 기록
        stats = self.model.get_stats()
        self.times.append(stats['time'])
        self.total_preys.append(stats['total_prey'])
        self.total_predators.append(stats['total_predator'])
        
        # 지역 개체수 초기화
        center = self.model.n_grid // 2
        self._update_local_populations(center)
        
        # WandB 초기화
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config['wandb_project'],
                entity=self.config['wandb_entity'],
                config=config
            )
    
    def _update_local_populations(self, center):
        """중앙 주변 영역의 개체수 업데이트"""
        # 1x1 영역
        self.local_preys_1.append(np.sum(self.model.prey_density[center:center+1, center:center+1]))
        self.local_predators_1.append(np.sum(self.model.predator_density[center:center+1, center:center+1]))
        
        # 5x5 영역
        start_5 = center - 2
        end_5 = center + 3
        self.local_preys_5.append(np.sum(self.model.prey_density[start_5:end_5, start_5:end_5]))
        self.local_predators_5.append(np.sum(self.model.predator_density[start_5:end_5, start_5:end_5]))
        
        # 10x10 영역
        start_10 = center - 5
        end_10 = center + 5
        self.local_preys_10.append(np.sum(self.model.prey_density[start_10:end_10, start_10:end_10]))
        self.local_predators_10.append(np.sum(self.model.predator_density[start_10:end_10, start_10:end_10]))
    
    def step(self):
        """시뮬레이션 한 단계 진행"""
        stats = self.model.step()
        
        # 중앙 영역 개체수 업데이트
        center = self.model.n_grid // 2
        self._update_local_populations(center)
        
        # WandB에 로깅
        if self.config.get('use_wandb', False):
            wandb.log({
                'total_prey': stats['total_prey'],
                'total_predator': stats['total_predator'],
                'time': stats['time']
            })
        
        return stats
    
    def render(self):
        """현재 상태 시각화"""
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.ax4.clear()
        self.ax5.clear()
        
        # 시각화를 위한 격자선 (n_grid 사용)
        if 'n_grid' in self.config:
            x_ticks = np.linspace(0, self.config['L'], self.config['n_grid'])
            y_ticks = np.linspace(0, self.config['L'], self.config['n_grid'])
            self.ax1.set_xticks(x_ticks, minor=True)
            self.ax1.set_yticks(y_ticks, minor=True)
            self.ax1.grid(True, which='minor', color=self.config['grid_color'], alpha=self.config['grid_alpha'])
        
        # 피식자와 포식자 밀도를 하나의 그래프에 표시
        prey_max = max(1, np.max(self.model.prey_density))
        pred_max = max(1, np.max(self.model.predator_density))
        
        # 피식자 밀도 컨투어 (녹색)
        if prey_max > 0:
            self.ax1.contour(self.model.prey_density,
                          levels=np.linspace(0, prey_max, 10),
                          extent=[0, self.config['L'], 0, self.config['L']],
                          cmap='YlGn',
                          alpha=0.7)
        
        # 포식자 밀도 컨투어 (빨간색)
        if pred_max > 0:
            self.ax1.contour(self.model.predator_density,
                          levels=np.linspace(0, pred_max, 10),
                          extent=[0, self.config['L'], 0, self.config['L']],
                          cmap='OrRd',
                          alpha=0.7)
        
        # 중앙 영역 표시
        center = self.config['L'] / 2
        # 1x1 영역
        self.ax1.plot([center-0.5, center+0.5, center+0.5, center-0.5, center-0.5],
                     [center-0.5, center-0.5, center+0.5, center+0.5, center-0.5],
                     'white', linestyle='--', alpha=0.5)
        # 5x5 영역
        self.ax1.plot([center-2.5, center+2.5, center+2.5, center-2.5, center-2.5],
                     [center-2.5, center-2.5, center+2.5, center+2.5, center-2.5],
                     'white', linestyle='--', alpha=0.5)
        # 10x10 영역
        self.ax1.plot([center-5, center+5, center+5, center-5, center-5],
                     [center-5, center-5, center+5, center+5, center-5],
                     'white', linestyle='--', alpha=0.5)
        
        self.ax1.set_title('Population Density')
        self.ax1.set_xlabel('X')
        self.ax1.set_ylabel('Y')
        
        # 전체 개체수 변화 그래프
        if len(self.total_preys) > 0:
            self.ax2.plot(self.times, self.total_preys, 
                        color='green', label='Prey', linewidth=self.config['line_width'])
            self.ax2.plot(self.times, self.total_predators,
                        color='red', label='Predator', linewidth=self.config['line_width'])
            max_pop = max(max(self.total_preys), max(self.total_predators))
            self.ax2.set_ylim([0, max(max_pop * 1.1, 0.1)])
        
        self.ax2.set_title('Total Population')
        self.ax2.set_xlabel('Time')
        self.ax2.set_ylabel('Population')
        self.ax2.legend()
        self.ax2.grid(True, alpha=self.config['grid_alpha_pop'])
        
        # 1x1 영역 개체수 변화 그래프
        if len(self.local_preys_1) > 0:
            self.ax3.plot(self.times, self.local_preys_1, 
                       color='green', label='Prey', linewidth=self.config['line_width'])
            self.ax3.plot(self.times, self.local_predators_1,
                       color='red', label='Predator', linewidth=self.config['line_width'])
            max_pop = max(max(self.local_preys_1), max(self.local_predators_1))
            self.ax3.set_ylim([0, max(max_pop * 1.1, 0.1)])
        
        self.ax3.set_title('1x1 Region')
        self.ax3.set_xlabel('Time')
        self.ax3.set_ylabel('Population')
        self.ax3.legend()
        self.ax3.grid(True, alpha=self.config['grid_alpha_pop'])
        
        # 5x5 영역 개체수 변화 그래프
        if len(self.local_preys_5) > 0:
            self.ax4.plot(self.times, self.local_preys_5, 
                       color='green', label='Prey', linewidth=self.config['line_width'])
            self.ax4.plot(self.times, self.local_predators_5,
                       color='red', label='Predator', linewidth=self.config['line_width'])
            max_pop = max(max(self.local_preys_5), max(self.local_predators_5))
            self.ax4.set_ylim([0, max(max_pop * 1.1, 0.1)])
        
        self.ax4.set_title('5x5 Region')
        self.ax4.set_xlabel('Time')
        self.ax4.set_ylabel('Population')
        self.ax4.legend()
        self.ax4.grid(True, alpha=self.config['grid_alpha_pop'])
        
        # 10x10 영역 개체수 변화 그래프
        if len(self.local_preys_10) > 0:
            self.ax5.plot(self.times, self.local_preys_10, 
                       color='green', label='Prey', linewidth=self.config['line_width'])
            self.ax5.plot(self.times, self.local_predators_10,
                       color='red', label='Predator', linewidth=self.config['line_width'])
            max_pop = max(max(self.local_preys_10), max(self.local_predators_10))
            self.ax5.set_ylim([0, max(max_pop * 1.1, 0.1)])
        
        self.ax5.set_title('10x10 Region')
        self.ax5.set_xlabel('Time')
        self.ax5.set_ylabel('Population')
        self.ax5.legend()
        self.ax5.grid(True, alpha=self.config['grid_alpha_pop'])
        
        plt.tight_layout()
        if not hasattr(self, '_recording'):
            plt.pause(self.config['pause_interval'])
    
    def run(self, total_steps, plot_interval=1, record_video=False):
        """시뮬레이션 실행"""
        if record_video:
            self._recording = True  # 녹화 모드 표시
            import tempfile
            import os
            import cv2
            # 임시 디렉토리 생성
            temp_dir = tempfile.mkdtemp()
            frame_files = []
        
        for step in range(total_steps):
            # 한 스텝 진행
            stats = self.step()
            
            # 데이터 기록
            self.times.append(stats['time'])
            self.total_preys.append(stats['total_prey'])
            self.total_predators.append(stats['total_predator'])
            
            # 시각화
            if step % plot_interval == 0:
                self.render()
                if record_video:
                    # 현재 프레임을 PNG 파일로 저장
                    frame_path = os.path.join(temp_dir, f'frame_{step:05d}.png')
                    self.fig.canvas.draw()
                    self.fig.savefig(frame_path, dpi=100, bbox_inches='tight', pad_inches=0.1)
                    frame_files.append(frame_path)
                    print(f"Frame {step}/{total_steps} saved")  # 진행상황 출력
        
        if record_video and frame_files:
            print(f"생성된 프레임 수: {len(frame_files)}")
            # 첫 프레임을 읽어서 비디오 크기 결정
            first_frame = cv2.imread(frame_files[0])
            if first_frame is None:
                print(f"Error: Cannot read frame {frame_files[0]}")
                return
                
            height, width = first_frame.shape[:2]
            print(f"프레임 크기: {width}x{height}")
            
            # 비디오 저장
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = min(30, self.config['video_fps'])
            video = cv2.VideoWriter('simulation.mp4', fourcc, fps, (width, height))
            
            if not video.isOpened():
                print("Error: VideoWriter failed to open")
                return
            
            # 모든 프레임을 비디오에 추가
            for i, frame_path in enumerate(frame_files):
                frame = cv2.imread(frame_path)
                if frame is not None:
                    video.write(frame)
                    print(f"Writing frame {i+1}/{len(frame_files)}")
                else:
                    print(f"Error: Cannot read frame {frame_path}")
                # 임시 파일 삭제
                os.remove(frame_path)
            
            video.release()
            # 임시 디렉토리 삭제
            os.rmdir(temp_dir)
            print("비디오 생성 완료")
            
            # 녹화 모드 해제
            delattr(self, '_recording')
        
        # 마지막 상태 표시
        self.render()
        plt.show() 