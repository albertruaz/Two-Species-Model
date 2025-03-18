import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.animation import ArtistAnimation

class Environment:
    """시뮬레이션 환경을 관리하는 클래스"""
    def __init__(self, model_type, width, height, config):
        """
        Parameters:
        model_type: "reaction_diffusion" 또는 "random_walk"
        width, height: 시뮬레이션 공간 크기
        config: 모델 설정
        """
        self.model_type = model_type
        self.width = width
        self.height = height
        self.config = config
        
        # 모델 초기화
        if model_type == "reaction_diffusion":
            from model import ReactionDiffusionModel
            self.model = ReactionDiffusionModel(width, height, config)
        else:  # random_walk
            from model import RandomWalkSimulation
            self.model = RandomWalkSimulation(width, height, config)
            
        # 시각화 설정
        self.setup_visualization()
        
    def setup_visualization(self):
        """시각화 설정"""
        plt.ion()
        
        if self.model_type == "reaction_diffusion":
            self.fig, self.axs = plt.subplots(1, 3, figsize=(15, 5))
            
            # 컬러맵 설정
            self.prey_img = self.axs[0].imshow(self.model.u, cmap=plt.cm.Greens, 
                                              vmin=0, vmax=self.model.K,
                                              extent=[0, self.width * self.model.dx, 
                                                     0, self.height * self.model.dx])
            self.predator_img = self.axs[1].imshow(self.model.v, cmap=plt.cm.Reds, 
                                                  vmin=0, vmax=3,
                                                  extent=[0, self.width * self.model.dx, 
                                                         0, self.height * self.model.dx])
            
            # 제목 및 레이블 설정
            self.axs[0].set_title("Prey Density")
            self.axs[1].set_title("Predator Density")
            self.axs[2].set_title("Population over Time")
            
            for ax in self.axs[:2]:
                ax.set_xlabel("Space")
                ax.set_ylabel("Space")
            
            self.fig.colorbar(self.prey_img, ax=self.axs[0], label="Density")
            self.fig.colorbar(self.predator_img, ax=self.axs[1], label="Density")
            
        else:  # random_walk
            self.fig, (self.ax, self.ax_counts) = plt.subplots(1, 2, figsize=(14, 6))
            self.ax.set_title("Simulation Space")
            self.ax_counts.set_title("Population over Time")
            self.ax_counts.set_xlabel("Time")
            self.ax_counts.set_ylabel("Count")
            
        # 데이터 추적용
        self.time_points = []
        self.prey_data = []
        self.predator_data = []
        
    def step(self):
        """한 단계 시뮬레이션 진행"""
        self.model.step()
        stats = self.model.get_stats()
        
        # 데이터 추적
        if self.model_type == "reaction_diffusion":
            self.time_points.append(stats["time"])
            self.prey_data.append(stats["total_prey"])
            self.predator_data.append(stats["total_predator"])
        else:
            self.time_points.append(stats["time"])
            self.prey_data.append(stats["total_prey"])
            self.predator_data.append(stats["total_predator"])
            
        return stats
    
    def render(self):
        """현재 상태 시각화"""
        if self.model_type == "reaction_diffusion":
            # 밀도 맵 업데이트
            self.prey_img.set_array(self.model.u)
            self.predator_img.set_array(self.model.v)
            
            # 시간 표시 업데이트
            self.fig.suptitle(f"Time: {self.model.time:.2f} / {self.model.T}")
            
            # 개체수 그래프 업데이트
            self.axs[2].clear()
            self.axs[2].set_title("Population over Time")
            self.axs[2].plot(self.time_points, self.prey_data, 'g-', label="Prey")
            self.axs[2].plot(self.time_points, self.predator_data, 'r-', label="Predator")
            self.axs[2].set_xlabel("Time")
            self.axs[2].set_ylabel("Total Population")
            self.axs[2].legend()
            
        else:  # random_walk
            self.ax.clear()
            self.ax.set_title(f"Step {len(self.time_points)}")
            self.ax.set_xlim(0, self.width)
            self.ax.set_ylim(0, self.height)
            
            # 개체 위치 산점도
            prey_x = [a.x for a in self.model.agents if a.species == "Prey" and a.alive]
            prey_y = [a.y for a in self.model.agents if a.species == "Prey" and a.alive]
            predator_x = [a.x for a in self.model.agents if a.species == "Predator" and a.alive]
            predator_y = [a.y for a in self.model.agents if a.species == "Predator" and a.alive]
            
            # 알(egg) 위치 표시
            egg_x = [egg["x"] for egg in self.model.eggs]
            egg_y = [egg["y"] for egg in self.model.eggs]
            egg_colors = ["green" if egg["species"] == "Prey" else "red" for egg in self.model.eggs]
            
            self.ax.scatter(prey_x, prey_y, color='green', label="Prey")
            self.ax.scatter(predator_x, predator_y, color='red', label="Predator")
            self.ax.scatter(egg_x, egg_y, color=egg_colors, marker='x', alpha=0.5, label="Eggs")
            self.ax.legend()
            
            # 개체수 그래프 업데이트
            self.ax_counts.clear()
            self.ax_counts.set_title("Population over Time")
            self.ax_counts.plot(self.time_points, self.prey_data, 'g-', label="Prey")
            self.ax_counts.plot(self.time_points, self.predator_data, 'r-', label="Predator")
            self.ax_counts.set_xlabel("Time")
            self.ax_counts.set_ylabel("Count")
            self.ax_counts.legend()
        
        plt.tight_layout()
        plt.pause(0.01)
    
    def run(self, total_steps, plot_interval=1, project_name="PredPreyProject", 
            record_video=False, video_filename="simulation.mp4"):
        """시뮬레이션 실행
        
        Parameters:
        total_steps: 총 시뮬레이션 단계 수
        plot_interval: 몇 단계마다 시각화할지
        project_name: WandB 프로젝트 이름
        record_video: 비디오 저장 여부
        video_filename: 저장할 비디오 파일명
        """
        # WandB 초기화
        wandb.init(project=project_name, entity="mathematical-modeling")
        wandb.config.update(self.config)
        
        # 비디오 프레임 저장용
        if record_video:
            frames = []
        
        # 메인 시뮬레이션 루프
        for step in range(total_steps):
            # 시뮬레이션 진행
            stats = self.step()
            
            # WandB 로깅
            wandb.log({"step": step, **stats})
            
            # 시각화
            if step % plot_interval == 0:
                self.render()
                
                if record_video:
                    frames.append([plt.gcf()])
        
        # 비디오 저장
        if record_video and frames:
            print(f"비디오 저장 중: {video_filename}")
            ani = ArtistAnimation(self.fig, frames, interval=200, blit=True)
            ani.save(video_filename, writer='ffmpeg')
            print(f"비디오 저장 완료: {video_filename}")
        
        plt.ioff()
        plt.show()
        wandb.finish() 