import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import wandb
import time
import os

def run_simulation(sim, total_steps, plot_interval=1, project_name="PredPreyProject", 
                   record_video=False, video_filename="simulation.mp4"):
    """
    - sim: RandomWalkSimulation 또는 ReactionDiffusionModel 객체
    - total_steps: 시뮬레이션 전체 단계 수
    - plot_interval: 몇 스텝마다 그래프(시각화)를 업데이트할지
    - project_name: WandB 프로젝트 명
    - record_video: 비디오 저장 여부
    - video_filename: 저장할 비디오 파일명
    """
    # 시뮬레이션 종류 확인 (RandomWalkSimulation 또는 ReactionDiffusionModel)
    sim_type = sim.__class__.__name__

    # Weights & Biases 초기화
    wandb.init(project=project_name, entity="mathematical-modeling")
    wandb.config.update(sim.config)

    plt.ion()  # interactive mode on
    
    if sim_type == "RandomWalkSimulation":
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter_prey = None
        scatter_predator = None
        
        # 시간에 따른 개체수 변화 그래프를 위한 데이터
        time_points = []
        prey_counts = []
        predator_counts = []
        
        # 오른쪽에 개체수 변화 그래프를 표시할 subplot 추가
        if total_steps > 10:  # 충분한 스텝 수가 있을 때만
            fig, (ax, ax_counts) = plt.subplots(1, 2, figsize=(14, 6))
            ax_counts.set_title("Population over Time")
            ax_counts.set_xlabel("Time")
            ax_counts.set_ylabel("Count")
            prey_line, = ax_counts.plot([], [], 'g-', label="Prey")
            predator_line, = ax_counts.plot([], [], 'r-', label="Predator")
            egg_line, = ax_counts.plot([], [], 'b--', label="Eggs")
            ax_counts.legend()
        
        # 비디오 저장 설정
        if record_video:
            frames = []  # 비디오 프레임을 저장할 리스트
    
    elif sim_type == "ReactionDiffusionModel":
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        # 컬러맵 설정
        prey_cmap = plt.cm.Greens
        predator_cmap = plt.cm.Reds
        
        # 초기 플롯 생성
        prey_img = axs[0].imshow(sim.u, cmap=prey_cmap, vmin=0, vmax=sim.K,
                                extent=[0, sim.width * sim.dx, 0, sim.height * sim.dx])
        predator_img = axs[1].imshow(sim.v, cmap=predator_cmap, vmin=0, vmax=3,
                                    extent=[0, sim.width * sim.dx, 0, sim.height * sim.dx])
        
        # 제목 설정
        axs[0].set_title("Prey Density")
        axs[1].set_title("Predator Density")
        axs[2].set_title("Population over Time")
        
        # x, y 축 레이블 추가
        for ax in axs[:2]:
            ax.set_xlabel("Space")
            ax.set_ylabel("Space")
        
        # 색상 막대 추가
        fig.colorbar(prey_img, ax=axs[0], label="Density")
        fig.colorbar(predator_img, ax=axs[1], label="Density")
        
        # 인구 변화 그래프를 위한 데이터
        time_points = []
        prey_totals = []
        predator_totals = []
        
        # 시간 변화 그래프 초기화
        axs[2].set_xlabel("Time")
        axs[2].set_ylabel("Population")
        axs[2].set_xlim(0, sim.T)
        axs[2].set_ylim(0, sim.K * 1.1)
        
        # 비디오 저장 설정
        if record_video:
            frames = []  # 비디오 프레임을 저장할 리스트
    
    # 메인 시뮬레이션 루프
    for step in range(total_steps):
        # 시뮬레이션 한 스텝 진행
        sim.step()
        
        # 통계 데이터 수집
        if sim_type == "RandomWalkSimulation":
            stats = sim.get_stats()
            
            # WandB에 로그
            wandb.log({
                "step": step,
                "num_prey": stats["num_prey"],
                "num_predator": stats["num_predator"],
                "num_prey_eggs": stats["num_prey_eggs"],
                "num_predator_eggs": stats["num_predator_eggs"]
            })
            
            # 시간 추적 데이터에 추가
            time_points.append(step)
            prey_counts.append(stats["num_prey"])
            predator_counts.append(stats["num_predator"])
            
            # 실시간 Plot 업데이트
            if step % plot_interval == 0:
                # 기존 산포 삭제 후 다시 그림
                ax.clear()
                ax.set_title(f"Step {step}")
                ax.set_xlim(0, sim.width)
                ax.set_ylim(0, sim.height)
                
                # 개체 위치 산점도 
                prey_x = [a.x for a in sim.agents if a.species == "Prey" and a.alive]
                prey_y = [a.y for a in sim.agents if a.species == "Prey" and a.alive]
                predator_x = [a.x for a in sim.agents if a.species == "Predator" and a.alive]
                predator_y = [a.y for a in sim.agents if a.species == "Predator" and a.alive]
                
                # 알(egg) 위치도 표시
                egg_x = [egg["x"] for egg in sim.eggs]
                egg_y = [egg["y"] for egg in sim.eggs]
                egg_colors = ["green" if egg["species"] == "Prey" else "red" for egg in sim.eggs]
                
                scatter_prey = ax.scatter(prey_x, prey_y, color='green', label="Prey")
                scatter_predator = ax.scatter(predator_x, predator_y, color='red', label="Predator")
                scatter_eggs = ax.scatter(egg_x, egg_y, color=egg_colors, marker='x', alpha=0.5, label="Eggs")
                
                ax.legend(loc="upper right")
                
                # 개체수 변화 그래프 업데이트 (충분한 데이터가 있을 때)
                if total_steps > 10 and len(time_points) > 1:
                    prey_line.set_data(time_points, prey_counts)
                    predator_line.set_data(time_points, predator_counts)
                    
                    # y축 범위 조정
                    max_count = max(max(prey_counts), max(predator_counts)) if prey_counts and predator_counts else 10
                    ax_counts.set_xlim(0, total_steps)
                    ax_counts.set_ylim(0, max_count * 1.1)  # 여유 있게 10% 더 여백
                
                plt.tight_layout()
                plt.pause(0.01)  # 짧은 지연으로 그림 업데이트
                
                # 비디오 프레임 저장
                if record_video:
                    frames.append([plt.gcf()])
                
        elif sim_type == "ReactionDiffusionModel":
            stats = sim.get_stats()
            
            # WandB에 로그
            wandb.log({
                "time": stats["time"],
                "total_prey": stats["total_prey"],
                "total_predator": stats["total_predator"],
                "max_prey": stats["max_prey"],
                "max_predator": stats["max_predator"]
            })
            
            # 데이터 추적
            time_points.append(stats["time"])
            prey_totals.append(stats["total_prey"])
            predator_totals.append(stats["total_predator"])
            
            # 주기적 시각화 업데이트
            if step % plot_interval == 0:
                # 밀도 맵 업데이트
                prey_img.set_array(sim.u)
                predator_img.set_array(sim.v)
                
                # 시간 표시 업데이트
                fig.suptitle(f"Time: {sim.time:.2f} / {sim.T}")
                
                # 인구 그래프 업데이트
                axs[2].clear()
                axs[2].set_title("Population over Time")
                axs[2].plot(time_points, prey_totals, 'g-', label="Prey")
                axs[2].plot(time_points, predator_totals, 'r-', label="Predator")
                axs[2].set_xlabel("Time")
                axs[2].set_ylabel("Total Population")
                axs[2].set_xlim(0, sim.T)
                axs[2].set_ylim(0, max(max(prey_totals), max(predator_totals)) * 1.1 if prey_totals else sim.K)
                axs[2].legend()
                
                plt.tight_layout()
                plt.pause(0.01)
                
                # 비디오 프레임 저장
                if record_video:
                    frames.append([plt.gcf()])
    
    # 비디오 저장
    if record_video and frames:
        print(f"비디오 저장 중: {video_filename}")
        ani = animation.ArtistAnimation(fig, frames, interval=200, blit=True)
        ani.save(video_filename, writer='ffmpeg')
        print(f"비디오 저장 완료: {video_filename}")
    
    plt.ioff()  # 인터랙티브 모드 종료
    # 마지막 결과만 표시하는 일반 플롯으로 전환
    plt.show()
    wandb.finish()

def run_reaction_diffusion_comparison(width, height, configs, total_steps, plot_interval=5,
                                      project_name="ReactionDiffusionComparison"):
    """여러 설정을 가진 반응-확산 모델을 비교 실행하는 함수
    
    Parameters:
    width, height: 시뮬레이션 공간 크기
    configs: 여러 설정의 리스트 [config1, config2, ...]
    total_steps: 시뮬레이션 전체 단계 수
    plot_interval: 몇 스텝마다 그래프 업데이트할지
    project_name: WandB 프로젝트 명
    """
    from model import ReactionDiffusionModel
    
    # 설정 개수에 따라 subplot 구성
    n_configs = len(configs)
    fig, axs = plt.subplots(n_configs, 3, figsize=(15, 5*n_configs))
    
    # 단일 설정인 경우 axs를 2D 배열로 변환
    if n_configs == 1:
        axs = np.array([axs])
    
    # 모델 초기화
    models = []
    imgs_prey = []
    imgs_predator = []
    time_points = [[] for _ in range(n_configs)]
    prey_totals = [[] for _ in range(n_configs)]
    predator_totals = [[] for _ in range(n_configs)]
    
    # WandB 초기화
    wandb.init(project=project_name, entity="mathematical-modeling")
    
    # 각 설정별 모델 및 초기 플롯 준비
    for i, config in enumerate(configs):
        # 모델 생성
        model = ReactionDiffusionModel(width, height, config)
        models.append(model)
        
        # 설정 상세 정보를 제목에 표시
        param_str = f"du={config.get('du', 0.25):.2f}, dv={config.get('dv', 0.05):.2f}"
        axs[i, 0].set_title(f"Prey - {param_str}")
        axs[i, 1].set_title(f"Predator - {param_str}")
        axs[i, 2].set_title(f"Population - {param_str}")
        
        # 초기 이미지 생성
        img_prey = axs[i, 0].imshow(model.u, cmap=plt.cm.Greens, vmin=0, vmax=model.K)
        img_predator = axs[i, 1].imshow(model.v, cmap=plt.cm.Reds, vmin=0, vmax=3)
        
        imgs_prey.append(img_prey)
        imgs_predator.append(img_predator)
        
        # 컬러바 추가
        plt.colorbar(img_prey, ax=axs[i, 0])
        plt.colorbar(img_predator, ax=axs[i, 1])
        
        # WandB에 설정 기록
        wandb.config.update({f"config_{i}": config})
    
    plt.ion()  # 인터랙티브 모드 활성화
    
    # 메인 시뮬레이션 루프
    for step in range(total_steps):
        # 각 모델 스텝 진행 및 데이터 수집
        for i, model in enumerate(models):
            model.step()
            stats = model.get_stats()
            
            # 데이터 추적
            time_points[i].append(stats["time"])
            prey_totals[i].append(stats["total_prey"])
            predator_totals[i].append(stats["total_predator"])
            
            # WandB에 로그
            wandb.log({
                f"time_{i}": stats["time"],
                f"total_prey_{i}": stats["total_prey"],
                f"total_predator_{i}": stats["total_predator"]
            })
        
        # 주기적 시각화 업데이트
        if step % plot_interval == 0:
            for i, model in enumerate(models):
                # 밀도 맵 업데이트
                imgs_prey[i].set_array(model.u)
                imgs_predator[i].set_array(model.v)
                
                # 인구 그래프 업데이트
                axs[i, 2].clear()
                axs[i, 2].plot(time_points[i], prey_totals[i], 'g-', label="Prey")
                axs[i, 2].plot(time_points[i], predator_totals[i], 'r-', label="Predator")
                axs[i, 2].set_xlabel("Time")
                axs[i, 2].set_ylabel("Total Population")
                axs[i, 2].legend()
            
            # 전체 제목 업데이트
            fig.suptitle(f"Comparison at time {models[0].time:.2f}")
            
            plt.tight_layout()
            plt.pause(0.01)
    
    plt.ioff()  # 인터랙티브 모드 종료
    plt.show()
    wandb.finish()

def run_delay_experiment(width, height, base_config, delay_values, total_steps, plot_interval=5,
                         project_name="DelayExperiment"):
    """번식 지연 효과를 실험하는 함수
    
    Parameters:
    width, height: 시뮬레이션 공간 크기
    base_config: 기본 설정 딕셔너리
    delay_values: 테스트할 지연 값 리스트 [(prey_delay1, pred_delay1), ...]
    total_steps: 시뮬레이션 전체 단계 수
    plot_interval: 몇 스텝마다 그래프 업데이트할지
    project_name: WandB 프로젝트 명
    """
    from model import RandomWalkSimulation, Agent
    
    # WandB 초기화
    wandb.init(project=project_name, entity="mathematical-modeling")
    wandb.config.update({"base_config": base_config, "delay_values": delay_values})
    
    # 각 지연 설정에 대한 결과 저장
    results = {f"prey_delay={p}_pred_delay={q}": {"time": [], "prey": [], "predator": []} 
               for p, q in delay_values}
    
    # 각 지연 값 조합에 대해 시뮬레이션 실행
    for prey_delay, pred_delay in delay_values:
        print(f"Simulating with prey_delay={prey_delay}, pred_delay={pred_delay}")
        
        # 설정 복사 및 지연 값 설정
        config = base_config.copy()
        config.update({
            "delay_prey": prey_delay,
            "delay_predator": pred_delay
        })
        
        # 시뮬레이션 초기화
        sim = RandomWalkSimulation(width, height, config)
        
        # 초기 개체 배치
        num_init_prey = config.get("num_init_prey", 100)
        num_init_predator = config.get("num_init_predator", 20)
        
        for _ in range(num_init_prey):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            sim.add_agent(Agent("Prey", x, y))
            
        for _ in range(num_init_predator):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            sim.add_agent(Agent("Predator", x, y))
        
        # 결과 키 생성
        result_key = f"prey_delay={prey_delay}_pred_delay={pred_delay}"
        
        # 시뮬레이션 실행
        for step in range(total_steps):
            sim.step()
            stats = sim.get_stats()
            
            # 결과 저장
            results[result_key]["time"].append(step)
            results[result_key]["prey"].append(stats["total_prey"])
            results[result_key]["predator"].append(stats["total_predator"])
            
            # WandB에 로그
            wandb.log({
                f"step": step,
                f"prey_{result_key}": stats["total_prey"],
                f"predator_{result_key}": stats["total_predator"]
            })
    
    # 결과 종합 플롯
    plt.figure(figsize=(15, 10))
    
    # 피식자 개체수 그래프
    plt.subplot(2, 1, 1)
    for key, data in results.items():
        plt.plot(data["time"], data["prey"], label=f"Prey - {key}")
    plt.title("Prey Population with Different Delay Values")
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.legend()
    
    # 포식자 개체수 그래프
    plt.subplot(2, 1, 2)
    for key, data in results.items():
        plt.plot(data["time"], data["predator"], label=f"Predator - {key}")
    plt.title("Predator Population with Different Delay Values")
    plt.xlabel("Time Step")
    plt.ylabel("Population")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("delay_experiment_results.png")
    wandb.log({"delay_experiment_plot": wandb.Image("delay_experiment_results.png")})
    
    plt.show()
    wandb.finish()
    
    return results 