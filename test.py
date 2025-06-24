import gymnasium as gym
import numpy as np

# Humanoid-v5 환경 생성 (gymnasium-robotics 필요)
env = gym.make("Humanoid-v5", render_mode="human") # "human"으로 설정하여 시각화
observation, info = env.reset()

# 1. 모델에서 발(body)의 ID를 이름으로 찾아옵니다.
#    (매번 찾을 필요 없이 한 번만 실행하면 됩니다.)
core_env = env.unwrapped
left_foot_id = core_env.model.body('left_foot').id
right_foot_id = core_env.model.body('right_foot').id

print(f"왼발(left_foot)의 Body ID: {left_foot_id}")
print(f"오른발(right_foot)의 Body ID: {right_foot_id}")
print("-" * 30)
print("시뮬레이션 시작! 100 스텝 동안 발의 위치를 출력합니다.")
print("창을 보면서 좌표가 어떻게 변하는지 확인해보세요.")

for i in range(100):
    action = env.action_space.sample()  # 무작위 행동 선택
    observation, reward, terminated, truncated, info = env.step(action)

    # 2. 매 스텝마다 업데이트되는 data 객체에서 위치 정보를 가져옵니다.
    # data.xpos는 모든 body의 (x, y, z) 위치를 담고 있는 배열입니다.
    left_foot_pos = core_env.data.xpos[left_foot_id]
    right_foot_pos = core_env.data.xpos[right_foot_id]
    
    if i % 10 == 0: # 10 스텝마다 출력
        print(f"Step {i+1:3d} | L-Foot: {np.round(left_foot_pos, 2)} | R-Foot: {np.round(right_foot_pos, 2)}")

    if terminated or truncated:
        observation, info = env.reset()

env.close() 