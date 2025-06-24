import gymnasium
import numpy as np
import time

class RewardConverter(gymnasium.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.core_env = env.unwrapped
        self.left_foot_id = self.core_env.model.body('left_foot').id
        self.right_foot_id = self.core_env.model.body('right_foot').id
        self.is_spread_leg = True
        self.t = 0
        self.start = time.time()

    def reward(self, reward):
        left_foot_pos = self.core_env.data.xpos[self.left_foot_id]
        right_foot_pos = self.core_env.data.xpos[self.right_foot_id]

        left_foot_z = left_foot_pos[2]
        right_foot_z = right_foot_pos[2]

        end_time = time.time()
        self.t = end_time - self.start
        
        phase = 0.2 * (2 * np.pi * 0.1 * self.t)

        target_left = np.sin(phase)
        target_right = np.sin(phase + np.pi)

        left_diff = (left_foot_z - target_left)**2
        right_diff = (right_foot_z - target_right)**2

        phase_reward = -(left_diff + right_diff)

        self.start = end_time
        return reward + 0.1 * phase_reward
        
# --- 사용 방법 ---

# 1. 기본 Humanoid 환경을 생성합니다.
#base_env = gym.make("Humanoid-v4", render_mode="human") 

# 2. 생성한 래퍼로 환경을 감쌉니다.
# 가중치 값(weight)은 하이퍼파라미터이므로, 여러 값으로 실험하며 튜닝해야 합니다.
#wrapped_env = HumanoidRewardWrapper(
#    base_env,
#    control_cost_weight=0.1,
#    symmetry_cost_weight=0.5
#)
