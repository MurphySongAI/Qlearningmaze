import numpy as np

# ==========================================
# 0. 【关键修复】解决 numpy 和 gym 的版本冲突
# ==========================================
# 如果 numpy 版本较新，手动补上 bool8 属性
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import random
from collections import deque

# ==========================================
# 1. 定义 Q 网络 (大脑)
# ==========================================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# ==========================================
# 2. 定义 DQN 智能体
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10

        self.memory = deque(maxlen=10000) 

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        
        self.update_count = 0

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        else:
            # 确保 state 是 tensor 并且维度正确
            state = torch.FloatTensor(state)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_net(state)
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, action, reward, next_state, done = zip(*batch)
        
        state = torch.FloatTensor(np.array(state))
        action = torch.LongTensor(action).unsqueeze(1)
        reward = torch.FloatTensor(reward).unsqueeze(1)
        next_state = torch.FloatTensor(np.array(next_state))
        done = torch.FloatTensor(done).unsqueeze(1)

        q_eval = self.q_net(state).gather(1, action)

        with torch.no_grad():
            q_next = self.target_net(next_state).max(1)[0].unsqueeze(1)
        
        q_target = reward + (1 - done) * self.gamma * q_next

        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

# ==========================================
# 3. 主训练循环 (兼容修复版)
# ==========================================
if __name__ == '__main__':
    # 创建环境
    env = gym.make('CartPole-v1')
    
    # 获取状态维度 (兼容不同版本的 gym API)
    if hasattr(env.observation_space, 'shape'):
        state_dim = env.observation_space.shape[0]
    else:
        state_dim = 4 # CartPole 默认为 4
        
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    EPISODES = 200
    
    print("🚀 开始训练 DQN (已修复 numpy 和 reset 问题)...")
    
    for episode in range(EPISODES):
        # --- 兼容性修复 1: reset 返回值 ---
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0] # 新版 gym
        else:
            state = reset_result    # 旧版 gym
            
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            
            # --- 兼容性修复 2: step 返回值 ---
            step_result = env.step(action)
            if len(step_result) == 5:
                next_state, reward, terminated, truncated, _ = step_result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_result
            
            # 兼容性修复 3: 某些环境返回的 done 是 bool 类型，但也可能是 numpy.bool_
            # 这里统一转为 python 的 bool，避免 tensor 报错
            done = bool(done)

            # 修改奖励逻辑，杆子倒了给惩罚
            reward_to_store = reward
            if done and total_reward < 499:
                reward_to_store = -10
            
            agent.store_transition(state, action, reward_to_store, next_state, done)
            agent.learn()
            
            state = next_state
            total_reward += reward
            
            if done:
                print(f"Episode: {episode}, Score: {int(total_reward)}, Epsilon: {agent.epsilon:.2f}")
                
        if total_reward >= 500:
            print(f"✅ 在第 {episode} 局解决了问题！")
            break
            
    print("训练结束！")