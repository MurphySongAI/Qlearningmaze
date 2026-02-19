from sys import modules
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
    # ==========================================
    # 继承自 nn.Module
    # nn.Module 是 PyTorch 所有神经网络模型的基类
    # 说明这是一个 神经网络模型
    # ==========================================
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        # 调用父类（nn.Module）的初始化函数，nn.Module 里面做了很多重要的初始化工作。
        # 如果你不调用它：你的模型就不是一个“真正的 PyTorch 模型”。
        # super() 的意思是：找到父类
        # 等价于super().__init__()
        self.fc = nn.Sequential(
            # nn.Sequential 代表按顺序把多个层串起来
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    # ==========================================
    # 前向传播
    # 定义模型的前向传播逻辑：输入 x，经过全连接网络 self.fc，输出结果。
    # ==========================================
    def forward(self, x):
        return self.fc(x)


# ==========================================
# 2. 定义 DQN 智能体
# 负责三件事：
# 1. 选动作
# 2. 存经验
# 3. 学习更新网络
# ==========================================
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim #状态向量
        self.action_dim = action_dim #动作空间
        
        self.gamma = 0.99 #折扣因子 Q=r+γQ(s',a') 表示未来奖励的重要程度。
        self.epsilon = 1.0 #探索率
        self.epsilon_min = 0.01 #最小探索率
        self.epsilon_decay = 0.995 #探索率衰减
        # ε-greedy 策略：探索率从1.0开始，随着训练进行逐渐衰减到0.01，确保初期充分探索，后期利用已有知识。
        self.learning_rate = 0.001 #学习率
        self.batch_size = 64 #每次训练采样64条经验
        self.target_update_freq = 10 #目标网络更新频率，每隔10次迭代更新一次目标网络，保持目标网络稳定，避免震荡。

        self.memory = deque(maxlen=10000) #经验回放
        # 存储数据(state, action, reward, next_state, done)，最多10000条

        self.q_net = QNetwork(state_dim, action_dim) #当前训练的网络
        self.target_net = QNetwork(state_dim, action_dim) #固定一段时间的“稳定目标网络”
        # 核心是两个网络
        self.target_net.load_state_dict(self.q_net.state_dict()) #一开始两个网络参数相同。

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
        self.loss_func = nn.MSELoss()
        # 优化器：Adam 优化器，用于更新 q_net 的参数。
        # 损失函数：均方误差 (MSE)，用于衡量预测值与目标值之间的差距。loss=（Q_eval-Q_target）^2
        
        self.update_count = 0

    def select_action(self, state):
        # 选动作 以epsilon的概率随机选，否则以1-epsilon的概率选Q值最大的动作
        # 最开始的探索率最大
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
        # 往经验池中添加一条经验

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # 如果经验池中的经验少于batch_size，不进行学习，不训练，不更新网络
        # 会先开始随机行动，收集经验，存入memory，等memory大于64的时候，再开始学习

        batch = random.sample(self.memory, self.batch_size)
        # 从经验池中随机采样batch_size = 64条经验
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
        # 贝尔曼方程 Q(s,a) = r + gamma * max Q(s',a')

        loss = self.loss_func(q_eval, q_target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # 反向传播，更新网络参数：清空梯度，计算梯度，更新参数

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            # 探索率衰减
            
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

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
    
    # Set up logging
    log_f = open("dqn_training_log.txt", "w", encoding='utf-8')
    def log(msg):
        print(msg)
        log_f.write(str(msg) + "\n") # Ensure msg is converted to string
        log_f.flush()

    log("🚀 开始训练 DQN (已修复 numpy 和 reset 问题)...")
    
    for episode in range(EPISODES):
        # --- 兼容性修复 1: reset 返回值 ---
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            state = reset_result[0] # 新版 gym
        else:
            state = reset_result    # 旧版 gym
            
        total_reward = 0
        done = False
        step_count = 0
        log(f"\n=== Episode {episode} Start ===")
        
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
            
            # --- 记录日志 ---
            # 1. 获取当前状态的 Q 值 (仅用于展示)
            state_tensor = torch.FloatTensor(state)
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            
            with torch.no_grad():
                q_values_log = agent.q_net(state_tensor).detach().numpy().flatten()
                q_values_str = "[" + ", ".join([f"{q:.3f}" for q in q_values_log]) + "]"

            # 2. 执行学习并获取 Loss
            agent.store_transition(state, action, reward_to_store, next_state, done)
            loss = agent.learn()
            
            # 3. 打印详细日志
            loss_str = f"{loss:.5f}" if loss is not None else "N/A"
            log(f"Step: {step_count:3d} | State: {np.round(state, 2)} | Q-values: {q_values_str} | "
                  f"Action: {action} | Reward: {reward:.1f} | Loss: {loss_str} | Epsilon: {agent.epsilon:.3f}")
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            if done:
                log(f"Episode: {episode}, Score: {int(total_reward)}, Epsilon: {agent.epsilon:.2f}")
                log("-" * 100)
                
        if total_reward >= 500:
            log(f"✅ 在第 {episode} 局解决了问题！")
            break
            
    log("训练结束！")
    log_f.close()
    torch.save(agent.q_net, "DQN_model.pth")