import numpy as np
import time
import os
import random

# ==========================================
# 1. 定义迷宫环境 (Maze Environment)
# ==========================================
class MazeEnv:
    def __init__(self):
        # 5x5 地图设计
        # S:起点, T:终点, #:墙壁(不通), .:路, O:陷阱(通但扣分)
        self.map = [
            ['S', '.', '.', '#', '.'],
            ['.', '#', '.', '#', '.'],
            ['.', '#', '.', '.', '.'],
            ['.', '.', 'O', '#', '.'],
            ['.', '.', '.', '#', 'T']
        ]
        self.n_rows = 5
        self.n_cols = 5
        self.robot_pos = (0, 0) # 起始位置
        self.target_pos = (4, 4)
        
        # 动作空间：上(0), 下(1), 左(2), 右(3)
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = 4

    def reset(self):
        self.robot_pos = (0, 0)
        return self.robot_pos

    def step(self, action):
        x, y = self.robot_pos
        
        # 即使撞墙，原来的位置也不能变，先存一下
        next_x, next_y = x, y
        
        # 尝试移动
        if action == 0:   # Up
            next_x = max(0, x - 1)
        elif action == 1: # Down
            next_x = min(self.n_rows - 1, x + 1)
        elif action == 2: # Left
            next_y = max(0, y - 1)
        elif action == 3: # Right
            next_y = min(self.n_cols - 1, y + 1)

        # 检查是否撞墙 (#)
        if self.map[next_x][next_y] == '#':
            # 撞墙了！位置保持不变，给个惩罚
            reward = -5
            done = False
            next_state = (x, y) # 弹回原地
        else:
            # 移动成功
            self.robot_pos = (next_x, next_y)
            next_state = (next_x, next_y)
            
            # 判断当前位置的奖励
            cell_type = self.map[next_x][next_y]
            
            if cell_type == 'T':    # 到达终点
                reward = 50
                done = True
            elif cell_type == 'O':  # 掉进陷阱
                reward = -20
                done = False
            else:                   # 普通路面
                reward = -1         # 每走一步扣1分，强迫它找最短路径
                done = False
                
        return next_state, reward, done

    def render(self):
        # 简单的文本可视化
        # os.system('cls' if os.name == 'nt' else 'clear') # 如果想清屏可以取消注释
        print("-" * 20)
        for i in range(self.n_rows):
            row_str = ""
            for j in range(self.n_cols):
                if (i, j) == self.robot_pos:
                    row_str += "🤖 " # 机器人当前位置
                elif self.map[i][j] == '#':
                    row_str += "⬛ " # 墙壁
                elif self.map[i][j] == 'T':
                    row_str += "🏁 " # 终点
                elif self.map[i][j] == 'O':
                    row_str += "❌ " # 陷阱
                else:
                    row_str += "⬜ " # 路
            print(row_str)
        print("-" * 20)

# ==========================================
# 2. Q-Learning 智能体
# ==========================================
class QLearningAgent:
    def __init__(self, n_rows, n_cols, n_actions):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.n_actions = n_actions
        # 初始化 Q 表：5x5x4 的三维数组
        self.q_table = np.zeros((n_rows, n_cols, n_actions))
        
        self.lr = 0.1       # 学习率 Alpha
        self.gamma = 0.9    # 折扣因子 Gamma
        self.epsilon = 0.1  # 探索率 Epsilon

    def choose_action(self, state, is_training=True):
        # Epsilon-Greedy 策略
        if is_training and np.random.uniform() < self.epsilon:
            return np.random.choice(self.n_actions) # 随机探索
        else:
            x, y = state
            # 即使 Q 值都一样，也随机选一个，防止死板
            state_action = self.q_table[x, y, :]
            # 找到最大值的索引（如果有多个最大值，随机选一个）
            max_indices = np.where(state_action == np.max(state_action))[0]
            return np.random.choice(max_indices)

    def learn(self, state, action, reward, next_state, done):
        x, y = state
        nx, ny = next_state
        
        q_predict = self.q_table[x, y, action]
        
        if done:
            q_target = reward
        else:
            q_target = reward + self.gamma * np.max(self.q_table[nx, ny, :])
            
        # 更新公式
        self.q_table[x, y, action] += self.lr * (q_target - q_predict)

# ==========================================
# 3. 主程序：训练 + 演示
# ==========================================
if __name__ == "__main__":
    env = MazeEnv()
    agent = QLearningAgent(env.n_rows, env.n_cols, env.n_actions)
    
    print("🚀 开始训练智能体...")
    
    # --- 训练阶段 ---
    EPISODES = 500
    for episode in range(EPISODES):
        state = env.reset()
        done = False
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state

    print("✅ 训练完成！现在演示智能体的走法：\n")
    time.sleep(1)

    # --- 演示阶段 (可视化) ---
    state = env.reset()
    done = False
    step_count = 0
    
    env.render() # 打印初始状态
    
    while not done:
        time.sleep(0.5) # 暂停0.5秒让你看清楚每一步
        
        # 这一步完全按照学到的 Q 表走 (不探索)
        action = agent.choose_action(state, is_training=False)
        state, reward, done = env.step(action)
        
        env.render()
        step_count += 1
        
        if step_count > 20: # 防止死循环（如果没训练好的话）
            print("迷路了...")
            break
            
    if done:
        print(f"🎉 成功抵达终点！共用了 {step_count} 步。")