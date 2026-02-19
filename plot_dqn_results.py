import matplotlib.pyplot as plt
import re
import numpy as np

log_file = "dqn_training_log.txt"

episodes = []
scores = []
avg_losses = []

current_episode_losses = []

with open(log_file, "r") as f:
    for line in f:
        # Check for Step line to extract loss
        if line.strip().startswith("Step:"):
            # specific format: Loss: 0.12345 or Loss: N/A
            match = re.search(r"Loss: ([0-9\.]+|N/A)", line)
            if match:
                loss_str = match.group(1)
                if loss_str != "N/A":
                    try:
                        current_episode_losses.append(float(loss_str))
                    except ValueError:
                        pass
        
        # Check for Episode summary line
        elif line.strip().startswith("Episode:"):
            # Format: Episode: 0, Score: 26, Epsilon: 1.00
            match = re.search(r"Episode: (\d+), Score: (\d+)", line)
            if match:
                episode_num = int(match.group(1))
                score = int(match.group(2))
                
                episodes.append(episode_num)
                scores.append(score)
                
                if current_episode_losses:
                    avg_loss = np.mean(current_episode_losses)
                else:
                    avg_loss = 0 # Or NaN, but 0 is easier to plot for initial episodes if needed
                
                avg_losses.append(avg_loss)
                current_episode_losses = [] # Reset for next episode

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:red'
ax1.set_xlabel('Episode')
ax1.set_ylabel('Score (Total Reward)', color=color)
ax1.plot(episodes, scores, color=color, label='Score')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Average Loss', color=color)  # we already handled the x-label with ax1
ax2.plot(episodes, avg_losses, color=color, label='Avg Loss', alpha=0.7)
ax2.tick_params(axis='y', labelcolor=color)

plt.title('DQN Training Results: Score and Loss per Episode')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig('dqn_training_results.png')
print("Plot saved to dqn_training_results.png")
