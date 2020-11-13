try:
    import seaborn as sns
    import pandas as pd
    import json
    import numpy as np
    import matplotlib.pyplot as plt

except:
    get_ipython().system('pip install seaborn')
    get_ipython().system('pip install pandas')
    get_ipython().system('pip install json')
    get_ipython().system('pip install matplotlib')
    import seaborn as sns
    import pandas as pd
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    %
 
# INSERT TRAINING LOG PATHS
dqn_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Logs/2layers_128units_DQN3.log'
ddqn_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Logs/2layers_128units_DDQN1.log'
duelavg_dqn_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Logs/2layers_128units_DuelAvgDQN1.log'
duelmax_dqn_path = 'C:/Users/Sam/Documents/UTIAS/2nd Year/2nd Semester/AER 1517/Assignments/Project/Logs/2layers_128units_DuelMaxDQN1.log'


# Load in training logs
with open(dqn_path) as f:
    dqn_log = pd.read_json(f)
    
with open(ddqn_path) as f:
    ddqn_log = pd.read_json(f)

with open(duelavg_dqn_path) as f:
    duelavg_dqn_log = pd.read_json(f)
    
with open(duelmax_dqn_path) as f:
    duelmax_dqn_log = pd.read_json(f)


# Rolling mean rewards with window of 20 to smooth data
dqn_log['rolling_reward'] = dqn_log.episode_reward.rolling(20).mean()
ddqn_log['rolling_reward'] = ddqn_log.episode_reward.rolling(20).mean()
duelavg_dqn_log['rolling_reward'] = duelavg_dqn_log.episode_reward.rolling(20).mean()
duelmax_dqn_log['rolling_reward'] = duelmax_dqn_log.episode_reward.rolling(20).mean()

dqn_log['rolling_reward_std'] = dqn_log.episode_reward.rolling(20).std()
ddqn_log['rolling_reward_std'] = ddqn_log.episode_reward.rolling(20).std()
duelavg_dqn_log['rolling_reward_std'] = duelavg_dqn_log.episode_reward.rolling(20).std()
duelmax_dqn_log['rolling_reward_std'] = duelmax_dqn_log.episode_reward.rolling(20).std()

# Rolling mean Q-values
dqn_log['rolling_q'] = dqn_log.mean_q.rolling(5).mean()
ddqn_log['rolling_q'] = ddqn_log.mean_q.rolling(5).mean()
duelavg_dqn_log['rolling_q'] = duelavg_dqn_log.mean_q.rolling(5).mean()
duelmax_dqn_log['rolling_q'] = duelmax_dqn_log.mean_q.rolling(5).mean()

dqn_log['rolling_q_std'] = dqn_log.mean_q.rolling(5).std()
ddqn_log['rolling_q_std'] = ddqn_log.mean_q.rolling(5).std()
duelavg_dqn_log['rolling_q_std'] = duelavg_dqn_log.mean_q.rolling(5).std()
duelmax_dqn_log['rolling_q_std'] = duelmax_dqn_log.mean_q.rolling(5).std()

# Extract total training times
print('Total Training Time')
print('DQN: {}'.format(dqn_log['duration'].sum()))
print('DDQN: {}'.format(ddqn_log['duration'].sum()))
print('DuelMax DQN: {}'.format(duelavg_dqn_log['duration'].sum()))
print('DuelAvg DQN: {}'.format(duelmax_dqn_log['duration'].sum()))


# Reward plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))
ax1.plot(np.ones(20)*300, np.linspace(-200,400,20), '--b')
sns.lineplot(x = "episode", y = "rolling_reward", data = dqn_log, ax = ax1).set_title('DQN')
ax2.plot(np.ones(20)*250, np.linspace(-250,400,20), '--b')
sns.lineplot(x = "episode", y = "rolling_reward", data = ddqn_log, ax = ax2).set_title('DDQN')
ax3.plot(np.ones(20)*100, np.linspace(-180,400,20), '--b')
sns.lineplot(x = "episode", y = "rolling_reward", data = duelavg_dqn_log, ax = ax3).set_title('Dueling DQN ("avg")')
ax4.plot(np.ones(20)*180, np.linspace(-400,350,20), '--b')
sns.lineplot(x = "episode", y = "rolling_reward", data = duelmax_dqn_log, ax = ax4).set_title('Dueling DQN ("max")')
ax1.errorbar(dqn_log['episode'], dqn_log['rolling_reward'], yerr=dqn_log['rolling_reward_std'], fmt='-', ecolor = 'blue', alpha = 0.02)
ax2.errorbar(ddqn_log['episode'], ddqn_log['rolling_reward'], yerr=ddqn_log['rolling_reward_std'], fmt='-', ecolor = 'blue', alpha = 0.02)
ax3.errorbar(duelavg_dqn_log['episode'], duelavg_dqn_log['rolling_reward'], yerr=duelavg_dqn_log['rolling_reward_std'], fmt='-', ecolor = 'blue', alpha = 0.02)
ax4.errorbar(duelmax_dqn_log['episode'], duelmax_dqn_log['rolling_reward'], yerr=duelmax_dqn_log['rolling_reward_std'], fmt='-', ecolor = 'blue', alpha = 0.02)
ax1.set_xlabel(' ')
ax1.set_ylabel('Cumulative Reward')
ax1.legend(['Task Learned'], loc = 4)
ax2.set_xlabel(' ')
ax2.set_ylabel('Cumulative Reward')
ax2.legend(['Task Learned'], loc = 4)
ax3.set_xlabel('Episode')
ax3.set_ylabel('Cumulative Reward')
ax3.legend(['Task Learned'], loc = 4)
ax4.set_xlabel('Episode')
ax4.set_ylabel('Cumulative Reward')
ax4.legend(['Task Learned'], loc = 4)

# Mean Q-value plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2, figsize=(15,10))
sns.lineplot(x = "episode", y = "rolling_q", data = dqn_log, ax = ax1).set_title('DQN')
sns.lineplot(x = "episode", y = "rolling_q", data = ddqn_log, ax = ax2).set_title('DDQN')
sns.lineplot(x = "episode", y = "rolling_q", data = duelavg_dqn_log, ax = ax3).set_title('Dueling DQN ("avg")')
sns.lineplot(x = "episode", y = "rolling_q", data = duelmax_dqn_log, ax = ax4).set_title('Dueling DQN ("max")')


# Merge Q-values to one dataframe
df = pd.DataFrame({'dqn': dqn_log['mean_q'], 'ddqn': ddqn_log['mean_q'], 'duelavg_dqn': duelavg_dqn_log['mean_q'], 'duelmax_dqn': duelmax_dqn_log['mean_q']},
                  index = dqn_log['episode'])

# DQN vs. DDQN comparison plot
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,10))
ax1.plot(df.index, df['dqn'].values)
ax1.plot(df.index, df['ddqn'].values)
ax1.legend(['DQN', 'DDQN'])
ax1.set_ylabel('Mean Q-Value')
ax1.set_title('DQN vs. DDQN')
sns.lineplot(x = "episode", y = "rolling_reward", data = dqn_log, ax = ax2)
sns.lineplot(x = "episode", y = "rolling_reward", data = ddqn_log, ax = ax2)
ax2.legend(['DQN', 'DDQN'])
ax2.set_xlabel('Episode')
ax2.set_ylabel('Cumulative Reward')