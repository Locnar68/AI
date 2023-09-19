import numpy as np
import gym
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state, :])

    def update(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.gamma * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def ensure_single_integer(state):
    if isinstance(state, tuple):
        return state[0]
    return state

	
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode="human")

n_actions = env.action_space.n
n_states = env.observation_space.n

agent = QLearningAgent(n_states, n_actions)

# Training the agent
n_episodes = 2000
max_steps_per_episode = 100

for episode in range(n_episodes):
    state = env.reset()
    state = ensure_single_integer(state)
    done = False
    for _ in range(max_steps_per_episode):
        action = agent.choose_action(state)
        step_results = env.step(action)
        next_state, reward, done = step_results[0], step_results[1], step_results[2]
        next_state = ensure_single_integer(next_state)
        
        # Visualize the game being played during training
        env.render()
        
        agent.update(state, action, reward, next_state)
        state = next_state
        if done:
            break

print("Training finished.\n")

# Test the agent's performance after training
n_test_episodes = 10
total_wins = 0

for episode in range(n_test_episodes):
    state = env.reset()
    state = ensure_single_integer(state)
    done = False
    while not done:
        action = np.argmax(agent.q_table[state, :])
        step_results = env.step(action)
        next_state, reward, done = step_results[0], step_results[1], step_results[2]
        next_state = ensure_single_integer(next_state)
        
        # Visualize the game being played during testing
        env.render()
        
        state = next_state
        if done and reward == 1.0:
            total_wins += 1

print(f"Agent won {total_wins} out of {n_test_episodes} episodes.")
