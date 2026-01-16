import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bar

# Constants for readability
ACTION_HIT = 0
ACTION_STICK = 1
ACTIONS = [ACTION_HIT, ACTION_STICK]

MIN_CARD = 1
MAX_CARD = 10

# ==========================================
# 1. Environment Implementation (Part 1)
# ==========================================
class Easy21Env:
    """
    Easy21 Environment Implementation.
    Rules based on the assignment description.
    """
    def __init__(self):
        # Discount factor is 1 (no discounting)
        self.gamma = 1.0 

    def _draw_card(self):
        """
        Draws a card from the infinite deck.
        - Value: 1 to 10 (uniform)
        - Color: Red (prob 1/3, value -x), Black (prob 2/3, value +x)
        """
        value = np.random.randint(MIN_CARD, MAX_CARD + 1)
        # 1/3 chance of Red (-1), 2/3 chance of Black (+1)
        color = -1 if np.random.random() < (1/3) else 1
        return value * color

    def reset(self):
        """
        Starts a new game.
        Both player and dealer draw one black card (positive value).
        Returns: initial state (dealer_card, player_sum)
        """
        dealer_card = np.random.randint(MIN_CARD, MAX_CARD + 1)
        player_card = np.random.randint(MIN_CARD, MAX_CARD + 1)
        # State: (dealer's first card, player's current sum)
        return (dealer_card, player_card)

    def step(self, state, action):
        """
        Executes an action in the environment.
        Args:
            state: (dealer_card, player_sum)
            action: 0 (Hit) or 1 (Stick)
        Returns:
            next_state: (dealer_card, player_sum)
            reward: -1, 0, or 1
            done: Boolean indicating if the game is over
        """
        dealer_card, player_sum = state

        # --- Player's Turn ---
        if action == ACTION_HIT:
            # Player draws a card
            card = self._draw_card()
            player_sum += card
            
            # Check for Bust (sum < 1 or sum > 21)
            if player_sum < 1 or player_sum > 21:
                # Player goes bust, reward -1, game over
                return (dealer_card, player_sum), -1, True
            else:
                # Game continues, reward 0
                return (dealer_card, player_sum), 0, False

        # --- Dealer's Turn (if Player Sticks) ---
        elif action == ACTION_STICK:
            dealer_sum = dealer_card
            
            # Dealer hits until sum is 17 or greater
            while dealer_sum < 17 and dealer_sum >= 1:
                dealer_sum += self._draw_card()
            
            # Determine outcome
            reward = 0
            if dealer_sum < 1 or dealer_sum > 21:
                # Dealer busts, Player wins
                reward = 1
            elif player_sum > dealer_sum:
                # Player has higher sum, Player wins
                reward = 1
            elif player_sum < dealer_sum:
                # Dealer has higher sum, Player loses
                reward = -1
            else:
                # Draw
                reward = 0
            
            # Game is over after dealer's turn
            return (dealer_card, player_sum), reward, True
        
        else:
            raise ValueError("Invalid Action")

# ==========================================
# 2. Monte-Carlo Control Agent (Part 2)
# ==========================================
class MCAgent:
    """
    Monte-Carlo Control Agent for Easy21.
    """
    def __init__(self, n0=100):
        # Q-table dimensions: Dealer(1-10), Player(1-21), Actions(2)
        # Using size 11 and 22 to match indices directly (ignoring index 0)
        self.Q = np.zeros((11, 22, 2))
        
        # State visit counts N(s)
        self.N_state = np.zeros((11, 22))
        
        # State-Action visit counts N(s, a)
        self.N_action = np.zeros((11, 22, 2))
        
        self.n0 = n0

    def get_action(self, state):
        """
        Epsilon-greedy policy based on N0 constant.
        epsilon_t = N0 / (N0 + N(s_t))
        """
        d, p = state
        # Safety check for bounds (though env logic should prevent this)
        p = max(1, min(21, p)) 

        epsilon = self.n0 / (self.n0 + self.N_state[d, p])
        
        if np.random.random() < epsilon:
            return np.random.choice(ACTIONS)
        else:
            # Pick action with max Q value
            # If values are equal, pick random to avoid bias
            values = self.Q[d, p, :]
            max_val = np.max(values)
            best_actions = [a for a, v in enumerate(values) if v == max_val]
            return np.random.choice(best_actions)

    def train(self, env, num_episodes):
        """
        Runs the MC Control algorithm.
        """
        for _ in tqdm(range(num_episodes), desc="MC Training"):
            state = env.reset()
            episode = [] # Store (state, action, reward)
            done = False
            
            # 1. Generate an episode
            while not done:
                action = self.get_action(state)
                next_state, reward, done = env.step(state, action)
                
                # Store experience
                episode.append((state, action, reward))
                
                # Update counters immediately (needed for epsilon)
                d, p = state
                self.N_state[d, p] += 1
                self.N_action[d, p, action] += 1
                
                state = next_state

            # 2. Update Q-values (Backward pass)
            # Since gamma = 1, the return G_t is just the terminal reward for all steps
            # (Because intermediate rewards are 0)
            G = episode[-1][2] 
            
            for t in range(len(episode)):
                s, a, r = episode[t]
                d, p = s
                
                # Update rule: Q(s,a) <- Q(s,a) + alpha * (G - Q(s,a))
                # Step-size alpha = 1 / N(s, a)
                alpha = 1.0 / self.N_action[d, p, a]
                self.Q[d, p, a] += alpha * (G - self.Q[d, p, a])

# ==========================================
# 3. Helper Functions for Visualization
# ==========================================
def plot_value_function(Q, filename="mc_value_function.png"):
    """
    Plots the optimal value function V*(s) = max_a Q*(s, a)
    as a 3D surface plot.
    """
    # Calculate V*(s)
    # Extract only valid ranges: Dealer 1-10, Player 1-21
    V = np.max(Q[1:11, 1:22, :], axis=2)

    # Create grid
    X = np.arange(1, 11)   # Dealer showing
    Y = np.arange(1, 22)   # Player sum
    X, Y = np.meshgrid(X, Y)
    
    # Transpose V to match meshgrid shape (Player x Dealer) for plotting
    Z = V.T 

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_zlabel('Value V*')
    ax.set_title('Optimal Value Function (Monte-Carlo)')
    
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.show()

def plot_policy(Q, filename="mc_optimal_policy.png"):
    """
    Plots the optimal policy: Stick (1) or Hit (0)
    """
    # 1. Getting the optimal action (argmax)
    # dealer: 1~10, player: 1~21
    # Q[d, p, 0] = Hit Value, Q[d, p, 1] = Stick Value
    # policy[d, p] = 0 (Hit) or 1 (Stick)
    policy = np.argmax(Q[1:11, 1:22, :], axis=2)

    # 2. Visualization (Heatmap)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cax = ax.imshow(policy.T, cmap='coolwarm', origin='lower', extent=[0.5, 10.5, 0.5, 21.5])
    
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title('Optimal Policy (Red=Stick, Blue=Hit)')
    
    cbar = fig.colorbar(cax, ticks=[0, 1])
    cbar.ax.set_yticklabels(['Hit (0)', 'Stick (1)'])

    plt.savefig(filename)
    print(f"Policy plot saved as {filename}")
    plt.show()

# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # Settings
    NUM_EPISODES = 2_000_000 # Recommended to use > 500k for convergence
    
    # Initialize Environment and Agent
    env = Easy21Env()
    agent = MCAgent(n0=100)
    
    # Run Training
    print(f"Starting Monte-Carlo Control for {NUM_EPISODES} episodes...")
    agent.train(env, NUM_EPISODES)
    print("Training Completed.")
    
    # Plot Results
    print("Plotting Value Function...")
    plot_value_function(agent.Q)

    print("Plotting Optimal Policy...")
    plot_policy(agent.Q)