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
    def __init__(self, n0=100, off_policy=False):
        # Q-table dimensions: Dealer(1-10), Player(1-21), Actions(2)
        # Using size 11 and 22 to match indices directly (ignoring index 0)
        self.Q = np.zeros((11, 22, 2))
        # State visit counts N(s)
        self.N_state = np.zeros((11, 22))
        # State-Action visit counts N(s, a)
        self.N_action = np.zeros((11, 22, 2))
        self.n0 = n0
        self.off_policy = off_policy # True: Use Importance Sampling

    def get_action_prob(self, state):
        """Returns action probabilities (epsilon-greedy)."""
        d, p = state
        p = max(1, min(21, p))
        epsilon = self.n0 / (self.n0 + self.N_state[d, p])
        
        probs = np.ones(2) * (epsilon / 2.0)
        best_action = np.argmax(self.Q[d, p, :])
        probs[best_action] += (1.0 - epsilon)
        return probs

    def get_action(self, state):
        probs = self.get_action_prob(state)
        return np.random.choice(ACTIONS, p=probs)

    def train(self, env, num_episodes, Q_star=None, report_interval=None):        
        """
        Runs the MC Control algorithm.
        """
        mse_history = []
        if report_interval is None: report_interval = num_episodes

        for episode_idx in tqdm(range(num_episodes), desc=f"MC({'Off' if self.off_policy else 'On'})"):
            state = env.reset()
            episode = []
            done = False
            
            # 1. Generate Episode
            while not done:
                # Behavior Policy is always epsilon-greedy
                action_probs = self.get_action_prob(state)
                action = np.random.choice(ACTIONS, p=action_probs)
                next_state, reward, done = env.step(state, action)
                episode.append((state, action, reward, action_probs[action]))
                
                d, p = state
                self.N_state[d, p] += 1
                self.N_action[d, p, action] += 1
                state = next_state

            # 2. Update Q-values
            G = episode[-1][2] # Since gamma=1
            W = 1.0 # Importance Sampling Weight

            # Loop backwards
            for t in range(len(episode) - 1, -1, -1):
                s, a, r, prob_behavior = episode[t]
                d, p = s
                
                # --- On-Policy Update ---
                if not self.off_policy:
                    alpha = 1.0 / self.N_action[d, p, a]
                    self.Q[d, p, a] += alpha * (G - self.Q[d, p, a])
                
                # --- Off-Policy Update (Weighted Importance Sampling) ---
                else:
                    alpha = 1.0 / self.N_action[d, p, a]
                    self.Q[d, p, a] += alpha * W * (G - self.Q[d, p, a])

                    # Target Policy is Greedy (Deterministic)
                    # pi(a|s) = 1 if a is best, else 0
                    best_a = np.argmax(self.Q[d, p, :])
                    
                    if a == best_a:
                        prob_target = 1.0
                    else:
                        prob_target = 0.0
                        
                    # Update Weight W
                    # If action was not greedy, W becomes 0 and kills the update for previous steps
                    if prob_target == 0:
                        W = 0
                        break # No need to continue backwards as W is 0
                    
                    W = W * (prob_target / prob_behavior)

            if Q_star is not None and (episode_idx + 1) % report_interval == 0:
                mse_history.append(compute_mse(self.Q, Q_star))
        
        return mse_history
# ==========================================
# 3. Sarsa(lambda) Agent (Part 3)
# ==========================================
class SarsaAgent:
    """
    Sarsa(lambda) Agent for Easy21.
    """
    def __init__(self, lambd=0.0, n0=100, off_policy=False):
        self.lambd = lambd  # Lambda value (0 ~ 1)
        self.n0 = n0
        self.off_policy = off_policy # True: Q-Learning (Watkins), False: Sarsa
        self.Q = np.zeros((11, 22, 2))
        self.N_state = np.zeros((11, 22))
        self.N_action = np.zeros((11, 22, 2))
        self.E = np.zeros((11, 22, 2)) # Eligibility Trace

    def get_action(self, state):
        """Epsilon-greedy policy based on N0 constant."""
        d, p = state
        # Safety bound
        p = max(1, min(21, p))
        epsilon = self.n0 / (self.n0 + self.N_state[d, p])
        
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        else:
            values = self.Q[d, p, :]
            max_val = np.max(values)
            best_actions = [a for a, v in enumerate(values) if v == max_val]
            return np.random.choice(best_actions)

    def train(self, env, num_episodes, Q_star=None, report_interval=None):
        """
        Runs Sarsa(lambda) algorithm.
        If Q_star is provided, calculates MSE every report_interval episodes.
        """
        mse_history = []
        if report_interval is None: report_interval = num_episodes

        desc = f"Q-Learning({self.lambd})" if self.off_policy else f"Sarsa({self.lambd})"

        for episode_idx in tqdm(range(num_episodes), desc=desc , leave=False):
            # 1. Initialize Trace
            self.E.fill(0)
            
            state = env.reset()
            action = self.get_action(state)
            
            done = False
            while not done:
                next_state, reward, done = env.step(state, action)
                
                # --- Target Calculation ---
                if self.off_policy:
                    # Off-policy (Q-Learning): Target is based on Max Q
                    if not done:
                        best_next_val = np.max(self.Q[next_state[0], next_state[1], :])
                        td_target = reward + 1.0 * best_next_val
                        # For next loop: we still need actual action
                        next_action = self.get_action(next_state)
                    else:
                        td_target = reward
                        next_action = 0
                else:
                    # On-policy (Sarsa): Target is based on Actual Next Action
                    if not done:
                        next_action = self.get_action(next_state)
                        td_target = reward + 1.0 * self.Q[next_state[0], next_state[1], next_action]
                    else:
                        td_target = reward
                        next_action = 0

                # TD Error
                d, p = state
                delta = td_target - self.Q[d, p, action]
                
                self.N_state[d, p] += 1
                self.N_action[d, p, action] += 1
                self.E[d, p, action] += 1
                
                # Update Q
                visited_mask = self.N_action > 0
                alpha = np.zeros_like(self.Q)
                alpha[visited_mask] = 1.0 / self.N_action[visited_mask]
                self.Q += alpha * delta * self.E
                
                # Update Trace (Decay)
                if self.off_policy:
                    # Watkins' Q(lambda): If action taken was greedy, decay. Else, cut trace.
                    if not done:
                        best_next_action = np.argmax(self.Q[next_state[0], next_state[1], :])
                        if next_action == best_next_action:
                            self.E *= (1.0 * self.lambd)
                        else:
                            self.E.fill(0) # Cut trace
                else:
                    # Sarsa: Always decay
                    self.E *= (1.0 * self.lambd)
                
                state = next_state
                action = next_action

            if Q_star is not None and (episode_idx + 1) % report_interval == 0:
                mse_history.append(compute_mse(self.Q, Q_star))
        
        return mse_history
    
# ==========================================
# 4. Linear Function Approximation Agent (Part 4)
# ==========================================
class LinearSarsaAgent:
    """
    Sarsa(lambda) with Linear Function Approximation.
    Features: Coarse coding (overlapping cuboids).
    """
    def __init__(self, lambd=0.0, alpha=0.01, epsilon=0.05):
        self.lambd = lambd
        self.alpha = alpha
        self.epsilon = epsilon
        
        # Feature dimensions
        self.d_ranges = [[1, 4], [4, 7], [7, 10]]
        self.p_ranges = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self.n_features = len(self.d_ranges) * len(self.p_ranges) * 2 # 3*6*2 = 36
        
        # Weights (initialized to 0 or small random)
        self.w = np.zeros(self.n_features)
        
        # Eligibility Trace for weights
        self.e = np.zeros(self.n_features)

    def get_features(self, state, action):
        """
        Converts (state, action) into a binary feature vector phi(s, a).
        Returns a vector of size 36 with 1s at active indices.
        """
        d, p = state
        phi = np.zeros(self.n_features)
        
        # Compute active features
        # Structure: [Dealer(3) * Player(6) * Action(2)]
        # We flatten this 3D structure into 1D array of size 36.
        
        idx = 0
        for a in range(2): # Action loop (Hit=0, Stick=1)
            for d_idx, (d_min, d_max) in enumerate(self.d_ranges):
                for p_idx, (p_min, p_max) in enumerate(self.p_ranges):
                    
                    # Check if current state/action matches this feature's range
                    # Note: action must match exactly
                    is_active = (action == a) and \
                                (d_min <= d <= d_max) and \
                                (p_min <= p <= p_max)
                    
                    if is_active:
                        phi[idx] = 1.0
                    
                    idx += 1
        return phi

    def get_q_value(self, state, action):
        """
        Q(s, a) = w . phi(s, a)
        """
        phi = self.get_features(state, action)
        return np.dot(self.w, phi)

    def get_action(self, state):
        """
        Epsilon-greedy based on estimated Q-values.
        Note: For LFA, we usually use constant epsilon.
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 2)
        else:
            q_hit = self.get_q_value(state, ACTION_HIT)
            q_stick = self.get_q_value(state, ACTION_STICK)
            
            if q_hit > q_stick: return ACTION_HIT
            elif q_stick > q_hit: return ACTION_STICK
            else: return np.random.randint(0, 2)

    def train(self, env, num_episodes, Q_star=None, report_interval=None):
        mse_history = []
        if report_interval is None: report_interval = num_episodes
        
        for episode_idx in tqdm(range(num_episodes), desc=f"LinearSarsa({self.lambd})", leave=False):
            self.e.fill(0) # Reset traces
            state = env.reset()
            action = self.get_action(state)
            
            done = False
            while not done:
                # 1. Get Features for current (S, A)
                phi = self.get_features(state, action)
                q_val = np.dot(self.w, phi)
                
                # 2. Step
                next_state, reward, done = env.step(state, action)
                
                # 3. Calculate TD Target & Error
                if not done:
                    next_action = self.get_action(next_state)
                    q_next = self.get_q_value(next_state, next_action)
                    td_target = reward + 1.0 * q_next
                else:
                    next_action = 0 # Dummy
                    td_target = reward
                
                delta = td_target - q_val
                
                # 4. Update Trace
                # Accumulating traces: e <- gamma*lambda*e + phi
                self.e *= (1.0 * self.lambd)
                self.e += phi
                
                # 5. Update Weights
                # w <- w + alpha * delta * e
                self.w += self.alpha * delta * self.e
                
                state = next_state
                action = next_action

            if Q_star is not None and (episode_idx + 1) % report_interval == 0:
                # To compute MSE, we must reconstruct the full Q-table from weights
                current_Q = self.reconstruct_Q_table()
                mse_history.append(compute_mse(current_Q, Q_star))
                
        return mse_history

    def reconstruct_Q_table(self):
        """
        Helper to convert learned weights back to a table for comparison/plotting.
        """
        Q_table = np.zeros((11, 22, 2))
        for d in range(1, 11):
            for p in range(1, 22):
                for a in range(2):
                    Q_table[d, p, a] = self.get_q_value((d, p), a)
        return Q_table

# ==========================================
# 5. Helper Functions
# ==========================================
def compute_mse(Q, Q_star):
    """
    Computes Sum of Squared Errors between Q and Q_star.
    Only considers valid states (Dealer 1-10, Player 1-21).
    """
    diff = Q[1:11, 1:22, :] - Q_star[1:11, 1:22, :]
    return np.sum(diff ** 2)

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
    plt.close()

def plot_learning_curve(history_lambda0, history_lambda1, interval, filename="sarsa_learning_curve.png"):
    """
    Plots learning curves for lambda=0 and lambda=1.
    """
    plt.figure(figsize=(10, 6))
    
    x_axis = np.arange(len(history_lambda0)) * interval
    
    plt.plot(x_axis, history_lambda0, label="lambda = 0")
    plt.plot(x_axis, history_lambda1, label="lambda = 1")
    
    plt.xlabel("Episode Number")
    plt.ylabel("Mean Squared Error (Sum of Squares)")
    plt.title("Sarsa Learning Curve (MSE vs Episodes)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.show()

def plot_mse_vs_lambda(lambdas, final_errors, filename="mse_vs_lambda.png"):
    """
    Plots Final MSE vs Lambda values.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, final_errors, marker='o')
    plt.xlabel("Lambda")
    plt.ylabel("Mean Squared Error (Sum of Squares)")
    plt.title("Performance by Lambda (after fixed episodes)")
    plt.xticks(lambdas)
    plt.grid(True)
    plt.savefig(filename)
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

def plot_comparison(results, interval, title, filename):
    plt.figure(figsize=(10, 6))
    for name, history in results.items():
        plt.plot(np.arange(len(history)) * interval, history, label=name)
    plt.title(title)
    plt.xlabel("Episode"); plt.ylabel("MSE (log scale)")
    plt.legend(); plt.grid(True)
    # learning speed will be more visible if seen as log scale
    # plt.yscale('log') 
    plt.savefig(filename); plt.show()
# ==========================================
# 6. Main Execution
# ==========================================
if __name__ == "__main__":
    # --- Experiment Settings ---
    NUM_EPISODES_MC = 1_000_000     # To get a good Q*
    NUM_EPISODES_SARSA = 10_000     # Recommended 1000 in assignment, but 10k gives smoother curves
    MSE_REPORT_INTERVAL = 100       # Calculate MSE every X episodes
    
    env = Easy21Env()
    
    # -------------------------------------------------
    # Step 1: Run MC Control to get Ground Truth (Q*)
    # -------------------------------------------------
    print(f"--- [Step 1] Running MC for {NUM_EPISODES_MC} eps to get Q* ---")
    mc_agent = MCAgent(n0=100, off_policy=False)
    mc_agent.train(env, NUM_EPISODES_MC)
    Q_star = mc_agent.Q.copy() # Copy Q to freeze it as ground truth
    plot_value_function(Q_star, "ground_truth_Q_star.png")
    print("MC Training Done. Q* Saved.\n")

    # -------------------------------------------------
    # Step 2: Run Sarsa(lambda) Experiments
    # -------------------------------------------------
    lambdas = np.linspace(0, 1, 11) # 0.0, 0.1, ... 1.0
    final_errors = []
    
    # Store history specifically for lambda=0 and lambda=1 for plotting
    history_lambda_0 = []
    history_lambda_1 = []

    print(f"--- [Step 2] Running Sarsa Experiments (Lambdas: {lambdas}) ---")
    
    for lam in lambdas:
        # Important: Create a FRESH agent for each lambda experiment
        sarsa = SarsaAgent(lambd=lam, n0=100)
        
        # Train and collect MSE history
        mse_hist = sarsa.train(env, NUM_EPISODES_SARSA, Q_star, MSE_REPORT_INTERVAL)
        
        # Record final error
        final_errors.append(mse_hist[-1])
        
        # Save specific histories for learning curve plot
        if lam == 0.0:
            history_lambda_0 = mse_hist
        elif lam == 1.0:
            history_lambda_1 = mse_hist
            
    print("\nExperiments Completed.")

    plot_mse_vs_lambda(lambdas, final_errors)
    print("Saved 'mse_vs_lambda.png'")

    # -------------------------------------------------
    # Step 3: MC vs TD
    # -------------------------------------------------
    print("\n--- [Step 3] MC vs TD Performance Comparison ---")
    results = {}
    
    # (1) Sarsa(0) - Pure TD
    print("Running Sarsa(0)...")
    sarsa0 = SarsaAgent(lambd=0.0, n0=100)
    results['Sarsa(0) (TD)'] = sarsa0.train(env, 10000, Q_star, 100)
    
    # (2) Sarsa(1) - Monte Carlo Equivalent
    print("Running Sarsa(1)...")
    sarsa1 = SarsaAgent(lambd=1.0, n0=100)
    results['Sarsa(1) (Trace)'] = sarsa1.train(env, 10000, Q_star, 100)
    
    # (3) MC On-policy
    print("Running MC On-policy...")
    mc = MCAgent(n0=100, off_policy=False)
    results['MC On-policy'] = mc.train(env, 10000, Q_star, 100)
    
    plot_comparison(results, 100, "Learning Curve: MC vs TD", "mc_vs_td_comparison.png")
    print("\nAll Done! Check 'mc_vs_td_comparison.png'.")

    # -------------------------------------------------
    # Step 4: Compare TD Algorithms (Sarsa vs Q-Learning)
    # -------------------------------------------------
    print(f"--- [Step 2] Comparing TD Algorithms ({NUM_EPISODES_SARSA} episodes) ---")
    
    results_td = {}
    
    # 1. Sarsa(0) - On-policy
    print("Running Sarsa(0)...")
    agent = SarsaAgent(lambd=0.0, n0=100, off_policy=False)
    results_td['Sarsa(0)'] = agent.train(env, NUM_EPISODES_SARSA, Q_star, MSE_REPORT_INTERVAL)
    
    # 2. Q-Learning(0) - Off-policy
    print("Running Q-Learning(0)...")
    agent = SarsaAgent(lambd=0.0, n0=100, off_policy=True)
    results_td['Q-Learning(0)'] = agent.train(env, NUM_EPISODES_SARSA, Q_star, MSE_REPORT_INTERVAL)
    
    # 3. Sarsa(0.5) - On-policy with Trace
    print("Running Sarsa(0.5)...")
    agent = SarsaAgent(lambd=0.5, n0=100, off_policy=False)
    results_td['Sarsa(0.5)'] = agent.train(env, NUM_EPISODES_SARSA, Q_star, MSE_REPORT_INTERVAL)
    
    # 4. Watkins' Q(0.5) - Off-policy with Trace
    print("Running Watkins-Q(0.5)...")
    agent = SarsaAgent(lambd=0.5, n0=100, off_policy=True)
    results_td['Watkins-Q(0.5)'] = agent.train(env, NUM_EPISODES_SARSA, Q_star, MSE_REPORT_INTERVAL)

    # Plot TD Comparison
    plot_comparison(results_td, MSE_REPORT_INTERVAL, 
                   "MSE Convergence: Sarsa vs Q-Learning", 
                   "td_comparison.png")
    
    # -------------------------------------------------
    # Step 5: Linear Approximator Experiments
    # -------------------------------------------------
    # Settings
    NUM_EPISODES = 10_000
    INTERVAL = 100
    
    results = {}
    
    # Linear Sarsa Comparisons
    print("\n--- Comparing Linear Approximation vs Tabular ---")
    
    # (1) Linear Sarsa (lambda=0)
    lin_sarsa0 = LinearSarsaAgent(lambd=0.0, alpha=0.01, epsilon=0.05)
    results['Linear Sarsa(0)'] = lin_sarsa0.train(env, NUM_EPISODES, Q_star, INTERVAL)
    
    # (2) Linear Sarsa (lambda=1)
    lin_sarsa1 = LinearSarsaAgent(lambd=1.0, alpha=0.01, epsilon=0.05)
    results['Linear Sarsa(1)'] = lin_sarsa1.train(env, NUM_EPISODES, Q_star, INTERVAL)
    
    # (3) Tabular Sarsa(0)
    tab_sarsa0 = SarsaAgent(lambd=0.0, n0=100)
    results['Tabular Sarsa(0)'] = tab_sarsa0.train(env, NUM_EPISODES, Q_star, INTERVAL)

    # Plot
    plot_comparison(results, INTERVAL, "MSE: Linear Approx vs Tabular", "linear_vs_tabular.png")
    
    # Plot Final Value Function of Linear Agent
    print("Plotting Linear Agent's Value Function...")
    final_Q_linear = lin_sarsa0.reconstruct_Q_table()
    plot_value_function(final_Q_linear, "linear_value_function.png")