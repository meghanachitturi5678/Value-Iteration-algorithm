import numpy as np

# Defining the Grid World
grid = [    [0,  1, -1],
    [0,  0,  0],
    [0, 'wall', 0],
    [0,  0,  0]
]

# Defining the rewards, step cost, probabilities and discount factor
reward_goal = 1
reward_penalty = -1
step_cost = -0.04
prob_action = 0.7
prob_perp = 0.15
discount_factor = 0.95

# Initializing the utilities of all states,the goal state to 1 and penalty state to -1
num_rows, num_cols = len(grid), len(grid[0])
utilities = np.zeros((num_rows, num_cols))
utilities[0][1]=1
utilities[0][2]=-1

# Initializing the policy of all states to None
policy = [[None for j in range(num_cols)] for i in range(num_rows)]

# Defining a helper function to calculate the expected utility of a state
def expected_utility(row, col, action, utilities):
    total_utility = 0
    for a in [(action, prob_action), ((action+1)%4, prob_perp), ((action-1)%4, prob_perp)]:
        if a[0] == 0: # go up
            if row == 0 or grid[row-1][col] == 'wall':
                total_utility += a[1] * utilities[row][col]
            else:
                total_utility += a[1] * utilities[row-1][col]
        elif a[0] == 1: # go right
            if col == num_cols-1 or grid[row][col+1] == 'wall':
                total_utility += a[1] * utilities[row][col]
            else:
                total_utility += a[1] * utilities[row][col+1]
        elif a[0] == 2: # go down
            if row == num_rows-1 or grid[row+1][col] == 'wall':
                total_utility += a[1] * utilities[row][col]
            else:
                total_utility += a[1] * utilities[row+1][col]
        else: # go left
            if col == 0 or grid[row][col-1] == 'wall':
                total_utility += a[1] * utilities[row][col]
            else:
                total_utility += a[1] * utilities[row][col-1]
    return total_utility

print("Assuming Utility of wall to be zero")
print()

# Value Iteration Algorithm
num_iteration = 0
while True:
    delta = 0
    utilities_new = np.zeros((num_rows, num_cols))
    num_iteration+=1

    for i in range(num_rows):
        for j in range(num_cols):
            if grid[i][j] == 'wall':
                continue
            if (i, j) == (0, 1):
                utilities_new[i][j] = reward_goal
                continue
            if (i, j) == (0, 2):
                utilities_new[i][j] = reward_penalty
                continue

            max_utility = float("-inf")
            for a in range(4): # for all actions
                u = expected_utility(i, j, a, utilities)
                # print(u)
                if u > max_utility:
                    max_utility = u
                    best_action = a

            utilities_new[i][j] = step_cost + discount_factor * max_utility
            policy[i][j] = best_action
            delta = max(delta, abs(utilities_new[i][j] - utilities[i][j]))
            
    utilities=utilities_new
    if delta <= 0.0001:
        break
    print("Iteration",num_iteration, "Utilities:")
    print(utilities)
    print()

# Printing utility and policy of converged iteration
print("Iteration",num_iteration, "Utilities:")
print(utilities)
print()
print("Convergence is seen from iteration 21 to iteration 22")
print()
print("Policy for each cell in iteration 22:")
print()
print("Note:In Policy below,0=>up,1=>right,2=>down,3=>left")
print()
print(policy)
