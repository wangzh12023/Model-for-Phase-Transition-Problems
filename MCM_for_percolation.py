import random
import numpy as np
import matplotlib.pyplot as plt

def dfs(grid, visited, x, y, n) -> bool: 
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]   
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        if visited[cx, cy]:
            continue
        visited[cx, cy] = True
        for dx, dy in DIRECTIONS:
            nx, ny = cx + dx, cy + dy
            if (0 <= nx < n and 0 <= ny < n) and not visited[nx, ny] and grid[nx, ny] == 1:
                if nx == n - 1:  
                    return True
                stack.append((nx, ny))
    return False

def percolates(grid, n) -> bool:
    visited = np.zeros((n, n), dtype=bool)
    for col in range(n):
        if grid[0, col] == 1 and not visited[0, col] and dfs(grid, visited, 0, col, n):
            return True
    return False

def monte_carlo_simulation(n) -> float:
    grid = np.zeros((n, n), dtype=int)  
    open_sites = 0
    while True:
        x, y = random.randint(0, n - 1), random.randint(0, n - 1)
        while grid[x, y] == 1:  
            x, y = random.randint(0, n - 1), random.randint(0, n - 1)
        grid[x, y] = 1
        open_sites += 1
        if percolates(grid, n):
            return open_sites / (n * n)  

def estimate_percolation_threshold(n, trials):
    thresholds = []
    for _ in range(trials):
        threshold = monte_carlo_simulation(n)
        thresholds.append(threshold)
    return np.mean(thresholds)


n = 20  
trials = 10 


estimated_threshold = estimate_percolation_threshold(n, trials)
print(f"mean threshold:{estimated_threshold}")


# thresholds = [monte_carlo_simulation(n) for _ in range(trials)]
# plt.hist(thresholds, bins=20, density=True)
# plt.xlabel('Percolation Threshold')
# plt.ylabel('Frequency')
# plt.title('Percolation Threshold Estimation')
# plt.show()
