import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class IsingModel:
    def __init__(self, n, beta):
        self.n = n  
        self.beta = beta  
        self.lattice = np.random.choice([-1, 1], size=(n, n))  

    def energy_change(self, i, j):
        
        neighbors = (
            self.lattice[(i - 1) % self.n, j]
            + self.lattice[(i + 1) % self.n, j]
            + self.lattice[i, (j - 1) % self.n]
            + self.lattice[i, (j + 1) % self.n]
        )
        delta_E = 2 * self.lattice[i, j] * neighbors
        return delta_E

    def metropolis_step(self):
        
        i = np.random.randint(0, self.n)
        j = np.random.randint(0, self.n)
        delta_E = self.energy_change(i, j)
        if delta_E < 0 or np.random.rand() < np.exp(-self.beta * delta_E):
            self.lattice[i, j] *= -1

    def simulate(self, steps):
        
        for _ in tqdm(range(steps)):
            self.metropolis_step()


def run_ising_simulation(n, beta, steps):
    
    model = IsingModel(n, beta)
    model.simulate(steps)
    return model.lattice



n = 100  
betas = [-1, 0, 0.441, 0.8]  
steps = 10000000  
threads = 4  


results = []  
with ThreadPoolExecutor(max_workers=threads) as executor:
    futures = {executor.submit(run_ising_simulation, n, beta, steps): beta for beta in betas}
    for future in tqdm(as_completed(futures), total=len(betas), desc="Simulating Ising Model"):
        beta = futures[future]
        results.append((beta, future.result()))


results.sort(key=lambda x: betas.index(x[0]))  
for beta, lattice in results:
    plt.figure(figsize=(6, 6))
    plt.title(f"Ising Model (n={n}, β={beta})")
    plt.imshow(lattice, cmap="gray")
    plt.colorbar(label="Spin")
    plt.show()
    
    