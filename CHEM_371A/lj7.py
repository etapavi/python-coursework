import numpy as np

# Simulation parameters
N = 64  # Number of particles
L = 20.0  # Box length
rho = 0.8  # Density
T = 1.0  # Temperature
dt = 0.005  # Time step
nsteps = 10000  # Number of simulation steps

# LJ potential and force
def LJ(r):
    sigma = 1.0
    epsilon = 1.0
    sr6 = (sigma / r) ** 6
    pot = 4 * epsilon * (sr6 ** 2 - sr6)
    force = 24 * epsilon / r * (2 * sr6 ** 2 - sr6)
    return pot, force

# Initialize particle positions and velocities
r = np.zeros((N, 3))
v = np.random.normal(0, np.sqrt(T), (N, 3))
v -= np.mean(v, axis=0)
r[:, 0] = np.random.uniform(0, L, N)
r[:, 1] = np.random.uniform(0, L, N)
r[:, 2] = np.random.uniform(0, L, N)

# Main MD loop
with open('trajectory.xyz', 'w') as f:
    for step in range(nsteps):
        # Compute forces and potential energy
        f_total = np.zeros((N, 3))
        pot = 0
        for i in range(N):
            for j in range(i+1, N):
                rij = r[i] - r[j]
                rij -= L * np.round(rij / L)
                r2 = np.dot(rij, rij)
                if r2 < 2.5**2:
                    e, fij = LJ(np.sqrt(r2))
                    pot += e
                    f_total[i] += fij * rij
                    f_total[j] -= fij * rij

        # Update positions
        r += v * dt + 0.5 * f_total / N * dt**2
        r = np.mod(r, L)

        # Compute forces with updated positions
        f_new = np.zeros((N, 3))
        pot_new = 0
        for i in range(N):
            for j in range(i+1, N):
                rij = r[i] - r[j]
                rij -= L * np.round(rij / L)
                r2 = np.dot(rij, rij)
                if r2 < 2.5**2:
                    e, fij = LJ(np.sqrt(r2))
                    pot_new += e
                    f_new[i] += fij * rij
                    f_new[j] -= fij * rij

        # Update velocities
        v += 0.5 * (f_total + f_new) / N * dt
        T_new = np.sum(v**2) / (3 * N)
        v *= np.sqrt(T / T_new)
        v -= np.mean(v, axis=0)
        # Output
        if step % 100 == 0:
                  print(f"Step {step}, Potential energy: {pot:.3f}")

        # Write current positions to file in xyzzy format
        f.write(f"{N}\n")
        f.write(f"Step {step}\n")
        for i in range(N):
            x, y, z = r[i]
            f.write(f"C {x:.4f} {y:.4f} {z:.4f} 0.0 0.0 0.0\n")

