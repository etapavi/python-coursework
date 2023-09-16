import numpy as np

# Simulation parameters
N = 8  # Number of particles
L = 10.0  # Box length
rho = 0.8  # Density
T = 1.0  # Temperature
dt = 0.005  # Time step
nsteps = 10000  # Number of simulation steps

# Morse potential and force
def Morse(r):
    D = 1.0
    a = 1.0
    r_eq = 1.0
    exp_ar = np.exp(-a*(r-r_eq))
    pot = D*(1-exp_ar)**2 - D
    force = 2*a*D*exp_ar*(1-exp_ar)
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
                    e, fij = Morse(np.sqrt(r2))
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
                    e, fij = Morse(np.sqrt(r2))
                    pot_new += e
                    f_new[i] += fij * rij
                    f_new[j] -= fij * rij

        # Update velocities
        v += 0.5 * (f_total + f_new) / N * dt
        T_new = np.sum(v**2) / (3 * N)
        # Output
        if step % 100 == 0:
                  print(f"Step {step}, Potential energy: {pot:.3f}")


       
        # Write current positions to file in xyzzy format
        f.write(f"{N}\n")
        f.write(f"Step {step}\n")
        for i in range(N):
            x, y, z = r[i]
            f.write(f"C {x:.4f} {y:.4f} {z:.4f} 0.0 0.0 0.0\n")

