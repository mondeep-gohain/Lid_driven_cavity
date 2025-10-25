import numpy as np #You need to install numpy to run this code. You can install it by running the command "pip install numpy" in the terminal.
import matplotlib.pyplot as plt #You need to install matplotlib to run this code. You can install it by running the command "pip install matplotlib" in the terminal.
import time

#adding the progress bar to keep the user updated on the progress of the simulation
def print_progress(iteration, error, grid_size, start_time):
    elapsed_time = time.time() - start_time
    print(f"Grid {grid_size}: Iteration {iteration}, Error = {error:.2e}, Time elapsed: {elapsed_time:.2f}s")

# Parameters for Lid Driven Cavity Flow
Re = 100                  # Reynolds number (lid-driven cavity flow) Re = U * L / nu where U is the lid velocity, L is the length of the cavity and nu is the kinematic viscosity
nu = 1 / Re               # Kinematic viscosity (m^2/s)
dom_length = 1            # Domain length (square domain). This is the length of the cavity 
grid_sizes = [21, 51, 81] # Grid sizes for grid independence study. You can change the grid sizes to see how the results change with different resolutions. Although, in the plots attached in the report,we have used 21, 51, and 81 grid sizes.
alpha = 0.8               # We have used the under-relaxation factor to stabilize the solution process. This factor is used to update the velocity field at each iteration.
alpha_p = 0.8             # We have used the under-relaxation factor to stabilize the solution process. This factor is used to update the pressure field at each iteration.
error_req = 1e-7          # Convergence tolerance
max_iterations = 5000     # Maximum iterations for the solver

print(f"Running simulation with:") #We have printed the simulation parameters to the console to keep track of the simulation progress.
print(f"Reynolds number: {Re}") #We have printed the Reynolds number to the console to keep track of the simulation progress.
print(f"Grid sizes: {grid_sizes}") #We have printed the grid sizes to the console to keep track of the simulation progress.
print(f"Convergence tolerance: {error_req}") #We have printed the convergence tolerance to the console to keep track of the simulation progress.
print("----------------------------------------")

# Store results for grid independence study
results = []

print("Starting simulation for Lid Driven Cavity Flow...")

# Loop over grid sizes
#This loop iterates over the different grid sizes specified in the grid_sizes list. 
# For each grid size, the simulation is run using the SIMPLE algorithm to solve the Navier-Stokes equations for the lid-driven cavity flow.
for grid_index, n_points in enumerate(grid_sizes):
    print(f"\nProcessing grid size: {n_points} x {n_points}")
    start_time = time.time()

    # Grid setup
    print("Setting up grid...")
    h = dom_length / (n_points - 1)           # Grid spacing (m).This is the distance between grid points in the x and y directions and is calculated based on the domain length and the number of grid points.
    x = np.linspace(0, dom_length, n_points)  # x-coordinates of grid points
    y = np.linspace(0, dom_length, n_points)  # y-coordinates of grid points

    # Initialize variables
    print("Initializing variables...")
    u = np.zeros((n_points + 1, n_points))         # Horizontal velocity (m/s) 
    v = np.zeros((n_points, n_points + 1))         # Vertical velocity (m/s)
    p = np.ones((n_points + 1, n_points + 1))      # Pressure field (Pa)
    pc = np.zeros_like(p)                          # Pressure correction. This is used to correct the pressure field at each iteration.
    u_star = np.zeros_like(u)                      # Intermediate x-velocity field
    v_star = np.zeros_like(v)                      # Intermediate y-velocity field
    d_e = np.zeros_like(u)                         # Coefficients for x-momentum equation for east face
    d_n = np.zeros_like(v)                         # Coefficients for y-momentum equation for north face
    b = np.zeros_like(p)                           # Source term for momentum equations

    u_final = np.zeros((n_points, n_points))       # Final x-velocity field
    v_final = np.zeros((n_points, n_points))       # Final y-velocity field
    p_final = np.zeros((n_points, n_points))       # Final pressure field

    # Boundary conditions
    print("Applying boundary conditions...")
    u[0, :] = 1  # Lid velocity (top wall moving to the right)

    print("Starting SIMPLE algorithm iterations...")
    # Solver Loop
    error = 1
    iterations = 0
    last_print_time = time.time()

    while error > error_req and iterations < max_iterations:
        iterations += 1

         # Print progress every 100 iterations or if error changes significantly
        current_time = time.time()
        if iterations % 100 == 0 or current_time - last_print_time >= 5:
            print_progress(iterations, error, n_points, start_time)
            last_print_time = current_time

        # Solving x-momentum equation
        for i in range(1, n_points):
            for j in range(1, n_points - 1):
                u_E = 0.5 * (u[i, j] + u[i, j + 1])
                u_W = 0.5 * (u[i, j] + u[i, j - 1])
                v_N = 0.5 * (v[i - 1, j] + v[i - 1, j + 1])
                v_S = 0.5 * (v[i, j] + v[i, j + 1])

                a_E = -0.5 * u_E * h + nu
                a_W = 0.5 * u_W * h + nu
                a_N = -0.5 * v_N * h + nu
                a_S = 0.5 * v_S * h + nu
                a_P = a_E + a_W + a_N + a_S

                d_e[i, j] = h / a_P
                u_star[i, j] = (a_E * u[i, j + 1] + a_W * u[i, j - 1] +
                                a_N * u[i - 1, j] + a_S * u[i + 1, j]) / a_P - \
                                d_e[i, j] * (p[i, j + 1] - p[i, j])

        # Solving y-momentum equation
        for i in range(1, n_points - 1):
            for j in range(1, n_points):
                u_E = 0.5 * (u[i, j] + u[i + 1, j])
                u_W = 0.5 * (u[i, j - 1] + u[i + 1, j - 1])
                v_N = 0.5 * (v[i - 1, j] + v[i, j])
                v_S = 0.5 * (v[i, j] + v[i + 1, j])

                a_E = -0.5 * u_E * h + nu
                a_W = 0.5 * u_W * h + nu
                a_N = -0.5 * v_N * h + nu
                a_S = 0.5 * v_S * h + nu
                a_P = a_E + a_W + a_N + a_S

                d_n[i, j] = h / a_P
                v_star[i, j] = (a_E * v[i, j + 1] + a_W * v[i, j - 1] +
                                a_N * v[i - 1, j] + a_S * v[i + 1, j]) / a_P - \
                                d_n[i, j] * (p[i, j] - p[i + 1, j])

        # Computing the pressure correction term
        for i in range(1, n_points):
            for j in range(1, n_points):
                b[i, j] = -(u_star[i, j] - u_star[i, j - 1]) * h + \
                          (v_star[i, j] - v_star[i - 1, j]) * h
                pc[i, j] = alpha_p * b[i, j]

        # Updating the pressure and velocity fields
        p += pc
        for i in range(1, n_points):
            for j in range(1, n_points):
                u[i, j] = u_star[i, j] - d_e[i, j] * (pc[i, j + 1] - pc[i, j])
                v[i, j] = v_star[i, j] - d_n[i, j] * (pc[i, j] - pc[i + 1, j])

        # Compute residual
        error = np.max(np.abs(b))
    
    end_time = time.time()
    print(f"\nGrid {n_points}x{n_points} completed:")
    print(f"Total iterations: {iterations}")
    print(f"Final error: {error:.2e}")
    print(f"Total computation time: {end_time - start_time:.2f} seconds")
    print("----------------------------------------")

    # Mapping results to collocated grid
    print("Mapping results to collocated grid...")
    # Interpolate to cell centers for visualization
    for i in range(n_points):
        for j in range(n_points):
            u_final[i, j] = 0.5 * (u[i, j] + u[i + 1, j])
            v_final[i, j] = 0.5 * (v[i, j] + v[i, j + 1])
            p_final[i, j] = 0.25 * (p[i, j] + p[i + 1, j] + p[i, j + 1] + p[i + 1, j + 1])

    # Store results for grid independence study
    results.append({
        "u_final": u_final,
        "v_final": v_final,
        "p_final": p_final,
        "x": x,
        "y": y,
        "n_points": n_points
    })

print("\nStarting visualization...")

#Visualization of the finest grid as it provides the most accurate results.
#In the visualization section, we will create contour plots, centerline plots, and streamline plots to visualize the results of the simulation.
#In the question, we are asked to visualize the results for the finest grid size. Therefore, we will focus on the finest grid size in our visualization.
#We have shown the results for all grid sizes in the grid independence study section.
#Also, the streamline plots are created for all grid sizes to visualize the flow patterns in the domain as asked in the question.

# Visualizations for the finest grid
finest_grid = results[-1]
print("\nGenerating plots")

u_final = finest_grid["u_final"]
v_final = finest_grid["v_final"]
p_final = finest_grid["p_final"]
x = finest_grid["x"]
y = finest_grid["y"]

X, Y = np.meshgrid(x, y)

# Contour plots for the finest grid
plt.figure()
plt.contourf(X, Y, u_final.T, levels=20, cmap="viridis")
plt.title("X-Velocity Contour of the finest grid")
plt.colorbar()

plt.figure()
plt.contourf(X, Y, v_final.T, levels=20, cmap="viridis")
plt.title("Y-Velocity Contour of the finest grid")
plt.colorbar()

plt.figure()
plt.contourf(X, Y, p_final.T, levels=20, cmap="viridis")
plt.title("Pressure Contour of the finest grid")
plt.colorbar()

# Centerline plots for the finest grid as asked in the question
plt.figure()
plt.plot(y, u_final[len(x) // 2, :], '-o', label='u (y = 0.5)')
plt.plot(x, v_final[:, len(y) // 2], '-s', label='v (x = 0.5)')
plt.xlabel('Position')
plt.ylabel('Velocity')
plt.legend()
plt.grid(True)
plt.title('Centerline Velocities of the Finest Grid')

plt.figure()
plt.plot(y, p_final[len(x) // 2, :], '-o', label='p (y = 0.5)')
plt.plot(x, p_final[:, len(y) // 2], '-s', label='p (x = 0.5)')
plt.xlabel('Position')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)
plt.title('Centerline Pressure of the Finest Grid')

# Streamline plots for all grid sizes as asked in the question
for result in results:
    x, y, u_final, v_final = result["x"], result["y"], result["u_final"], result["v_final"]
    X, Y = np.meshgrid(x, y)

    plt.figure()
    plt.streamplot(X, Y, u_final.T, v_final.T, density=1)
    plt.title(f"Streamlines for Grid Size {result['n_points']}")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')

plt.show()

# End of script
print("\nSimulation completed successfully!")
print(f"\nTotal Runtime: {time.time() - start_time:.2f} seconds")
