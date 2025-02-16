import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.animation import FuncAnimation

""" 
Step 1: plot satellites over time
Step 2: identify spot of emitter
step 3: compute location

"""


def get_geos(tdoas: pl.DataFrame) -> pl.DataFrame:
    return tdoas


def from_x():
    # Constants
    G = 6.67430e-11  # Gravitational constant in m^3 kg^-1 s^-2
    M_earth = 5.97e24  # Mass of Earth in kg
    R_earth = 6.371e6  # Radius of Earth in meters
    dt = 10  # Time step in seconds

    # Initial conditions for satellite in 3D
    r0 = 7e6  # Initial radius (m), higher than Earth's radius for orbit
    v0 = np.sqrt(
        G * M_earth / r0,
    )  # Initial tangential velocity for circular orbit in x-y plane

    # Initial state vector: [x, y, z, vx, vy, vz]
    state = np.array([r0, 0, 0, 0, v0, 0])  # Circular orbit in x-y plane

    # Function to compute acceleration due to gravity
    def acceleration(state):
        r = np.linalg.norm(state[:3])  # Distance from Earth's center
        a = -(G * M_earth / r**3) * state[:3]  # Acceleration vector towards Earth
        return np.concatenate((np.zeros(3), a))  # [0, 0, 0, ax, ay, az]

    # Runge-Kutta 4th order method for numerical integration
    def rk4_step(state, dt):
        k1 = dt * acceleration(state)
        k2 = dt * acceleration(state + k1 / 2)
        k3 = dt * acceleration(state + k2 / 2)
        k4 = dt * acceleration(state + k3)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    # Simulation
    num_steps = 10000
    trajectory = [state]

    for _ in range(num_steps):
        state = rk4_step(state, dt)
        trajectory.append(state)

    # Convert to numpy array for easier manipulation
    trajectory = np.array(trajectory)

    # Plotting
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Earth representation
    u, v = np.mgrid[0 : 2 * np.pi : 20j, 0 : np.pi : 10j]
    x = R_earth * np.cos(u) * np.sin(v)
    y = R_earth * np.sin(u) * np.sin(v)
    z = R_earth * np.cos(v)
    ax.plot_wireframe(x, y, z, color="blue", alpha=0.3)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-2 * R_earth, 2 * R_earth])
    ax.set_ylim([-2 * R_earth, 2 * R_earth])
    ax.set_zlim([-2 * R_earth, 2 * R_earth])

    # Animation update function
    def update(frame):
        ax.clear()

        # Plot Earth
        ax.plot_wireframe(x, y, z, color="blue", alpha=0.3)

        # Plot orbit path
        ax.plot(
            trajectory[: frame + 1, 0],
            trajectory[: frame + 1, 1],
            trajectory[: frame + 1, 2],
            "r-",
            linewidth=0.5,
        )

        # Plot current position of satellite
        ax.scatter(
            trajectory[frame, 0],
            trajectory[frame, 1],
            trajectory[frame, 2],
            color="red",
            s=50,
        )

        # Set axis limits and labels again since clear() resets them
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xlim([-2 * R_earth, 2 * R_earth])
        ax.set_ylim([-2 * R_earth, 2 * R_earth])
        ax.set_zlim([-2 * R_earth, 2 * R_earth])
        ax.set_title(f"3D Satellite Orbit Simulation - Time Step: {frame}")

    # Create animation
    anim = FuncAnimation(fig, update, frames=num_steps, interval=10, repeat=False)

    plt.show()


if __name__ == "__main__":
    from_x()
