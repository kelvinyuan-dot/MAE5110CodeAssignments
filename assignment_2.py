from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from models import inverted_pendulum_walker as model
from integrators import rk4

# Fixed controls for this visualization example.
params = {
    "gravity": 9.81,  # m/s^2
    "length": 1.0,  # m
    "mass": 1.0,  # kg
    "incline": 0.06,  # rad
    "angle_of_attack": np.pi / 8,  # rad
    "ankle_torque": 0.0,  # N m
}

def ankle_controller(state, params, kp, kd):
    theta, angular_velocity = state

    gravity = params["gravity"]
    mass = params["mass"]
    length = params["length"]

    torque = (-mass * gravity * length * np.sin(theta) -mass * length**2 * (kp * theta + kd * angular_velocity))

    torque_min = -0.1 * mass * gravity * length
    torque_max = 0.05 * mass * gravity * length

    torque = np.clip(torque, torque_min, torque_max)

    return torque

# Controller gains
kp = 5.0
kd = 2.0

def rk4_step(t, state, timestep, model, params):
    time_traj = np.array([t, t + timestep])

    state_traj = np.zeros((2, 2))
    state_traj[:, 0] = state

    state_traj = rk4.integrate(
        time_traj,
        state_traj,
        timestep,
        model,
        params,
    )

    return state_traj[:, 1]

def simulate_one_step(initial_velocity, alpha, params, timestep=1e-3):
    state = np.array([0.0, initial_velocity])

    params["angle_of_attack"] = alpha
    params["ankle_torque"] = 0.0

    impact_occurred = False
    max_steps = 10000

    for step in range(max_steps):
        t = step * timestep

        next_state = rk4_step(
            t,
            state,
            timestep,
            model,
            params,
        )

        # Detect foot strike and apply impact dynamics
        if model.event_guard(state, next_state, params):
            next_state = model.event_dynamics(
                next_state,
                params,
            )
            impact_occurred = True

        # After impact, detect next theta = 0 crossing
        elif (
            impact_occurred
            and state[0] < 0 <= next_state[0]
            and next_state[1] > 0
        ):
            return next_state[1]

        state = next_state

    # Did not successfully return to the Poincare section
    return np.nan

def compute_return_map(velocity_grid, alpha_grid, params):
    table = np.full(
        (len(velocity_grid), len(alpha_grid)),
        np.nan,
    )

    for i, angular_velocity in enumerate(velocity_grid):
        for j, alpha in enumerate(alpha_grid):
            table[i, j] = simulate_one_step(
                angular_velocity,
                alpha,
                params,
            )

    return table
#Testing one-step Poincare map; removed after testing was succesful
#test_velocity = 2.0
#test_alpha = np.pi / 8

#next_velocity = simulate_one_step(test_velocity, test_alpha, params,)

#print("Initial velocity:", test_velocity)
#print("Alpha:", test_alpha)
#print("Next velocity:", next_velocity)
##Results:
#Initial velocity: 2.0
#Alpha: 0.39269908169872414
#Next velocity: 1.3834668067010358

def test_stabilization(initial_state, params, kp, kd):
    state = initial_state.copy()

    timestep = 1e-4
    test_time = 3.0
    n_steps = int(test_time / timestep)

    for _ in range(n_steps):

        # Turn the ankle controller on for the stabilization test
        params["ankle_torque"] = ankle_controller(
            state, params, kp, kd
        )

        # rk4 integration
        state = rk4_step(0.0, state, timestep, model, params)

        # If the pendulum moves too far from upright,
        # consider this initial condition a failure.
        if abs(state[0]) > np.pi / 2:
            params["ankle_torque"] = 0.0
            return False

    # Check whether the pendulum finished close to upright
    stabilized = (
        abs(state[0]) < 0.01
        and abs(state[1]) < 0.05
    )

    # Reset torque after the test
    params["ankle_torque"] = 0.0

    return stabilized

# Grid of initial states around the upright equilibrium
theta_values = np.linspace(-np.pi / 7, np.pi / 7, 21)
velocity_values = np.linspace(-1.0, 1.0, 21)

# True = ankle controller can stabilize this initial condition
roa = np.zeros(
    (len(velocity_values), len(theta_values)),
    dtype=bool,
)

print("Computing region of attraction...")

for i, angular_velocity in enumerate(velocity_values):
    for j, theta in enumerate(theta_values):

        initial_state = np.array([
            theta,
            angular_velocity,
        ])

        roa[i, j] = test_stabilization(
            initial_state,
            params,
            kp,
            kd,
        )

print("Finished computing region of attraction.")


plt.figure(figsize=(7, 5))

# Red = failure, Green = successfully stabilized
roa_cmap = ListedColormap(["red", "green"])

plt.imshow(
    roa,
    origin="lower",
    extent=[
        theta_values[0],
        theta_values[-1],
        velocity_values[0],
        velocity_values[-1],
    ],
    aspect="auto",
    cmap=roa_cmap,
)

plt.xlabel("Theta (rad)")
plt.ylabel("Angular velocity (rad/s)")
plt.title("Region of Attraction of Ankle Controller")

plt.show()


def roa_event_guard(state, theta_values, velocity_values, roa):
    theta, angular_velocity = state

    # State must be inside the bounds of the grid
    if (
        theta < theta_values[0]
        or theta > theta_values[-1]
        or angular_velocity < velocity_values[0]
        or angular_velocity > velocity_values[-1]
    ):
        return False

    # Find the nearest point in the RoA grid
    theta_index = np.argmin(
        np.abs(theta_values - theta)
    )

    velocity_index = np.argmin(
        np.abs(velocity_values - angular_velocity)
    )

    return roa[velocity_index, theta_index]

# State-action grid for the step-to-step dynamics

gravity = params["gravity"]
length = params["length"]

# Maximum angular velocity corresponding to Froude number = 2
max_velocity = np.sqrt(2 * gravity / length)

# Grid of current angular velocities
poincare_velocity_values = np.linspace(
    0.0,
    max_velocity,
    21,
)

# Grid of allowable angle-of-attack controls
alpha_values = np.linspace(
    np.pi / 8,
    np.pi / 7,
    21,
)

# Table containing the next angular velocity
return_map = np.full(
    (len(poincare_velocity_values), len(alpha_values)),
    np.nan,
)

print("Computing step-to-step lookup table...")

for i, angular_velocity in enumerate(poincare_velocity_values):
    for j, alpha in enumerate(alpha_values):

        return_map[i, j] = simulate_one_step(
            angular_velocity,
            alpha,
            params,
        )

print("Finished computing step-to-step lookup table.")

# Plot the state-action lookup table
plt.figure(figsize=(8, 5))

plt.imshow(
    return_map,
    origin="lower",
    aspect="auto",
    extent=[
        alpha_values[0],
        alpha_values[-1],
        poincare_velocity_values[0],
        poincare_velocity_values[-1],
    ],
)

plt.colorbar(label="Next angular velocity (rad/s)")
plt.xlabel("Angle of attack alpha (rad)")
plt.ylabel("Current angular velocity (rad/s)")
plt.title("Step-to-Step State-Action Map")

plt.show()

# Test lookup-table grid resolution using nearest-neighbor error

grid_sizes = [11, 21, 31, 41]
mean_errors = []

for grid_size in grid_sizes:

    velocity_grid = np.linspace(
        0.0,
        max_velocity,
        grid_size,
    )

    alpha_grid = np.linspace(
        np.pi / 8,
        np.pi / 7,
        grid_size,
    )

    test_map = compute_return_map(
        velocity_grid,
        alpha_grid,
        params,
    )

    errors = []

    # Test velocities halfway between stored grid points
    for i in range(len(velocity_grid) - 1):

        test_velocity = (
            velocity_grid[i] + velocity_grid[i + 1]
        ) / 2

        # Find closest stored velocity
        nearest_i = np.argmin(
            np.abs(velocity_grid - test_velocity)
        )

        for j, alpha in enumerate(alpha_grid):

            # Prediction from lookup table
            predicted_velocity = test_map[
                nearest_i,
                j,
            ]

            # Actual simulation at the state between grid points
            actual_velocity = simulate_one_step(
                test_velocity,
                alpha,
                params,
            )

            if (
                not np.isnan(predicted_velocity)
                and not np.isnan(actual_velocity)
            ):
                errors.append(
                    abs(
                        actual_velocity
                        - predicted_velocity
                    )
                )

    mean_error = np.mean(errors)
    mean_errors.append(mean_error)

    print(
        f"Grid {grid_size}x{grid_size}: "
        f"mean nearest-neighbor error = "
        f"{mean_error:.6f} rad/s"
    )


plt.figure(figsize=(7, 5))

plt.plot(
    grid_sizes,
    mean_errors,
    "o-",
)

plt.axhline(
    0.01,
    linestyle="--",
    label="0.01 rad/s tolerance",
)

plt.xlabel("Grid size")
plt.ylabel("Mean nearest-neighbor error (rad/s)")
plt.title("Lookup Table Grid Convergence")
plt.legend()
plt.grid()

plt.show()

initial_state = np.array([0.0, 3.0])
timestep = 1e-4
sim_time = 3.0
desired_number_of_steps = 3

n_timesteps = round(sim_time / timestep) + 1
time_traj = np.arange(n_timesteps) * timestep
state_traj = np.zeros((2, n_timesteps))
state_traj[:, 0] = initial_state
completed_steps = 0

# Simulation loop. Replaced with RK4 integration.
reached_roa = False

for step, t in enumerate(time_traj[:-1]):
    state = state_traj[:, step]

    # Check if the current state has entered the ankle controller RoA
    if roa_event_guard(
        state,
        theta_values,
        velocity_values,
        roa,
    ):
        print("Reached ankle controller RoA!")
        reached_roa = True
        break

    # Outside the RoA, leave the ankle controller off
    params["ankle_torque"] = 0.0

    # Advance one timestep using RK4
    next_state = rk4_step(
        t,
        state,
        timestep,
        model,
        params,
    )

    # Check for foot strike
    if model.event_guard(state, next_state, params):
        next_state = model.event_dynamics(
            next_state,
            params,
        )
        completed_steps += 1

    state_traj[:, step + 1] = next_state

    if completed_steps == desired_number_of_steps:
        break

time_traj = time_traj[: step + 2]
state_traj = state_traj[:, : step + 2]

fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")


def draw_frame(index):
    # The massless swing leg is repositioned instantaneously at each impact.
    model.visualize(state_traj[:, index], params, ax=ax)
    ax.set_title(f"t = {time_traj[index]:.2f} s")


# Simulate at a small timestep, but render only 25 frames per second.
fps = 25
frame_stride = round(1 / (fps * timestep))
frame_indices = list(range(0, time_traj.size, frame_stride))
if frame_indices[-1] != time_traj.size - 1:
    frame_indices.append(time_traj.size - 1)

animation = FuncAnimation(
    fig, draw_frame, frames=frame_indices, interval=1000 / fps, repeat=False
)
output = Path("output/assignment_2")
output.mkdir(parents=True, exist_ok=True)
animation.save(output / "walker.gif", writer=PillowWriter(fps=fps))

# To save an MP4 instead, install FFmpeg and use:
# animation.save(output / "walker.mp4", writer="ffmpeg", fps=fps)
print(f"Saved {output / 'walker.gif'} ({completed_steps} footstrikes).")
plt.show()
