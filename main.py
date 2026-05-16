import matplotlib.pyplot as plt
import numpy as np

TIMESTEP_ms = 1
TIMESTEP_s = TIMESTEP_ms / 1000

GRAVITY_MPS2 = -9.81

# Standard air density at sea level
AIR_DENSITY_KGPM3 = 1.225


class PhysThing:
    def __init__(self, mass_kg, area_m2, drag_coeff,
                 x_m=0, y_m=0, z_m=0,
                 v_x_mps=0, v_y_mps=0, v_z_mps=0):
        self.mass_kg = mass_kg
        self.area_m2 = area_m2
        self.drag_coeff = drag_coeff

        # position in world frame
        self.x_m = x_m
        self.y_m = y_m
        self.z_m = z_m

        # velocity in world frame
        self.v_x_mps = v_x_mps
        self.v_y_mps = v_y_mps
        self.v_z_mps = v_z_mps

    def pos(self):
        return np.array([self.x_m, self.y_m, self.z_m])

    def vel(self):
        return np.array([self.v_x_mps, self.v_y_mps, self.v_z_mps])

    def set_pos(self, pos):
        self.x_m = pos[0]
        self.y_m = pos[1]
        self.z_m = pos[2]

    def set_vel(self, vel):
        self.v_x_mps = vel[0]
        self.v_y_mps = vel[1]
        self.v_z_mps = vel[2]


def get_gravity_force(thing):
    return np.array([
        0,
        0,
        thing.mass_kg * GRAVITY_MPS2
    ])


def get_air_resistance_force(thing):
    """
    Quadratic air resistance:

        F_drag = 0.5 * rho * Cd * A * v^2

    Direction is opposite to velocity.
    """

    velocity = thing.vel()
    speed = np.linalg.norm(velocity)

    if speed == 0:
        return np.array([0, 0, 0])

    drag_magnitude = (
        0.5
        * AIR_DENSITY_KGPM3
        * thing.drag_coeff
        * thing.area_m2
        * speed**2
    )

    velocity_direction = velocity / speed

    drag_force = -drag_magnitude * velocity_direction

    return drag_force


def apply_dynamics(thing, dt_s):
    gravity_force = get_gravity_force(thing)
    drag_force = get_air_resistance_force(thing)

    total_force = gravity_force + drag_force

    acceleration = total_force / thing.mass_kg

    new_vel = thing.vel() + acceleration * dt_s
    new_pos = thing.pos() + new_vel * dt_s

    thing.set_vel(new_vel)
    thing.set_pos(new_pos)


def simulate_drop(thing, simulation_time_s):
    time_list = []
    z_list = []
    v_z_list = []

    num_steps = int(simulation_time_s / TIMESTEP_s)

    for i in range(num_steps):
        time_s = i * TIMESTEP_s

        time_list.append(time_s)
        z_list.append(thing.z_m)
        v_z_list.append(thing.v_z_mps)

        apply_dynamics(thing, TIMESTEP_s)

        if thing.z_m <= 0:
            thing.z_m = 0
            time_list.append(time_s + TIMESTEP_s)
            z_list.append(0)
            v_z_list.append(thing.v_z_mps)
            break

    return time_list, z_list, v_z_list


# -----------------------------
# Stone vs feather
# -----------------------------

drop_height_m = 20
simulation_time_s = 10

stone = PhysThing(
    mass_kg=0.1,        # 100 g stone
    area_m2=0.0005,     # small frontal area
    drag_coeff=0.47,    # sphere-like object
    z_m=drop_height_m
)

feather = PhysThing(
    mass_kg=0.001,      # 1 g feather
    area_m2=0.01,       # large area
    drag_coeff=1.3,     # irregular flat object
    z_m=drop_height_m
)

stone_t, stone_z, stone_vz = simulate_drop(stone, simulation_time_s)
feather_t, feather_z, feather_vz = simulate_drop(feather, simulation_time_s)


# -----------------------------
# Plot height vs time
# -----------------------------

plt.plot(stone_t, stone_z, label="Stone")
plt.plot(feather_t, feather_z, label="Feather")

plt.xlabel("Time (s)")
plt.ylabel("Height z (m)")
plt.title("Stone vs Feather Drop with Air Resistance")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------
# Plot vertical velocity vs time
# -----------------------------

plt.plot(stone_t, stone_vz, label="Stone")
plt.plot(feather_t, feather_vz, label="Feather")

plt.xlabel("Time (s)")
plt.ylabel("Vertical velocity vz (m/s)")
plt.title("Vertical Velocity with Air Resistance")
plt.legend()
plt.grid(True)
plt.show()


print("Stone final time:", stone_t[-1], "s")
print("Feather final time:", feather_t[-1], "s")

print("Stone final velocity:", stone_vz[-1], "m/s")
print("Feather final velocity:", feather_vz[-1], "m/s")