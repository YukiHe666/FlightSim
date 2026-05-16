import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

TIMESTEP_ms = 1
TIMESTEP_s = TIMESTEP_ms / 1000

GRAVITY_MPS2 = -9.81


# ------------------------------------------------------------
# Quaternion math
# Quaternion format: q = [w, x, y, z]
# ------------------------------------------------------------

def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_multiply(q1, q2):
    """
    Hamilton product q1 ⊗ q2.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dtype=float)


def quat_to_rotation_matrix(q):
    """
    Converts body-to-world quaternion into rotation matrix R.

    v_world = R @ v_body
    """
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),         2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),         1 - 2*(x*x + y*y)]
    ], dtype=float)


def rotation_matrix_to_euler_deg(R):
    """
    Returns roll, pitch, yaw in degrees.

    This is only for plotting/debugging.
    The actual simulation uses quaternions.
    """
    pitch = np.arcsin(-R[2, 0])
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# ------------------------------------------------------------
# Mass point
# ------------------------------------------------------------

class MassPoint:
    def __init__(self, mass_kg, local_pos_m):
        self.mass_kg = float(mass_kg)

        # Position in body frame before COM correction
        self.local_pos_m = np.array(local_pos_m, dtype=float)


# ------------------------------------------------------------
# Rigid body made from multiple point masses
# ------------------------------------------------------------

class RigidBody:
    def __init__(
        self,
        points,
        pos_world_m=(0, 0, 0),
        vel_world_mps=(0, 0, 0),
        q_body_to_world=(1, 0, 0, 0),
        omega_body_radps=(0, 0, 0)
    ):
        self.points = points

        self.pos_world_m = np.array(pos_world_m, dtype=float)
        self.vel_world_mps = np.array(vel_world_mps, dtype=float)

        self.q_body_to_world = quat_normalize(
            np.array(q_body_to_world, dtype=float)
        )

        # Angular velocity expressed in body frame
        self.omega_body_radps = np.array(omega_body_radps, dtype=float)

        self.mass_kg = sum(p.mass_kg for p in self.points)

        self.shift_origin_to_center_of_mass()

        self.I_body = self.compute_body_inertia_tensor()
        self.I_body_inv = np.linalg.inv(self.I_body)

    def shift_origin_to_center_of_mass(self):
        """
        Shift all local point positions so the rigid body's body-frame origin
        is exactly at the center of mass.
        """
        com = sum(
            p.mass_kg * p.local_pos_m for p in self.points
        ) / self.mass_kg

        for p in self.points:
            p.local_pos_m = p.local_pos_m - com

    def compute_body_inertia_tensor(self):
        """
        Moment of inertia tensor for point masses:

            I = Σ m (|r|² I₃ - r rᵀ)

        where r is the point position relative to the COM.
        """
        I = np.zeros((3, 3), dtype=float)

        for p in self.points:
            r = p.local_pos_m
            r_squared = np.dot(r, r)

            I += p.mass_kg * (
                r_squared * np.eye(3) - np.outer(r, r)
            )

        return I

    def rotation_matrix(self):
        return quat_to_rotation_matrix(self.q_body_to_world)

    def point_world_offset(self, point):
        """
        World-frame offset from COM to this point.
        """
        R = self.rotation_matrix()
        return R @ point.local_pos_m

    def point_world_position(self, point):
        return self.pos_world_m + self.point_world_offset(point)

    def point_world_velocity(self, point):
        """
        Velocity of a point on a rigid body:

            v_point = v_COM + ω × r

        omega is converted from body frame to world frame first.
        """
        R = self.rotation_matrix()

        omega_world = R @ self.omega_body_radps
        r_world = self.point_world_offset(point)

        return self.vel_world_mps + np.cross(omega_world, r_world)


# ------------------------------------------------------------
# Real physics force model: gravity only
# ------------------------------------------------------------

def gravity_force_on_point(point):
    """
    Uniform gravity:

        F_g = m g

    Since gravity is uniform, it acts through the center of mass overall.
    Therefore it does not create net torque about the COM.
    """
    return np.array([
        0,
        0,
        point.mass_kg * GRAVITY_MPS2
    ], dtype=float)


# ------------------------------------------------------------
# Rigid body time step
# ------------------------------------------------------------

def step_rigidbody(body, dt_s):
    R = body.rotation_matrix()

    total_force_world = np.zeros(3)
    total_torque_world = np.zeros(3)

    for p in body.points:
        r_world = body.point_world_offset(p)
        F_world = gravity_force_on_point(p)

        total_force_world += F_world

        # Torque about COM:
        #
        #     τ = r × F
        #
        total_torque_world += np.cross(r_world, F_world)

    # --------------------------------------------------------
    # Linear dynamics
    #
    #     F = ma
    # --------------------------------------------------------

    accel_world_mps2 = total_force_world / body.mass_kg

    body.vel_world_mps += accel_world_mps2 * dt_s
    body.pos_world_m += body.vel_world_mps * dt_s

    # --------------------------------------------------------
    # Rotational dynamics
    #
    # Euler's rigid-body equation in body frame:
    #
    #     τ = Iω_dot + ω × Iω
    #
    # so:
    #
    #     ω_dot = I⁻¹ [τ - ω × Iω]
    # --------------------------------------------------------

    torque_body = R.T @ total_torque_world

    omega = body.omega_body_radps
    I = body.I_body

    omega_dot = body.I_body_inv @ (
        torque_body - np.cross(omega, I @ omega)
    )

    body.omega_body_radps += omega_dot * dt_s

    # --------------------------------------------------------
    # Quaternion orientation update
    #
    # For body-frame angular velocity:
    #
    #     q_dot = 1/2 q ⊗ [0, ωx, ωy, ωz]
    # --------------------------------------------------------

    omega_quat = np.array([
        0,
        body.omega_body_radps[0],
        body.omega_body_radps[1],
        body.omega_body_radps[2]
    ], dtype=float)

    q_dot = 0.5 * quat_multiply(
        body.q_body_to_world,
        omega_quat
    )

    body.q_body_to_world += q_dot * dt_s
    body.q_body_to_world = quat_normalize(body.q_body_to_world)


# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

def simulate(body, simulation_time_s):
    time_list = []
    z_list = []

    roll_list = []
    pitch_list = []
    yaw_list = []

    x_nose_list = []
    z_nose_list = []

    num_steps = int(simulation_time_s / TIMESTEP_s)

    # Choose the nose point for visualization
    nose_point = max(body.points, key=lambda p: p.local_pos_m[0])

    for i in range(num_steps):
        t = i * TIMESTEP_s

        R = body.rotation_matrix()
        roll, pitch, yaw = rotation_matrix_to_euler_deg(R)

        nose_world = body.point_world_position(nose_point)

        time_list.append(t)
        z_list.append(body.pos_world_m[2])

        roll_list.append(roll)
        pitch_list.append(pitch)
        yaw_list.append(yaw)

        x_nose_list.append(nose_world[0])
        z_nose_list.append(nose_world[2])

        step_rigidbody(body, TIMESTEP_s)

        if body.pos_world_m[2] <= 0:
            body.pos_world_m[2] = 0
            break

    return {
        "time_s": np.array(time_list),
        "z_m": np.array(z_list),
        "roll_deg": np.array(roll_list),
        "pitch_deg": np.array(pitch_list),
        "yaw_deg": np.array(yaw_list),
        "nose_x_m": np.array(x_nose_list),
        "nose_z_m": np.array(z_nose_list),
    }


# ------------------------------------------------------------
# Example rigid plane-shaped mass structure
# ------------------------------------------------------------

def make_plane_mass_structure():
    """
    This is only a rigid mass distribution.

    It is shaped like a simple airplane, but it has no aerodynamic lift,
    no thrust, no control surfaces, and no fake behavior.

    Body frame convention:

        x = forward
        y = right
        z = up
    """

    points = [
        # fuselage / center
        MassPoint(0.80, (0.00,  0.00, 0.00)),

        # nose
        MassPoint(0.20, (0.80,  0.00, 0.00)),

        # tail
        MassPoint(0.15, (-0.90, 0.00, 0.05)),

        # left and right wings
        MassPoint(0.12, (0.00, -0.70, 0.00)),
        MassPoint(0.12, (0.00,  0.70, 0.00)),

        # vertical tail mass, slightly above rear body
        MassPoint(0.06, (-0.80, 0.00, 0.25)),
    ]

    body = RigidBody(
        points=points,

        # Initial COM position
        pos_world_m=(0, 0, 20),

        # Initial COM velocity
        vel_world_mps=(5, 0, 0),

        # Initial orientation
        q_body_to_world=(1, 0, 0, 0),

        # Initial body angular velocity.
        # This makes the rigid body tumble while falling.
        # changed to not do that
        omega_body_radps=(0,0,0)
    )

    return body


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

body = make_plane_mass_structure()

print("Total mass:", body.mass_kg, "kg")
print("Body-frame inertia tensor:")
print(body.I_body)

result = simulate(body, simulation_time_s=5)


# ------------------------------------------------------------
# Plot center of mass height
# ------------------------------------------------------------

plt.plot(result["time_s"], result["z_m"])

plt.xlabel("Time (s)")
plt.ylabel("Center of mass height z (m)")
plt.title("Rigid Body Falling Under Gravity")
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Plot attitude
# ------------------------------------------------------------

plt.plot(result["time_s"], result["roll_deg"], label="Roll")
plt.plot(result["time_s"], result["pitch_deg"], label="Pitch")
plt.plot(result["time_s"], result["yaw_deg"], label="Yaw")

plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.title("Rigid Body Attitude While Falling")
plt.legend()
plt.grid(True)
plt.show()


# ------------------------------------------------------------
# Plot nose path
# ------------------------------------------------------------

plt.plot(result["nose_x_m"], result["nose_z_m"])

plt.xlabel("Nose x position (m)")
plt.ylabel("Nose z position (m)")
plt.title("Nose Point Path During Rigid-Body Falling Motion")
plt.grid(True)
plt.show()


print("Final COM position:", body.pos_world_m)
print("Final COM velocity:", body.vel_world_mps)
print("Final angular velocity in body frame:", body.omega_body_radps)
print("Final quaternion body-to-world:", body.q_body_to_world)