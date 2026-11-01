# %% [markdown]
# # Lab 1: Frames, rotations and transforms 📐
#
# You'll build the spatial math every robot runs, in plain NumPy: 2D rotation matrices,
# moving points between a robot's body frame and the map, 3D rotations from yaw/pitch/roll
# (and back, with gimbal lock detection), quaternions (multiply, rotate, slerp), 4×4
# homogeneous transforms with a cheap inverse, and finally a whole lidar point cloud moved
# onto the map, plus what a 1° calibration error does to it at 50 m.
#
# Setup: NumPy only (matplotlib optional, for one picture). pip install numpy
# Run cell by cell ("Run Cell" above each `# %%`, e.g. in VS Code) or `python3 phase1_frames.py`
# from the labs folder. Fill each TODO until its ✅ checks pass. Solutions: labs/solutions/
# Lessons to reread if stuck: 1.1 (frames), 1.2 (2D rotations), 1.3 (Euler angles, gimbal lock),
# 1.4 (quaternions), 1.5 (homogeneous transforms), 1.6 (tf2 naming: T_a_b)
#
# Conventions (ROS REP 103): x forward, y left, z up; angles in radians; R = Rz(yaw) Ry(pitch) Rx(roll);
# quaternions in this lab are (w, x, y, z), w FIRST (ROS messages put w last: careful at boundaries!).

# %%
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:  # the lab works without matplotlib
    plt = None

rng = np.random.default_rng(0)  # one seeded random generator → reproducible results


def check(cond, msg):
    assert cond, "❌ " + msg
    print("✅", msg)


def close(a, b, tol=1e-6):
    """True when every number in a is within tol of the matching number in b."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a.shape == b.shape and bool(np.all(np.abs(a - b) < tol))


def show():
    """Show the current plot. With no screen (e.g. MPLBACKEND=Agg) just close it quietly."""
    if plt is None:
        return
    if plt.get_backend().lower() == "agg":
        plt.close()
    else:
        plt.show()


# %% TODO 1: the 2D rotation matrix (Lesson 1.2)
def rot2d(theta):
    """2×2 matrix that turns a vector counter-clockwise by theta (radians)."""
    # TODO: the columns are where the axes land: x → (cos θ, sin θ), y → (−sin θ, cos θ).
    #   c, s = np.cos(theta), np.sin(theta)
    #   return np.array([[c, -s], [s, c]])
    raise NotImplementedError("TODO 1: rot2d")


R30 = rot2d(np.radians(30))
check(close(R30 @ [2, 1], [1.2320508, 1.8660254]), "(2, 1) turned by 30° → (1.232, 1.866), as in Lesson 1.2")
check(abs(np.linalg.norm(R30 @ [2, 1]) - np.sqrt(5)) < 1e-9, "rotations keep lengths: still √5")
check(close(R30.T @ R30, np.eye(2)), "Rᵀ R = I, so the transpose undoes a rotation")
check(close(rot2d(np.radians(30)) @ rot2d(np.radians(60)), rot2d(np.radians(90))), "R(30°) R(60°) = R(90°)")


# %% TODO 2: body frame → world frame (Lesson 1.1)
def body_to_world(pose, p_body):
    """pose = (x, y, heading θ) of the robot on the map; p_body = (forward, left). Return map coordinates."""
    x, y, th = pose
    # TODO: rotate into the map's directions, then shift by the robot's position: R(θ) p + t
    #   return rot2d(th) @ np.asarray(p_body, dtype=float) + np.array([x, y])
    raise NotImplementedError("TODO 2: body_to_world")


def world_to_body(pose, p_world):
    """The reverse: undo the shift, then the turn (Rᵀ)."""
    x, y, th = pose
    return rot2d(th).T @ (np.asarray(p_world, dtype=float) - np.array([x, y]))


check(close(body_to_world((10, 5, np.pi / 2), (2, 1)), [9, 7]), "Lesson 1.1's puzzle: 2 ahead, 1 left of (10, 5) facing 90° → (9, 7)")
check(close(world_to_body((10, 5, np.pi / 2), (9, 7)), [2, 1]), "and back again: (9, 7) → (2, 1) in base_link")
check(close(body_to_world((4, 2, -np.pi / 2), (3, 0)), [4, -1]), "facing −90°, 3 m ahead of (4, 2) is (4, −1)")


# %% TODO 3: 3D rotations from yaw, pitch, roll (Lesson 1.3)
def rotx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_zyx(yaw, pitch, roll):
    """R = Rz(yaw) · Ry(pitch) · Rx(roll): yaw first, then pitch and roll about the moving body axes."""
    # TODO: multiply the three elementary rotations in this order (order matters!)
    #   return rotz(yaw) @ roty(pitch) @ rotx(roll)
    raise NotImplementedError("TODO 3: euler_zyx")


R = euler_zyx(np.radians(30), np.radians(-20), 0.0)
check(close(R @ [1, 0, 0], [0.8137977, 0.4698463, 0.3420201]), "yaw 30°, pitch −20°: the nose points at (0.814, 0.470, 0.342)")
check(close(roty(np.radians(90)) @ [1, 0, 0], [0, 0, -1]), "ROS sign gotcha: +90° pitch tips the nose DOWN (y points left)")
A = rotz(np.radians(90)) @ roty(np.radians(-90))
B = roty(np.radians(-90)) @ rotz(np.radians(90))
check(close(A @ [1, 0, 0], [0, 0, 1]) and close(B @ [1, 0, 0], [0, 1, 0]), "order matters: yaw-then-pitch ends nose-up, pitch-then-yaw nose-left")
check(close(R.T @ R, np.eye(3)) and abs(np.linalg.det(R) - 1) < 1e-9, "a real rotation: RᵀR = I and det R = +1")


# %% TODO 4: back to angles, and spotting gimbal lock (Lesson 1.3)
def rot_to_euler(R, eps=1e-6):
    """Return (yaw, pitch, roll, locked) for R = Rz Ry Rx. At pitch ±90° set roll = 0 and report locked = True."""
    sp = np.clip(-R[2, 0], -1.0, 1.0)  # R[2, 0] = −sin(pitch)
    pitch = np.arcsin(sp)
    if abs(sp) > 1 - eps:
        # Gimbal lock: yaw and roll turn about the same axis; only their combination is known.
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        return yaw, pitch, 0.0, True
    # TODO: away from the lock, read yaw and roll off the matrix:
    #   yaw  = np.arctan2(R[1, 0], R[0, 0])
    #   roll = np.arctan2(R[2, 1], R[2, 2])
    #   return yaw, pitch, roll, False
    raise NotImplementedError("TODO 4: rot_to_euler")


angles = np.radians([30, 89, 20])
y, p, r, locked = rot_to_euler(euler_zyx(*angles))
check(close([y, p, r], angles) and not locked, "(30°, 89°, 20°) comes back exactly: no lock yet")
y, p, r, locked = rot_to_euler(euler_zyx(*np.radians([30, 90, 20])))
check(locked and abs(np.degrees(y) - 10) < 1e-6 and r == 0, "at pitch 90°: locked, and only yaw − roll = 10° survives")
check(close(euler_zyx(np.radians(30), np.radians(90), 0), euler_zyx(0, np.radians(90), np.radians(-30))), "(yaw 30, roll 0) and (yaw 0, roll −30) are the same orientation at the lock")
for pitch_deg in [0, 60, 80, 89]:
    print(f"pitch {pitch_deg:2d}°: a 30°/s spin about body z shows up as {30 / np.cos(np.radians(pitch_deg)):6.1f}°/s of Euler yaw")


# %% TODO 5: the Hamilton product (Lesson 1.4)
def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    return np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])


def quat_mul(a, b):
    """a ⊗ b for quaternions (w, x, y, z): 'a, then b about the new body axes', like R_a R_b."""
    aw, av = a[0], np.asarray(a[1:], dtype=float)
    bw, bv = b[0], np.asarray(b[1:], dtype=float)
    # TODO: scalar part  aw·bw − av·bv ;  vector part  aw·bv + bw·av + av × bv
    #   w = aw * bw - av @ bv
    #   v = aw * bv + bw * av + np.cross(av, bv)
    #   return np.concatenate([[w], v])
    raise NotImplementedError("TODO 5: quat_mul")


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


yaw90 = quat_from_axis_angle([0, 0, 1], np.pi / 2)
check(close(yaw90, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), "yaw 90° = (0.707, 0, 0, 0.707)")
check(close(quat_mul(yaw90, yaw90), [0, 0, 0, 1]), "two quarter-turns about z make a half-turn: (0, 0, 0, 1)")
check(close(quat_mul(yaw90, quat_conj(yaw90)), [1, 0, 0, 0]), "q ⊗ q* = identity: the conjugate undoes a unit quaternion")
qa, qb = quat_from_axis_angle([1, 0, 0], 0.7), quat_from_axis_angle([0, 1, 0], 0.4)
check(not close(quat_mul(qa, qb), quat_mul(qb, qa)), "order matters: a ⊗ b ≠ b ⊗ a")


# %% TODO 6: rotate a vector with a quaternion (Lesson 1.4)
def quat_rotate(q, v):
    """v′ = q ⊗ (0, v) ⊗ q*  (the vector part of the result)."""
    # TODO: wrap v as a pure quaternion, sandwich it, keep the last three numbers
    #   return quat_mul(quat_mul(q, np.concatenate([[0.0], v])), quat_conj(q))[1:]
    raise NotImplementedError("TODO 6: quat_rotate")


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


check(close(quat_rotate(yaw90, [1, 0, 0]), [0, 1, 0]), "yaw 90° sends x to y")
cyc = quat_from_axis_angle([1, 1, 1], np.radians(120))
check(close(cyc, [0.5, 0.5, 0.5, 0.5]) and close(quat_rotate(cyc, [1, 0, 0]), [0, 1, 0]), "120° about (1,1,1): q = (0.5, 0.5, 0.5, 0.5), and it cycles x → y → z")
v = rng.normal(size=3)
check(close(quat_rotate(cyc, v), quat_to_matrix(cyc) @ v), "the quaternion and its matrix rotate any vector identically")
check(close(quat_rotate(-cyc, v), quat_rotate(cyc, v)), "double cover: −q rotates exactly like q")


# %% TODO 7: slerp, the shortest arc (Lesson 1.4)
def quat_angle(a, b):
    """Angle (radians) between two orientations; |dot| handles q ≡ −q."""
    return 2 * np.arccos(min(1.0, abs(float(np.dot(a, b)))))


def slerp(a, b, t):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = float(np.dot(a, b))
    if d < 0:  # take the short way round
        b, d = -b, -d
    if d > 0.9995:  # almost equal: blend and renormalize
        q = a + t * (b - a)
        return q / np.linalg.norm(q)
    # TODO: Ω = arccos(d);  return (sin((1 − t)Ω) a + sin(tΩ) b) / sin Ω
    #   om = np.arccos(d)
    #   return (np.sin((1 - t) * om) * a + np.sin(t * om) * b) / np.sin(om)
    raise NotImplementedError("TODO 7: slerp")


q170 = quat_from_axis_angle([0, 0, 1], np.radians(170))
qm170 = quat_from_axis_angle([0, 0, 1], np.radians(-170))
mid = slerp(q170, qm170, 0.5)
mid_yaw = np.degrees(rot_to_euler(quat_to_matrix(mid))[0])
check(abs(abs(mid_yaw) - 180) < 1e-6, "the hook of Lesson 1.4: halfway from 170° to −170° is 180°, not 0°")
check(abs(np.degrees(quat_angle(q170, mid)) - 10) < 1e-6, "and slerp only turned 10° to get there (half of the 20° gap)")
qa, qb = quat_from_axis_angle([0.3, -1, 0.2], 0.4), quat_from_axis_angle([1, 0.5, 0], 2.1)
steps = [quat_angle(slerp(qa, qb, i / 10), slerp(qa, qb, (i + 1) / 10)) for i in range(10)]
check(np.std(steps) < 1e-9 and abs(sum(steps) - quat_angle(qa, qb)) < 1e-9, "slerp turns at a constant rate along the shortest arc")


# %% See it: integrating a gyro without renormalizing makes |q| drift (Lesson 1.4)
omega = np.radians(90) * np.array([0.3, 0.5, 0.81]) / np.linalg.norm([0.3, 0.5, 0.81])  # 90°/s
for hz, renorm in [(50, False), (400, False), (50, True)]:
    q, dt = np.array([1.0, 0, 0, 0]), 1.0 / hz
    for _ in range(10 * hz):  # 10 seconds
        q = q + 0.5 * quat_mul(q, np.concatenate([[0.0], omega])) * dt
        if renorm:
            q = q / np.linalg.norm(q)
    print(f"{hz:4d} Hz, renormalize={renorm!s:5}: |q| − 1 after 10 s = {np.linalg.norm(q) - 1:+.4%}")


# %% TODO 8: 4×4 homogeneous transforms (Lesson 1.5)
def make_T(R, t):
    """[[R, t], [0, 0, 0, 1]]"""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_T(T):
    """Inverse without np.linalg.inv: [[Rᵀ, −Rᵀ t], [0, 1]]."""
    # TODO:
    #   Rt = T[:3, :3].T
    #   return make_T(Rt, -Rt @ T[:3, 3])
    raise NotImplementedError("TODO 8: invert_T")


def apply_T(T, p):
    return (T @ np.append(np.asarray(p, dtype=float), 1.0))[:3]


T_map_base = make_T(rotz(np.pi / 2), [10, 5, 0])  # the car of Lesson 1.5's worked example
T_base_lidar = make_T(np.eye(3), [1.5, 0, 0])
T_map_lidar = T_map_base @ T_base_lidar  # inner names cancel
check(close(T_map_lidar[:3, 3], [10, 6.5, 0]), "chain: the lidar sits at map (10, 6.5)")
check(close(apply_T(T_map_lidar, [20, 0, 0]), [10, 26.5, 0]), "a point 20 m ahead of the lidar is at map (10, 26.5)")
T_rand = make_T(euler_zyx(*rng.uniform(-3, 3, size=3)), rng.normal(size=3) * 5)
check(close(invert_T(T_rand), np.linalg.inv(T_rand)), "[Rᵀ, −Rᵀt] equals the general inverse")
check(close(apply_T(invert_T(T_map_base), [9, 7, 0]), [2, 1, 0]), "the inverse takes map (9, 7) back to base_link (2, 1)")


# %% TODO 9: a whole lidar point cloud onto the map (Lesson 1.5)
def cloud_to_map(points_lidar, T_map_lidar):
    """points_lidar: N×3 array in the lidar frame → N×3 array in the map frame, in one matrix product."""
    # TODO: add a column of ones, multiply every row by Tᵀ (rows are points), drop the ones
    #   hom = np.hstack([points_lidar, np.ones((len(points_lidar), 1))])
    #   return (hom @ T_map_lidar.T)[:, :3]
    raise NotImplementedError("TODO 9: cloud_to_map")


cloud = rng.uniform(-50, 50, size=(100_000, 3))
out = cloud_to_map(cloud, T_map_lidar)
check(out.shape == (100_000, 3), "100,000 points moved in one product")
check(close(out[123], apply_T(T_map_lidar, cloud[123])), "the vectorized version matches the point-by-point one")

# A car-wash bump: the real lidar is turned 1° more than the calibration file says.
T_base_lidar_true = make_T(rotz(np.radians(1)), [1.5, 0, 0])
ped_lidar = np.array([[50.0, 0.0, 0.0]])  # a pedestrian 50 m ahead of the lidar
believed = cloud_to_map(ped_lidar, T_map_base @ T_base_lidar)[0]
actual = cloud_to_map(ped_lidar, T_map_base @ T_base_lidar_true)[0]
miss = np.linalg.norm(believed - actual)
print(f"1° of extrinsic yaw error moves a pedestrian 50 m away by {miss:.2f} m")
check(abs(miss - 50 * 2 * np.sin(np.radians(0.5))) < 1e-9 and 0.85 < miss < 0.9, "about 87 cm: rotation errors grow with distance")

if plt is not None:
    ring = np.stack([30 * np.cos(np.linspace(0, 2 * np.pi, 200)), 30 * np.sin(np.linspace(0, 2 * np.pi, 200)), np.zeros(200)], 1)
    a, b = cloud_to_map(ring, T_map_base @ T_base_lidar), cloud_to_map(ring, T_map_base @ T_base_lidar_true)
    plt.figure(figsize=(5, 5))
    plt.plot(a[:, 0], a[:, 1], label="where the car thinks a 30 m ring of points is")
    plt.plot(b[:, 0], b[:, 1], "--", label="where it really is (1° bump)")
    plt.axis("equal"); plt.legend(fontsize=8); plt.title("1° of extrinsic error")
    show()


# %% [markdown]
# ## 🔬 Try it yourself
#
# 1. Lesson 1.1's quick-fire: write `enu_to_ned(p)` and `yaw_enu_to_ned(yaw)`. Check that ENU (30, 40, 10)
#    → NED (40, 30, −10) and that a ROS yaw of 0° (East) is a PX4 heading of 90°.
# 2. Build a "translate then rotate" pose, R(p + t), and find the t that parks a car at the same spot as
#    R p + t′ (Lesson 1.2's Playground 2). Show that t = Rᵀ t′.
# 3. Average 100 random noisy copies of the same orientation two ways: average the Euler angles, and
#    average the quaternions (flip each to the same side first, then normalize). Which survives yaw near ±180°?
# 4. Recreate Lesson 1.5's calibration puzzle: simulate a car seeing a pole from 16 poses along a
#    figure-eight, then fit the lidar's (x, y, yaw) by minimizing the spread with a coarse grid search.
#    Repeat with poses on a circle around the pole. Why does the circle give a whole family of answers?
