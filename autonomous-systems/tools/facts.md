# Verified facts (shared by all lesson writers)

Each line: the fact, its date, and the primary source it was confirmed on. Add to this file when you verify something other
phases may reuse. Don't copy a fact from here into a lesson if you change its meaning; re-check anything older than a few months.

## Waymo
- Rider-only (no human driver) miles: **271.3 million through June 2026** (Phoenix 92.1M, SF Bay Area 82.4M, Los Angeles 67.1M,
  Austin 21.1M, Atlanta 8.6M). Versus a human benchmark on the same roads: **95% fewer serious-injury-or-worse crashes, 82% fewer
  injury-reported crashes, 82% fewer airbag-deployment crashes**. Source: https://waymo.com/safety/impact/ (checked 2026-11-01).
- 6th-generation Waymo Driver: **fully autonomous (rider-only) operations began on the “Ojai” vehicle, announced Feb 12, 2026**,
  initially for employees and guests; it uses a 17-megapixel imager, fewer than half the cameras of the 5th generation, short-range
  lidars, an imaging radar improved for rain and snow, and external audio receivers; vehicles named: Ojai and Hyundai IONIQ 5.
  Source: https://waymo.com/blog/2026/02/ro-on-6th-gen-waymo-driver (the post gives no exact sensor counts; secondary coverage says
  13 cameras, 4 lidars, 6 radars: don't state those counts as Waymo's own words).
- 6th-gen announcement (Aug 2024): https://waymo.com/blog/2024/08/meet-the-6th-generation-waymo-driver/
- Ridership: TechCrunch (Mar 27, 2026) reported **500,000 paid rides per week across 10 US cities** (up from 50,000/week in May 2024):
  https://techcrunch.com/2026/03/27/waymo-skyrocketing-ridership-in-one-chart/ . Waymo's co-CEO set a goal of 1 million paid rides
  a week by the end of 2026 (Feb 2026 coverage). City counts after that vary by source; verify on waymo.com before stating one.
- Waymo blog posts in Oct 2026 announce Madrid (Oct 20), Toronto (Oct 16), new public cities incl. San Antonio and San Diego (Oct 14),
  Las Vegas (Oct 21) and a Toyota collaboration (Oct 30). Source: https://waymo.com/blog/ (titles checked 2026-11-01).
- Waymo sept 2026 post: “Across Waymo's 270 million autonomous miles…” (blog index, Sep 24, 2026).

## Tesla
- **Radar removed** from Model 3 and Model Y delivered from **May 2021** in the US and Canada (“Transitioning to Tesla Vision”);
  Model S/X and other markets kept radar at the time. Source: https://www.tesla.com/support/transitioning-tesla-vision (page exists;
  details confirmed via contemporary coverage of the page, e.g. electrek 2021-05-25).
- Andrej Karpathy (then Tesla's head of AI) keynote, CVPR 2021 Workshop on Autonomous Driving (June 20, 2021): vision-only depth and
  velocity of other cars. Video: https://www.youtube.com/watch?v=g6bOwQdCJrc (oEmbed title “[CVPR'21 WAD] Keynote - Andrej Karpathy, Tesla”).
- Product name: **“Full Self-Driving (Supervised)”** (https://www.tesla.com/fsd). FSD v14 released Oct 2025; v14.3.x (2026) release
  notes say the model was **unified across Actually Smart Summon, FSD and Robotaxi** (notateslaapp, 2026). Robotaxi service in Austin;
  reports through 2026 conflict about how “unsupervised” rides are: describe cautiously and date it.

## Drones / flight stacks
- **PX4 EKF2** is a **24-state** extended Kalman filter: attitude quaternion, velocity, position, gyro bias, accelerometer bias, earth
  magnetic field, body magnetometer bias, wind velocity (+ terrain height in a separate filter). The **IMU (≥ 100 Hz)** drives the
  prediction; magnetometer, GNSS, baro/rangefinder/vision height, airspeed, optical flow are observations. Runs on a **delayed fusion
  time horizon** (EKF2_*_DELAY, EKF2_DELAY_MAX) with a complementary filter bringing the state forward to now. **Innovation consistency
  checks** with `EKF2_*_GATE` in standard deviations; test ratios should stay below 0.5 in normal flight, > 1 means rejected.
  Source: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
- **ArduPilot EKF3**: estimates position, velocity and attitude from gyros, accelerometers, compass, GPS, airspeed and baro; with two
  or more IMUs it runs one EKF **core per IMU** in parallel and uses the one with the best health (most consistent sensor data).
  Source: https://ardupilot.org/copter/docs/common-apm-navigation-extended-kalman-filter-overview.html
- **Skydio X10** release notes (support.skydio.com): Cayley release (Mar 11, 2026) mentions 45 mph “Fly Here Now” with obstacle
  avoidance; Draper operator manual (June 2026) says obstacle avoidance can be impaired in low light without NightSense, **does not
  avoid moving objects**, can be degraded near transparent/reflective surfaces, and is disabled during launch and landing.
  Skydio autonomy overview: https://www.skydio.com/autonomy

## History
- Kalman, R. E. (1960). “A New Approach to Linear Filtering and Prediction Problems.” J. Basic Engineering. https://doi.org/10.1115/1.3662552
- Stanley Schmidt (NASA Ames) adapted the Kalman filter for Apollo navigation in the late 1950s/early 1960s (the Schmidt–Kalman filter);
  the FAA later added Schmidt–Kalman filters to its aircraft-tracking calculations. Source (NASA, 2019):
  https://www.nasa.gov/feature/ames/math-invented-for-moon-landing-helps-your-flight-arrive-on-time ; McGee & Schmidt (1985) NASA TM
  “Discovery of the Kalman filter as a practical tool for aerospace and industry”: https://ntrs.nasa.gov/citations/19860003843

## Open source / research used in several phases
- AB3DMOT (Weng et al., arXiv 1907.03961): 3D Kalman filter + Hungarian matching baseline for 3D multi-object tracking.
- robot_localization (ROS): EKF/UKF nodes fusing odometry, IMU, pose and GPS: https://github.com/cra-ros-pkg/robot_localization
- tf2 concepts (ROS 2): https://docs.ros.org/en/rolling/Concepts/Intermediate/About-Tf2.html

## Phase 3 additions (3.6, 3.7)
- **PX4 EKF2 details** (checked 2026-11-01, https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html): “The EKF uses the IMU
  data for state prediction only”; IMU delta angle/velocity needed at ≥ 100 Hz; docs say “an ‘error-state’ formulation is implemented”
  (rotation uncertainty as a 3D vector); GNSS must meet minimums over `GNSS_REQ_TIME` (10 s default): eph < 3 m, epv < 5 m, ≥ 6 sats,
  sacc < 0.5 m/s, fix ≥ 3D, PDOP < 2.5; “transient errors due to multipath, obscuration and interference are more common causes” of poor
  GPS accuracy; vel_test_ratio > 1 = GPS velocity rejected; test ratios “should remain below 0.5” in normal flight.
- **PX4 parameter defaults** (checked 2026-11-01, https://docs.px4.io/main/en/advanced_config/parameter_reference.html):
  EKF2_GPS_P_GATE 5.0 SD, EKF2_GPS_V_GATE 5.0 SD, EKF2_BARO_GATE 5.0 SD, EKF2_MAG_GATE 3.0 SD; EKF2_DELAY_MAX 200 ms;
  SENS_GNSS0_DELAY 110 ms (GPS delay is now a SENS_ parameter, not EKF2_GPS_DELAY); EKF2_PREDICT_US 10000 µs (100 Hz);
  EKF2_NOAID_TOUT 5 s (max inertial dead-reckoning time before the horizontal solution is reported invalid).
- **ArduPilot EKF3 gates** (checked 2026-11-01, https://ardupilot.org/copter/docs/parameters.html): EK3_POS_I_GATE is a “percentage
  number of standard deviations” (300 = 3 SD); “Decreasing it makes it more likely that good measurements will be rejected. Increasing
  it makes it more likely that bad measurements will be accepted.” EK3_GLITCH_RAD: beyond this radius, position and velocity are reset
  to the GPS.
- **robot_localization** (checked 2026-11-01, raw docs on GitHub ros2 branch; docs.ros.org pages block bots with Anubis but work in a
  browser): integrating_gps.rst suggests two EKF instances (continuous odom-frame data only; map-frame with GPS) because GPS has
  “discrete discontinuities (‘jumps’)”; navsat_transform_node needs NavSatFix, an earth-referenced IMU heading and odometry.
  params/ekf.yaml: `*_rejection_threshold` values are “expressed as Mahalanobis distances” and each “defaults to
  numeric_limits<double>::max()”. https://github.com/cra-ros-pkg/robot_localization/blob/ros2/doc/integrating_gps.rst
- **MSCKF**: Mourikis & Roumeliotis, “A Multi-State Constraint Kalman Filter for Vision-aided Inertial Navigation”, ICRA 2007: EKF,
  no 3D feature positions in the state, complexity linear in the number of features, IMU error-state with an error quaternion.
  https://www-users.cse.umn.edu/~stergios/papers/ICRA07-MSCKF.pdf
- **VINS-Mono** (arXiv 1708.03852, IEEE T-RO): monocular camera + IMU, optimization-based sliding window, loop detection and pose-graph
  optimization; README targets drone state estimation and AR. https://github.com/HKUST-Aerial-Robotics/VINS-Mono
- **OpenVINS** (Geneva et al., ICRA 2020): MSCKF-based sliding-window EKF; mono/stereo. https://github.com/rpng/open_vins
  (docs.openvins.com timed out from here on 2026-11-01).
- **Apple ARKit** uses “visual-inertial odometry” (developer.apple.com/documentation/arkit/understanding-world-tracking, checked 2026-11-01).
- **x-io Fusion** AHRS (https://github.com/xioTechnologies/Fusion): README explains it “functions as a complementary filter”; Python
  package `imufusion`.
- **Waymo, mapping (Sep 2020)**: maps help cars “navigate places even where GPS struggles, like tunnels or between skyscrapers”; road
  changes detected by “cross-referencing the real-time sensor data with its on-board map”. No localization-accuracy number is given.
  https://waymo.com/blog/2020/09/the-waymo-driver-handbook-mapping/ . Waymo Feb 2026 post: “In conditions where a camera's view may be
  limited, our lidar and radar provide the necessary redundancy”.
- **Skydio autonomy page** (checked 2026-11-01): “Navigate complex sites, even where GPS isn't available”; no estimator details.
- **GTSAM Huber** (gtsam/linear/LossFunctions.h): weight 1 if |x| < k, k/|x| otherwise. Ceres has HuberLoss (include/ceres/loss_function.h);
  ceres-solver.org did not respond from here on 2026-11-01.
- Chi-square thresholds (computed with scipy): 1 dof 3.841/6.635/10.828; 2 dof 5.991/9.210/13.816; 3 dof 7.815/11.345/16.266
  (95/99/99.9%). P(|N(0,1)| > 3) = 0.27%, > 5 = 5.7e-7.

## Phase 3 additions (3.4, 3.5)
- **Stanley** (Thrun et al., "Stanley: The Robot that Won the DARPA Grand Challenge", J. Field Robotics 23(9):661–692, 2006):
  a **UKF** (Julier & Uhlmann 1997) estimates **15 state variables** (3D position, velocity, orientation as roll/pitch/yaw,
  accelerometer biases, gyro biases) at **100 Hz** from GPS, a GPS compass, the IMU and wheel encoders; a weak "moving mass" model
  while GPS is available, a stricter "moves only where it points" model during GPS outages; the paper notes sigma-point
  linearization "often yields a lower estimation error" than the EKF and needs no explicit Jacobians. Finished in **6 h 53 min 58 s**,
  average **19.1 mph**; 2005 course started and finished in Primm, NV. Sources: https://ri.cmu.edu/publications/stanley-the-robot-that-won-the-darpa-grand-challenge
  and PDF https://pages.cs.wisc.edu/~dyer/cs540/handouts/thrun.stanley05.pdf (checked 2026-11-01). The paper does not state the
  course length in a form I could confirm; don't quote one.
- **UKF origin**: Julier & Uhlmann, "A New Extension of the Kalman Filter to Nonlinear Systems", SPIE 3068 (1997); Proc. IEEE 92(3)
  (2004) "Unscented Filtering and Nonlinear Estimation" (doi 10.1109/JPROC.2003.823141). Uhlmann's own account of the name
  ("unscented", from a colleague's deodorant, chosen because it was meaningless): https://ethw.org/First-Hand:The_Unscented_Transform
- **Monte Carlo localization**: Dellaert, Fox, Burgard & Thrun, ICRA 1999, vol. 2, pp. 1322–1328: samples instead of a grid; localizes
  without knowing the start; "faster, more accurate and less memory-intensive than earlier grid-based methods".
  https://www.ri.cmu.edu/publications/monte-carlo-localization-for-mobile-robots
- **KLD-sampling**: D. Fox, NIPS 2001 (pp. 713–720), adapts the particle count; IJRR 2003 journal version.
  https://papers.nips.cc/paper_files/paper/2001/hash/c5b2cebf15b205503560c4e8e6d1ea78-Abstract.html
- **Particle filter, other origins**: bootstrap filter, Gordon, Salmond & Smith, IEE Proc. F 140(2):107–113 (1993);
  CONDENSATION, Isard & Blake, IJCV 29(1):5–28 (1998), https://doi.org/10.1023/A:1008078328650 (abstract: a Kalman filter's
  unimodal Gaussian "cannot represent simultaneous alternative hypotheses").
- **Nav2 AMCL** (docs checked 2026-11-01, https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/others/configuring_amcl/ ;
  the old URL docs.nav2.org/configuration/packages/configuring-amcl.html now 404s): defaults `min_particles` 500, `max_particles` 2000,
  `laser_model_type` "likelihood_field" (also "beam", "likelihood_field_prob"), `max_beams` 60, `sigma_hit` 0.2, `z_hit` 0.5,
  `z_rand` 0.5, `laser_likelihood_max_dist` 2.0, `update_min_d` 0.25 m, `update_min_a` 0.2 rad, `resample_interval` 1,
  `pf_err` 0.05, `pf_z` 0.99, `recovery_alpha_fast` / `recovery_alpha_slow` 0.0 (off; "a good value might be" 0.1 / 0.001),
  `robot_model_type` nav2_amcl::DifferentialMotionModel (or OmniMotionModel), `tf_broadcast` true (publishes global→odom).
  nav2_amcl README: "largely a refactored port from ROS 1 without any algorithmic changes"; pf.c implements KLD resample limit and
  w_slow/w_fast random-particle recovery. nav2_bringup `localization_launch.py` starts `amcl`. https://github.com/ros-navigation/navigation2/tree/main/nav2_amcl
- **robot_localization** docs: `ekf_localization_node` uses an omnidirectional motion model; `ukf_localization_node` uses sigma points
  through the same model, "eliminates the use of Jacobian matrices", more computationally taxing. ROS 2 branch executables:
  `ekf_node`, `ukf_node` (CMakeLists, ros2 branch). https://github.com/cra-ros-pkg/robot_localization
- **FilterPy API** (filterpy.readthedocs.io, checked 2026-11-01): `ExtendedKalmanFilter(dim_x, dim_z)`, `.update(z, HJacobian, Hx,
  R=None, args=(), hx_args=(), residual=np.subtract)`; `UnscentedKalmanFilter(dim_x, dim_z, dt, hx, fx, points, ...)`;
  `MerweScaledSigmaPoints(n, alpha, beta, kappa)`, also `JulierSigmaPoints`, `SimplexSigmaPoints`; `filterpy.monte_carlo`
  `systematic_resample`, `stratified_resample`, `residual_resample`, `multinomial_resample` (all take normalized weights, return indexes).
- **Waymo on localization** (blog, Sep 2020, "The Waymo Driver Handbook: mapping"): the Driver knows where it is from "highly-detailed,
  custom maps", built by driving streets so "our custom lidar can paint a 3D picture"; maps help "even where GPS struggles, like
  tunnels or between skyscrapers"; it detects road changes by cross-referencing sensor data with the onboard map. No accuracy figure
  and no algorithm given. https://waymo.com/blog/2020/09/the-waymo-driver-handbook-mapping/
- Tesla AI Day 2021 video (oEmbed title "Tesla AI Day 2021"): https://www.youtube.com/watch?v=j0z4FweCy4M . Tesla has not published
  its ego-localization method.

## Phase 1 additions
- **REP 103** (checked 2026-11-01, source text https://github.com/ros-infrastructure/rep/blob/master/rep-0103.rst, page
  https://www.ros.org/reps/rep-0103.html): SI units, angles in radians; right-handed; body axes **x forward, y left, z up**;
  geographic short-range frames **ENU**; camera **"_optical"** frames z forward, x right, y down; **"_ned"** suffix for NED frames;
  rotation preference order: quaternion, rotation matrix, fixed-axis RPY, Euler (“generally discouraged due to having 24 'valid'
  conventions”); geographic yaw is zero pointing **east**, counter-clockwise, “differs from a traditional compass bearing”.
- **REP 105** (same repo, rep-0105.rst): odom is continuous but drifts; map doesn't drift but jumps; tree earth → map → odom →
  base_link; each frame has one parent (so map is odom's parent); odometry publishes odom → base_link; localization computes
  map → base_link but **publishes map → odom**.
- **PX4 frames** (2026-11-01): https://docs.px4.io/main/en/ros2/user_guide.html: PX4 uses FRD body / NED world, ROS uses FLU / ENU;
  “The FRD (NED) conventions are adopted on all PX4 topics unless explicitly specified”; ENU→NED = π/2 about z then π about x;
  FLU↔FRD = π about x; px4_ros_com provides `frame_transforms`. https://docs.px4.io/main/en/contribute/notation.html: attitude is a
  Hamiltonian quaternion q = (q0, q1, q2, q3), scalar first. https://docs.px4.io/main/en/flight_stack/controller_diagrams.html:
  “The attitude controller makes use of quaternions.” Params: `SENS_BOARD_ROT` (board rotation relative to the platform),
  `EKF2_IMU_POS_X/Y/Z` (IMU position in body frame; EKF2 docs: outputs “adjusted to account for the offset between the IMU and the
  body frame”).
- **Apollo 11 gimbal lock** (Apollo Lunar Surface Journal, https://www.apollojournals.org/alsj/a11/a11.postland.html, checked
  2026-11-01): 104:59:35 Garriott: “We noticed you are maneuvering very close to gimbal lock. I suggest you move back away.”;
  105:00:30 Collins: “How about sending me a fourth gimbal for Christmas.” Armstrong's comment: “We had a four-gimbal platform on
  Gemini.” Journal explainer: three nested gimbals; mission planning used REFSMMAT alignments to avoid lock.
- **Hamilton / quaternions** (MacTutor, https://mathshistory.st-andrews.ac.uk/Biographies/Hamilton/): 16 October 1843, walking
  along the Royal Canal; carved i² = j² = k² = ijk = −1 on Broome (Brougham) Bridge; children asked “Well, Papa, can you multiply
  triplets?”; “the first noncommutative algebra to be studied”; Royal Irish Academy plaque in 1958.
- **Möbius** introduced homogeneous coordinates in *Der barycentrische Calcul* (1827) (MacTutor,
  https://mathshistory.st-andrews.ac.uk/Biographies/Mobius/).
- **Waymo Open Dataset frames** (arXiv 1912.04838, Sec. 3.1–3.2): 5 lidars + 5 cameras; global frame ENU; vehicle frame x forward,
  y left, z up; vehicle pose = 4×4 vehicle → global; sensor extrinsics = 4×4 sensor → vehicle; camera sensor frame has **x out of
  the lens**, z up; each range-image pixel carries the vehicle pose at capture time; camera–lidar sync error within [−6 ms, 7 ms]
  at 99.7% confidence; rolling-shutter-aware projections.
- **Tesla camera calibration** (Model 3 owner's manual, Cameras page,
  https://www.tesla.com/ownersmanual/model3/en_us/GUID-682FF4A7-D083-4C95-925A-5EE3752F4865.html, checked 2026-11-01):
  calibration “typically completes after driving 20-25 miles (32-40 km)”; contact Tesla if not done after 100 miles; clear it via
  Controls > Service > Camera Calibration > Clear Calibration “If a camera has shifted from its calibrated position (for example,
  the camera or windshield was replaced)”.
- **Tesla AI Day 2021** video (Tesla's channel): https://www.youtube.com/watch?v=j0z4FweCy4M (oEmbed title “Tesla AI Day 2021”);
  AI Day 2022: https://www.youtube.com/watch?v=ODSJsviD_SU. The eight-camera fusion into a vehicle-centered “vector space” is widely
  reported from the 2021 talk (secondary write-ups); cite the video and phrase as “Tesla showed/described”.
- **ROS 2 tf2** (ros2_documentation, jazzy branch, checked 2026-11-01): `ros2 run tf2_ros static_transform_publisher --x .. --y ..
  --z .. --yaw .. --pitch .. --roll .. --frame-id A --child-frame-id B` (or `--qx --qy --qz --qw`), radians, all but the two frame
  args optional; tf2 keeps transforms “for up to 10 seconds by default”; time 0 / `rclpy.time.Time()` = latest; static transforms
  “published once and assumed to not change, so no history is stored” (/tf_static); `lookup_transform(target, source, time,
  timeout=Duration())`. Error strings (geometry2 source): “Lookup would require extrapolation into the future.  Requested time …
  but the latest data is at time …”, “…into the past… earliest data…”, “"X" passed to lookupTransform argument target_frame does
  not exist.”, “…because they are not part of the same tree.Tf has two or more unconnected trees.” ROS 2 quaternions are (x, y, z, w),
  w last; geometry_msgs/Quaternion defaults to w = 1. Jazzy support: May 2024 – May 2029; Kilted: May 2025 – Nov 2026 (REP 2000).
- tf paper: T. Foote, “tf: The transform library”, IEEE TePRA 2013, doi:10.1109/TePRA.2013.6556373 (Crossref-verified).
- Nav2 (https://docs.nav2.org/jazzy/getting_started/navigation_concepts/state_estimation/): the TF tree must contain “a full map ->
  odom -> base_link -> [sensor frames]”; map → odom from localization/SLAM, odom → base_link from odometry, sensor links static
  in the URDF. robot_state_publisher (github.com/ros/robot_state_publisher): fixed joints to /tf_static once, movable joints to /tf.
- SciPy 1.13: `as_euler` at pitch ±90° warns “Gimbal lock detected. Setting third angle to zero since it is not possible to uniquely
  determine all angles.”; `as_quat()` is scalar-last (x, y, z, w). OpenCV `getRotationMatrix2D`: degrees, positive =
  counter-clockwise with origin top-left.
