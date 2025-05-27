import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy import interpolate


# パラメータ
m = 0.06  # 質量 (kg)
g = 9.80665  # 重力加速度 (m/s^2)
rho = 1.225  # 空気密度 (kg/m^3)
A = 0.6*0.25 #0.026  # 表面積 (m^2)
L = 0.26  # スリッパの長さ (m)
W = 0.1  # スリッパの幅 (m)
c = L  # 基準長さ (m)
x_g = 0.2 * L  # 重心位置（例: 中心）
x_cp = 0.21 * L  # 圧力中心（中心）
l = abs(x_cp - x_g)  # 重心から圧力中心までの距離 (m)
I = (1/12) * m * (L**2 + W**2)  # 慣性モーメント (kg·m^2)

angles = np.deg2rad(np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 45, 90, 135, 180, 225, 270, 350, 352, 354, 356, 358, 360]))

cl = np.array([0.000, 0.220, 0.440, 0.660, 0.880, 1.100, 1.300, 1.450, 1.500,
      1.400, 1.200, 0.800, 0.600, 0.000, -0.600, 0.000,
      0.600, 0.000, -1.100, -0.880, -0.660, -0.440, -0.220, 0.000])

cd = np.array([0.0060, 0.0065, 0.0070, 0.0080, 0.0095, 0.0120, 0.0150, 0.0200,
      0.0300, 0.0500, 0.0800, 0.2000, 0.4000, 1.2000, 0.4000, 0.0060,
      0.4000, 1.2000, 0.0120, 0.0095, 0.0080, 0.0070, 0.0065, 0.0060])
# 空力係数
def C_L(alpha):
    ret_CL = 1.2 * np.sin(2 * alpha)
    ret_CL = interpolate.interp1d(angles, cl)(alpha)
    return ret_CL

def C_D(alpha):
    ret_CD = 0.1 + 1.0 * np.sin(alpha)**2
    ret_CD = interpolate.interp1d(angles, cd)(alpha)
    return ret_CD

def C_m(alpha):
    return 0.0 * np.sin(2 * alpha)  # ピッチモーメント係数

# 運動方程式
def equations(state, t):
    x, x_dot, z, z_dot, theta, theta_dot = state
    v = np.sqrt(x_dot**2 + z_dot**2)
    gamma = np.arctan2(z_dot, x_dot) if v > 1e-6 else 0
    alpha = (theta - gamma) % (2 * np.pi)

    # 空気力
    F_L = 0.5 * rho * v**2 * A * C_L(alpha)
    F_D = 0.5 * rho * v**2 * A * C_D(alpha)
    M = 0.5 * rho * v**2 * A * c * C_m(alpha) * (l / c)

    # 空気力の方向（速度ベクトルに対する角度を考慮）
    F_x = -F_D * np.cos(gamma) - F_L * np.sin(gamma)  # x方向
    F_z = -F_D * np.sin(gamma) + F_L * np.cos(gamma)  # z方向

    # 並進運動
    x_ddot = F_x / m
    z_ddot = (-m * g + F_z) / m

    # 回転運動
    theta_ddot = M / I

    return [x_dot, x_ddot, z_dot, z_ddot, theta_dot, theta_ddot]

# 初期条件
x0 = 0.0  # 初期水平位置 (m)
v0 = 5.0  # 初期速度 (m/s)
theta0 = np.radians(10)  # 初期ピッチ角 (rad)

z0 = 1.5  # 初期高度 (m)

x_dot0 = v0*np.cos(theta0)
z_dot0 = v0*np.sin(theta0)  # 初期鉛直速度 (m/s)

theta_dot0 = np.radians(-5)  # 初期角速度 (rad/s)
state0 = [x0, x_dot0, z0, z_dot0, theta0, theta_dot0]

# 時間設定
t = np.linspace(0, 5, 1000)  # 0～5秒、1000点

# 微分方程式を解く（地面衝突を考慮）
solution = []
state = state0
dt = t[1] - t[0]
for ti in t:
    solution.append(state)
    if state[2] <= 0:  # 地面衝突 (z <= 0)
        break
    state = odeint(equations, state, [ti, ti + dt])[-1]

solution = np.array(solution)

# 結果の抽出
x = solution[:, 0]
z = solution[:, 2]
theta = solution[:, 4]

# 飛距離の出力
fly_distance = x[-1]
print(f"飛距離: {fly_distance:.2f} m")

# 可視化
fig = plt.figure(1,figsize=(12, 8))

# 軌跡
ax = fig.add_subplot(2, 1, 1)
ax.plot(x, z, label='Trajectory')

SP_plot_step = 10
ax.plot([x[::SP_plot_step] - 0.26*np.cos(solution[::SP_plot_step,4]), x[::SP_plot_step] + 0.26*np.cos(solution[::SP_plot_step,4])],
        [z[::SP_plot_step] - 0.26*np.sin(solution[::SP_plot_step,4]), z[::SP_plot_step] + 0.26*np.sin(solution[::SP_plot_step,4])])
ax.set_xlim([0,30])
ax.set_ylim([0,10])
ax.set_aspect('equal')
ax.set_xlabel('Horizontal Distance (m)')
ax.set_ylabel('Height (m)')
ax.grid(True)
ax.legend()

# ピッチ角
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(t[:len(x)], np.degrees(theta % (2 * np.pi)), label='Pitch Angle (θ mod 360°)')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Angle (degrees)')
ax2.grid(True)
ax2.legend()

# plt.tight_layout()
# plt.show()