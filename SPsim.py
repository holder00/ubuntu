import numpy as np
import matplotlib.pyplot as plt

# パラメータ
rho = 1.225  # 空気密度 [kg/m^3]
Sopen = 1.0*0.50     # 翼面積 [m^2]
Sclose = 0.02      # 翼面積 [m^2]
mass = 60/1000   # 質量 [kg]
I = 100/1000      # 慣性モーメント [kg·m^2]（例）
l = 0.01      # 揚力作用点と重心の距離 [m]
g = 9.80665
dt = 0.01
maxT = 10
steps = int(maxT / dt)

# 揚力係数と抗力係数（簡易モデル、直線と二次式）
def CL_open(alpha):
    CL1 = 4.0
    CL2 = 0.1
    if np.abs(alpha) < np.deg2rad(15):
        ret_CL = CL1 * alpha + CL2 # ラジアンでの迎角
    else:
        ret_CL = CL1 * alpha + CL2 -3*(alpha - np.sign(alpha)*np.deg2rad(15)) # ラジアンでの迎角
    # if np.sign(alpha *sou
    print(np.rad2deg(alpha),ret_CL)

    return ret_CL

def CD_open(alpha):
    Cd1 = 0.2
    Cd2 = 0.01
    ret_CD = Cd1 + Cd2 * (alpha ** 2)
    # print(ret_CD)
    return ret_CD

def CL_close(alpha):
    CL1 = 0.01
    CL2 = 0.0
    if np.abs(alpha) < np.deg2rad(15):
        ret_CL = CL1 * alpha + CL2 # ラジアンでの迎角
    else:
        ret_CL = -CL1 * alpha + CL2 # ラジアンでの迎角
    return ret_CL

def CD_close(alpha):
    Cd1 = 1.3
    Cd2 = 0.0
    return Cd1*np.sin(alpha)


# 初期状態
x, y = 0.0, 0.0
theta = np.deg2rad(45)
v0 = 30
vx, vy = v0*np.cos(theta), v0*np.sin(theta)  # x方向に初速
omega = 0.0     # 角速度

# theta = 0.0         # 姿勢角（ラジアン）

trajectory = []
time = 0
for _ in range(steps):
    time += dt
    V = np.sqrt(vx**2 + vy**2)
    # if V < 1e-6:
    #     break
    alpha = np.arctan2(vy, vx) - theta
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi

    if time < 0:
        T = 0.05
        S = Sclose
        CL = CL_close
        CD = CD_close
    else:
        T = 0
        S = Sopen
        CL = CL_open
        CD = CD_open

    L = 0.5 * rho * V**2 * S * CL(alpha)
    D = 0.5 * rho * V**2 * S * CD(alpha)


    # 単位ベクトル
    vx_unit = vx / V
    vy_unit = vy / V

    # 空力ベクトル（速度基準）
    drag_fx = -D * vx_unit
    drag_fy = -D * vy_unit
    lift_fx = -L * vy_unit
    lift_fy =  L * vx_unit

    # 合力
    fx = drag_fx + lift_fx + T*np.cos(theta)
    fy = drag_fy + lift_fy - mass * g + T*np.sin(theta)

    # モーメント（重心からの距離lを横方向にオフセット）
    M = l * L  # 正方向: 揚力が反時計回りトルクを与える

    # 運動方程式
    ax = fx / mass
    ay = fy / mass
    alpha_dot = M / I

    # 更新
    vx += ax * dt
    vy += ay * dt
    x += vx * dt
    y += vy * dt
    omega += alpha_dot * dt
    theta += omega * dt

    trajectory.append((x, y, vx, vy, theta))
    # print(np.rad2deg(alpha),vx,vy)
    if y < 0:
        break

# プロット
trajectory = np.array(trajectory)
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.plot(trajectory[:,0], trajectory[:,1])

SP_plot_step = 5
ax.plot([trajectory[::SP_plot_step,0] - 0.26*np.cos(trajectory[::SP_plot_step,4]), trajectory[::SP_plot_step,0] + 0.26*np.cos(trajectory[::SP_plot_step,4])],
        [trajectory[::SP_plot_step,1] - 0.26*np.sin(trajectory[::SP_plot_step,4]), trajectory[::SP_plot_step,1] + 0.26*np.sin(trajectory[::SP_plot_step,4])])
ax.set_xlim([0,30])
ax.set_ylim([0,10])
ax.set_aspect('equal')
ax.set_xlabel('x [m]')
ax.set_label('y [m]')
ax.set_title('2D Aerodynamic Trajectory')

ax.grid()
# plt.show()
