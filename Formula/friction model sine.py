import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# 1. 参数设定
# ---------------------------
sigma0 = 8.0         # [N/m]
sigma1 = 0.089       # [N·s/m]
sigma2 = 0.05        # [N·s/m]
mu_s   = 0.3
mu_c   = 0.2339
v_s    = 1e-3        # [m/s]

# 设置正弦波速度参数 (总是正值)
v0 = 0.005          # 平均值 [m/s]
A  = 0.005          # 振幅 [m/s]
T  = 10            # 正弦波周期 [s]

# Normal force, N = 1 (可乘以 N)
N = 1.0

# 速度因子 g(v)
def g(v):
    return mu_c + (mu_s - mu_c)*np.exp(-abs(v)/v_s)

# ---------------------------
# 2. 数值积分，采用Euler方法求解刚毛变形微分方程
#    修改后的动态方程（考虑 v 为时间函数）：
#       dz/dt = v(t) - (sigma0* v(t) / g(v(t)))*z(t)
# ---------------------------
t_end = 20      # 仿真总时长 [s]
dt    = 0.001     # 时间步长
n_step = int(t_end/dt) + 1
t_arr  = np.linspace(0, t_end, n_step)

z_arr  = np.zeros_like(t_arr)  # 存储 z(t)
f_arr  = np.zeros_like(t_arr)  # 存储摩擦力 f_r(t)
v_arr  = np.zeros_like(t_arr)  # 存储速度 v(t)
g_arr  = np.zeros_like(t_arr)  # 存储 g(v(t))
f_simple_arr = np.zeros_like(t_arr)  # 稳态简单模型 f = g(v) + sigma2*v

z = 0.0

for i, t in enumerate(t_arr):
    # 正弦波速度：保证速度始终为正
    v = v0 + A * np.sin(2*np.pi*t/T)
    # v = A * np.sin(2 * np.pi * t / T)

    v_arr[i] = v
    g_val = g(v)
    g_arr[i] = g_val
    
    # 刚毛变形微分方程 (Euler积分)
    # 注意: 对于正速度情形，此公式与之前一致
    z_dot = v - (sigma0 * v / g_val)*z
    z_arr[i] = z
    
    # 摩擦力计算
    # f_r = sigma0*z + sigma1*z_dot + sigma2*v
    f_arr[i] = sigma0*z + sigma1*z_dot + sigma2*v
    
    # 稳态摩擦力 (当 z 达到稳态 z_ss = g(v)/sigma0, 则 f_ss = g(v) + sigma2*v)
    f_simple_arr[i] = mu_c*N + sigma2*v
    
    # Euler积分更新 z
    z = z + z_dot*dt

# ---------------------------
# 3. 绘图
# ---------------------------
plt.figure(figsize=(10,10))

plt.subplot(4,1,1)
plt.plot(t_arr, v_arr, label='Velocity v(t)')
plt.title("input speed (sine): v(t) = {:.3f} + {:.3f} sin(2πt/T)".format(v0, A))
plt.xlabel("Time [s]")
plt.ylabel("v(t) [m/s]")
plt.legend()
plt.grid(True)

plt.subplot(4,1,2)
plt.plot(t_arr, f_arr, label='LuGre Friction')
plt.plot(t_arr, f_simple_arr, '--', label='simple friction model')
plt.title("friction f₍r₎(t) vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("Friction [N]")
plt.legend()
plt.grid(True)

plt.subplot(4,1,3)
plt.plot(t_arr, z_arr, label='Bristle Deformation z(t)')
plt.title("Bristle Deformation z(t) vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("z(t) [m]")
plt.legend()
plt.grid(True)

plt.subplot(4,1,4)
plt.plot(t_arr, g_arr, label='g(v(t))')
plt.title("g(v(t)) vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("g(v)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
