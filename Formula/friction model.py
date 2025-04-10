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
v      = 0.01        # [m/s]  给定恒定速度

# Normal force (若需要)
N = 1.0

# 速度因子 g(v)
def g(v):
    return mu_c + (mu_s - mu_c)*np.exp(-abs(v)/v_s)

# ---------------------------
# 2. 数值积分 (可用解析解, 这里演示Euler)
# ---------------------------
t_end = 10       # 仿真终止时间 [s]
dt    = 0.001     # 步长
n_step = int(t_end/dt) + 1
t_arr  = np.linspace(0, t_end, n_step)

z_arr  = np.zeros_like(t_arr)  # 存储 z(t)
f_arr  = np.zeros_like(t_arr)  # 存储 f_r(t)
z      = 0.0                   # 初值

g_val  = g(v)                  # 速度不变，g(v)也恒定

for i, t in enumerate(t_arr):
    # 计算摩擦力
    z_dot = v - (sigma0/g_val)*v*z
    f_r   = sigma0*z + sigma1*z_dot + sigma2*v
    
    z_arr[i] = z
    f_arr[i] = f_r
    
    # 用Euler积分 z(t+dt) = z(t) + z_dot*dt
    z = z + z_dot*dt

# ---------------------------
# 3. 画图
# ---------------------------
plt.figure(figsize=(10,8))

plt.subplot(3,1,1)
plt.plot(t_arr, f_arr, label='LuGre friction')
plt.axhline(mu_c + sigma2*v, color='r', linestyle='--', 
            label=r'$F=\mu_c + \sigma_2 v$ (simple model)')
plt.title("Friction Force vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("Friction [N]")
plt.legend()
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(t_arr, z_arr, label='Bristle Deformation z(t)')
plt.title("Bristle Deformation vs. Time")
plt.xlabel("Time [s]")
plt.ylabel("z(t) [m] (dimension depends on model)")
plt.legend()
plt.grid(True)

plt.subplot(3,1,3)
# 因为 g(v) 在此例里是常数, 直接画一条横线
g_arr = np.full_like(t_arr, g_val)
plt.plot(t_arr, g_arr, label='g(v)=const')
plt.title("g(v) vs. Time (v=0.01 m/s)")
plt.xlabel("Time [s]")
plt.ylabel("g(v)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
