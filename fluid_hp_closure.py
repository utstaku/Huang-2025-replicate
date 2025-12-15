import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
k = 0.35
A = 0.1
n0 = 1.0
T0 = 1.0
me = 1.0
vt = np.sqrt(T0/me)
qe = -1.0
eps0 = 1.0
kai = 2/np.sqrt(np.pi)
omega_pe = 1.0
dt = 2e-3
tmax = 40.0
ts = []#時間リスト
Eenergy = []#電場エネルギー
n_history = []#密度の時間変化
u_history = []#速度の時間変化
p_history = []#圧力の時間変化
dq_dx_history = []
Ex_history = []#電場の時間変化

L_x = 2*np.pi/k
N_x = 64
dx = L_x/N_x
x = np.linspace(0, L_x, N_x, endpoint=False)


# Initial conditions
n = n0*(1.0 + A*np.cos(k*x))
u = np.zeros_like(x)
p = n0*T0*np.ones_like(x)
p = n*T0
Ex = (qe * n0 * A / (eps0 * k)) * np.sin(k*x)
#Ex = np.zeros_like(x)

#微分演算子 
# #空間方向には差分法ではなくフーリエ変換を用いる 
kvec = 2*np.pi*np.fft.fftfreq(N_x, d=dx)#微分演算子のための波数ベクトル 



def d_dx(f): 
    return np.fft.ifft(1j*kvec*np.fft.fft(f)).real 

#HPclosure
absk = np.abs(kvec)
absk[0] = 1.0
def HPclosure(n,p):
    n_hat = np.fft.fft(n)
    p_hat = np.fft.fft(p)
    T_hat = p_hat - (T0/n0)*n_hat

    k2_over_absk = (kvec**2)/absk

    dqdx_hat = n0*kai*vt*T_hat*k2_over_absk
    dqdx_hat[0] = 0.0
    return np.fft.ifft(dqdx_hat).real
    
def rhs(n,u,p,Ex): 
    dn = -d_dx(n*u) 
    du = -u*d_dx(u) - (1.0/(me*n))*d_dx(p) + (qe/me)*Ex 
    dp = -u*d_dx(p) - 3.0*p*d_dx(u) -HPclosure(n,p)
    dEx = -(qe/eps0)*n*u 
    return dn, du, dp, dEx

#時間進化(4次ルンゲクッタ法)
def rk4_step(n,u,p,Ex,dt):
    k1 = rhs(n,u,p,Ex)
    k2 = rhs(n+0.5*dt*k1[0], u+0.5*dt*k1[1], p+0.5*dt*k1[2], Ex+0.5*dt*k1[3])
    k3 = rhs(n+0.5*dt*k2[0], u+0.5*dt*k2[1], p+0.5*dt*k2[2], Ex+0.5*dt*k2[3])
    k4 = rhs(n+dt*k3[0], u+dt*k3[1], p+dt*k3[2], Ex+dt*k3[3])
    n_new = n + (dt/6.0)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    u_new = u + (dt/6.0)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    p_new = p + (dt/6.0)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    Ex_new = Ex + (dt/6.0)*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
    return n_new, u_new, p_new, Ex_new

#time evolution
t = 0.0
ts = [t]
Eenergy = [0.5*L_x*np.mean(Ex**2)]
n_history = [n.copy()]
u_history = [u.copy()]
p_history = [p.copy()]
dq_dx_history = [HPclosure(n,p).copy()]
Ex_history = [Ex.copy()]
while t < tmax:
    n,u,p,Ex = rk4_step(n,u,p,Ex,dt)
    t += dt
    #print(Eenergy)
    
    ts.append(t)
    Eenergy.append(0.5*L_x*np.mean(Ex**2))
    n_history.append(n)
    u_history.append(u)
    p_history.append(p)
    dq_dx_history.append(HPclosure(n,p))
    Ex_history.append(Ex)

Eenergy = np.array(Eenergy)/Eenergy[0]

Energy_history = Eenergy/Eenergy[0]

t_history = np.array(ts)             # (Nt,)
n_history = np.array(n_history)             # (Nt, N)
u_history = np.array(u_history)             # (Nt, N)
p_history = np.array(p_history)             # (Nt, N)     # (Nt, N)
Energy_history = np.array(Energy_history)   # (Nt,)

# 保存用ディレクトリ（図と同じフォルダ）
outdir = "raw_data/hp_closure_A=0.1"
os.makedirs(outdir, exist_ok=True)

# まとめて .npz で保存（バイナリ。後で Python から読みやすい）
np.savez(
    f"{outdir}/moments.npz",
    t=t_history,
    x=x,
    n=n_history,
    u=u_history,
    p=p_history,
    dq_dx=dq_dx_history,
    Energy=Energy_history,  # 正規化前のエネルギー
)
# Plot

def spacetime_plot(data, title, fname):
    plt.figure(figsize=(8,6))
    plt.imshow(np.array(data).T, aspect='auto', origin='lower', extent=[0, tmax, 0, L_x])
    plt.xlabel('t'); plt.ylabel('x'); plt.title(title)
    plt.colorbar(); plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

spacetime_plot(n_history, 'density', 'picture/hp_closure_A=0.1/n.png')
spacetime_plot(u_history, 'velocity', 'picture/hp_closure_A=0.1/u.png')
spacetime_plot(p_history, 'pressure', 'picture/hp_closure_A=0.1/p.png')
spacetime_plot(Ex_history, 'electric field', 'picture/hp_closure_A=0.1/Ex.png')
plt.figure()
plt.plot(ts, Eenergy)
plt.xlabel('t')
plt.ylabel('Eenergy')
plt.yscale('log')
plt.grid(True)
plt.savefig('picture/hp_closure_A=0.1/Eenergy.png', dpi=150)
plt.close()