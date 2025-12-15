import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from neuralop.models import FNO

# Parameters
k = 0.35
A = 0.1
n0 = 1.0
T0 = 1.0
me = 1.0
vt = np.sqrt(T0/me)
qe = -1.0
eps0 = 1.0
omega_pe = 1.0
dt = 2e-3
tmax = 80.0
ts = []#時間リスト
Eenergy = []#電場エネルギー
n_history = []#密度の時間変化
u_history = []#速度の時間変化
p_history = []#圧力の時間変化
dqdx_history = []#∂q/∂xの時間変化
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

#MLclosure
num_modes = 16 # フーリエ空間で使用するモードの数
num_channels = 64 # インプットとアウトプットの間の層の数
device = 'cpu'
ckpt = torch.load('machine_learning/learnedmodel_from_vlasov_colab_ep90.pth', map_location=device, weights_only=False)
model = FNO(
    n_modes=(num_modes,), n_layers=4,hidden_channels=num_channels, in_channels= 3, out_channels=1
).to(device)
model.load_state_dict(ckpt)
model.eval()

sc = np.load("machine_learning/scaler_ep90.npz")
mu_n,  sig_n  = sc["mu_n"],  sc["sig_n"]
mu_u,  sig_u  = sc["mu_u"],  sc["sig_u"]
mu_p,  sig_p  = sc["mu_p"],  sc["sig_p"]
mu_dq, sig_dq = sc["mu_dq"], sc["sig_dq"]   # ∂q/∂x の教師側スケール（必要なら逆規格化に使用）
print("mu_n =", mu_n, "sig_n =", sig_n)
print("mu_u =", mu_u, "sig_u =", sig_u)
print("mu_p =", mu_p, "sig_p =", sig_p)
print("mu_dq =", mu_dq, "sig_dq =", sig_dq)

def normalize_inputs(n,u,p):
    n_ = (n - mu_n)/sig_n
    u_ = (u - mu_u)/sig_u
    p_ = (p - mu_p)/sig_p
    return n_, u_, p_

def denorm_dqdx(dqdx_norm):
    return dqdx_norm*sig_dq + mu_dq

@torch.no_grad()
def predict_dqdx(n, u, p):
    """
    n,u,p : np.ndarray shape [Nx]
    return : np.ndarray shape [Nx]  (物理スケールの ∂q/∂x)
    """
    n_, u_, p_ = normalize_inputs(n,u,p)
    x = np.stack([n_, u_, p_], axis=0)[None, ...]   # [1,3,Nx]
    x = torch.from_numpy(x.astype(np.float32)).to(device)
    y = model(x)                                    # [1,1,Nx]
    dqdx_norm = y[0,0].detach().cpu().numpy()
    dqdx = denorm_dqdx(dqdx_norm)
    #dqdx = y[0,0].detach().cpu().numpy()
    return dqdx

#右辺の計算
alpha = 1.0
def rhs(n,u,p,Ex): 
    dn = -d_dx(n*u) 
    du = -u*d_dx(u) - (1.0/(me*n))*d_dx(p) + (qe/me)*Ex 
    dp = -u*d_dx(p) - 3.0*p*d_dx(u) -alpha*predict_dqdx(n,u,p)
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
ts = []
Eenergy = []
n_history = []
u_history = []
p_history = []
dqdx_history = []
Ex_history = []
step=0
while t < tmax:
    ts.append(t)
    Eenergy.append(0.5*L_x*np.mean(Ex**2))
    n_history.append(n)
    u_history.append(u)
    p_history.append(p)
    dqdx_history.append(predict_dqdx(n,u,p))
    Ex_history.append(Ex)
    
    n,u,p,Ex = rk4_step(n,u,p,Ex,dt)
    t += dt
    step +=1
    #print(np.max(predict_dqdx(n,u,p)), np.min(predict_dqdx(n,u,p)))
    #print(Eenergy)
    if step % 1000 == 0:  # 100ステップごとくらいに
        print(f"t={t:.3f}",
              " n[min,max]=", np.min(n), np.max(n),
              " u[max]=", np.max(np.abs(u)),
              " p[max]=", np.max(np.abs(p)),
              " dqdx[max]=", np.max(np.abs(predict_dqdx(n,u,p))))


Eenergy = np.array(Eenergy)/Eenergy[0]

Energy_history = Eenergy/Eenergy[0]

t_history = np.array(ts)             # (Nt,)
n_history = np.array(n_history)             # (Nt, N)
u_history = np.array(u_history)             # (Nt, N)
p_history = np.array(p_history)             # (Nt, N)
dq_dx_history = np.array(dqdx_history)     # (Nt, N)
Energy_history = np.array(Energy_history)   # (Nt,)


# 保存用ディレクトリ（図と同じフォルダ）
outdir = "raw_data/ml_closure_A=0.1_t80"
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

"""
# Plot

def spacetime_plot(data, title, fname):
    plt.figure(figsize=(8,6))
    plt.imshow(np.array(data).T, aspect='auto', origin='lower', extent=[0, tmax, 0, L_x])
    plt.xlabel('t'); plt.ylabel('x'); plt.title(title)
    plt.colorbar(); plt.tight_layout()
    plt.savefig(fname, dpi=150)
    plt.close()

spacetime_plot(n_history, 'density', 'picture/ml_closure_A=0.105/n.png')
spacetime_plot(u_history, 'velocity', 'picture/ml_closure_A=0.105/u.png')
spacetime_plot(p_history, 'pressure', 'picture/ml_closure_A=0.105/p.png')
spacetime_plot(dqdx_history, '∂q/∂x', 'picture/ml_closure_A=0.105/dqdx.png')
spacetime_plot(Ex_history, 'electric field', 'picture/ml_closure_A=0.105/Ex.png')
plt.figure()
plt.plot(ts, Eenergy)
plt.xlabel('t')
plt.ylabel('Eenergy')
plt.yscale('log')
plt.grid(True)
plt.savefig('picture/ml_closure_A=0.105/Eenergy.png', dpi=150)
plt.close()
"""