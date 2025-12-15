import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from scipy.signal import argrelextrema

#parameters
A = 0.05
k0 = 0.1
k = 0.4
N = 64         # x方向のgrid数
M = 128         # v方向のgrid数(半分)
Vmax = 6.0     # vの打ち切り速度
dt = 2e-3       # time step (ω_pe^{-1} units)
tmax = 40.0    # end time (enough to see at least one recurrence)

# 保存用ディレクトリ（図と同じフォルダ）
outdir = "random_data/A=0.05_k=0.4"
os.makedirs(outdir, exist_ok=True)

# v方向の分割
dv = 2*Vmax/(2*M - 1)
## v方向のindex
v_index = np.arange(-M, M, 1, dtype = float)
v = v_index*dv

# x方向の分割
L = 2*np.pi / k0
dx = L/N
x = np.arange(N)*dx

# フーリエ空間における波数ベクトル
k_vec = 2*np.pi*np.fft.fftfreq(N, d=dx)
# 初期条件
f0 = (1/np.sqrt(2*np.pi))*np.exp(-0.5*v**2)
cos_kx = np.cos(k*x)
f = (1 + A * cos_kx[:, None]) * f0[None, :]

# x方向への半ステップシフト
def shift_x_semi_lagrangian(f_in, v_arr, dt_half):
    F = np.fft.fft(f_in,axis=0)
    phase = np.exp(-1j * k_vec[:,None] * v_arr[None,:] * dt_half)
    F_shift = F*phase
    return np.fft.ifft(F_shift,axis=0).real

# Eの計算
def poisson_E_caluc(f_in):
    #fをvについて台形積分することでnを求める
    n = np.trapezoid(f_in,v,axis=1)
    #poisson方程式の右辺
    dn = 1-n
    dn_hat = np.fft.fft(dn)
    E_hat = np.zeros_like(dn_hat, dtype=complex)
    for i, kk in enumerate(k_vec):
        if kk != 0:
            E_hat[i] = dn_hat[i]/(1j*kk)
        else:
            E_hat[i] = 0
    E = np.fft.ifft(E_hat).real
    return E,dn,dn_hat

# v方向への1ステップシフト
def shift_v_lagrangian(f_in, E_x, dt_full):
    v_shift = E_x * dt_full
    f_out = np.zeros_like(f_in)

    for ix in range(f_in.shape[0]):
        cs = CubicSpline(v, f_in[ix,:],bc_type='natural',extrapolate=False)
        vv = v+v_shift[ix]
        fout = cs(vv)

        vmin, vmax = v[0], v[-1] + (v[1]-v[0]) #有効な速度範囲
        fout[~np.isfinite(fout)] = 0.0 #補間で出てきたNan,infを０に置き換える
        mask = (vv < vmin) | (vv > vmax)
        fout[mask] = 0.0 #補間点が有効な速度範囲を出ていたら0

        f_out[ix,:] = fout
    return f_out

# densityの計算
def density(f):
    return np.sum(f, axis=1) * dv

#速度uの計算
def velocity(f):
    j1=np.sum(f*v[None, :], axis=1) * dv
    return j1/density(f)

#圧力pの計算
def pressure(f):
    vc = v[None, :] - velocity(f)[:, None]
    p = np.sum(f * (vc**2), axis=1)*dv
    return p

#熱流速勾配の計算
def dq_dx(f):
    vc = v[None, :] - velocity(f)[:, None]
    q = np.sum(f * (vc**3), axis=1)*dv
    q_hat = np.fft.fft(q)
    dq_dx = np.fft.ifft(1j * k_vec *q_hat).real
    return dq_dx

# モード別のEの振幅の計算
def mode_amp(E, m, k0, k_vec):
    Ehat = np.fft.fft(E)/E.size
    target = m * k0
    jpos = np.argmin(np.abs(k_vec - (+target)))
    jneg = np.argmin(np.abs(k_vec - (-target)))
    # 実関数なので ±の振幅は等しいはず。数値誤差低減のため平均
    return 0.5 * (np.abs(Ehat[jpos]) + np.abs(Ehat[jneg]))

# 格納配列
t_history = []
n_history = []
u_history = []
p_history = []
dq_dx_history = []
E1_amp = []
E2_amp = []
E3_amp = []
Energy_history=[]
t = 0.0

nsteps = int(np.round(tmax/dt))
while t<tmax:
    E, dn, dn_k = poisson_E_caluc(f)

    E1 = mode_amp(E, 1, k, k_vec)   # 一次
    E2 = mode_amp(E, 2, k, k_vec)   # 二次
    E3 = mode_amp(E, 3, k, k_vec)   # 三次
    E1_amp.append(E1)
    E2_amp.append(E2)
    E3_amp.append(E3)
    t_history.append(t)
    Energy_history.append(0.5*np.trapezoid(E**2, x))
    n_history.append(density(f))
    u_history.append(velocity(f))
    p_history.append(pressure(f))
    dq_dx_history.append(dq_dx(f))

    #xを半分
    f = shift_x_semi_lagrangian(f, v, dt*0.5)

    #Eの計算
    E_half,_,_ = poisson_E_caluc(f)

    #v方向に進める
    f = shift_v_lagrangian(f, E_half, dt)

    #x方向に半分進める
    f = shift_x_semi_lagrangian(f, v, dt*0.5)

    t+=dt



"""
#file output
np.savetxt('machine_learning/training_data/n.txt', ml_data[0])
np.savetxt('machine_learning/training_data/u.txt', ml_data[1])
np.savetxt('machine_learning/training_data/p.txt', ml_data[2])
np.savetxt('machine_learning/training_data/dq_dx.txt', ml_data[3])
"""

Energy_history = np.array(Energy_history)/Energy_history[0]
t_history = np.asarray(t_history, dtype=float)
E1_amp = np.asarray(E1_amp, dtype=float)
E2_amp = np.asarray(E2_amp, dtype=float)
E3_amp = np.asarray(E3_amp, dtype=float)

Energy_history = Energy_history/Energy_history[0]

t_history = np.array(t_history)             # (Nt,)
n_history = np.array(n_history)             # (Nt, N)
u_history = np.array(u_history)             # (Nt, N)
p_history = np.array(p_history)             # (Nt, N)
dq_dx_history = np.array(dq_dx_history)     # (Nt, N)
Energy_history = np.array(Energy_history)   # (Nt,)




# まとめて .npz で保存（バイナリ。後で Python から読みやすい）
np.savez(
    f"{outdir}/moments_vlasov.npz",
    A_data=A * np.ones(len(t_history)),
    t=t_history,
    x=x,
    n=n_history,
    u=u_history,
    p=p_history,
    dq_dx=dq_dx_history,
    Energy=Energy_history,  # 正規化前のエネルギー
)

print(len(t_history))

def fit_gamma_envelope(t, y, tmin, tmax):
    t = np.asarray(t); y = np.asarray(y)
    # まず時間窓
    m = (t>=tmin)&(t<=tmax)&np.isfinite(y)&(y>0)
    tt, yy = t[m], y[m]
    # 極大だけ抽出（包絡線）
    idx = argrelextrema(yy, np.greater)[0]
    if len(idx) < 3:  # まれに極大が少ないときはそのまま回帰
        idx = np.arange(len(yy))
    tpk, Epk = tt[idx], yy[idx]
    # log 回帰
    (slope, intercept) = np.polyfit(tpk, np.log(Epk), 1)
    # 1σの推定
    yfit = intercept + slope*tpk
    resid = np.log(Epk) - yfit
    dof = max(1, len(tpk)-2)
    s2  = np.sum(resid**2)/dof
    X   = np.vstack([tpk, np.ones_like(tpk)]).T
    cov = s2 * np.linalg.inv(X.T@X)
    slope_std = np.sqrt(cov[0,0])
    return slope, slope_std, intercept

# plot
"""
plt.figure()
gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E1_amp,2.0,12.0)
t_fit = np.linspace(2.0, 12.0, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax = plt.gca()
ax.semilogy(t_fit, E_fit, 'k-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E1_amp,20.0,38.0)
t_fit = np.linspace(20.0, 38.0, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax.semilogy(t_fit, E_fit, 'r-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

ax.semilogy(t_history, E1_amp, lw=1)
ax.set_xlabel('t  [$\omega_{pe}^{-1}$]')
ax.set_ylabel('AMPLITUDE $E_1$')
ax.grid(True, which='both')
# Fig.4 相当：初期減衰(～15) と再成長(15～40)
plt.legend()
plt.savefig('picture/vlasov_A=0.15/E1_amp_with_gamma.png', dpi=150)

plt.figure()
ax = plt.gca()
#げんすい
gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E2_amp,3.0,8.0)
t_fit = np.linspace(3.0, 8.0, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax.semilogy(t_fit, E_fit, 'k-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E2_amp,26.0,36.0)
t_fit = np.linspace(26.0, 36.0, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax.semilogy(t_fit, E_fit, 'r-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

ax.semilogy(t_history, E2_amp, lw=1)
ax.set_xlabel(r't  [$\omega_{pe}^{-1}$]')
ax.set_ylabel(r'AMPLITUDE $E_2$')
ax.set_ylim(1e-6, 1)
ax.grid(True, which='both')
# Fig.4 相当：初期強い減衰→再成長
plt.legend()
plt.savefig('picture/vlasov_A=0.1_dt0002/E2_amp_with_gamma.png', dpi=150)

plt.figure()
ax = plt.gca()

gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E3_amp,1.6,6.5)
t_fit = np.linspace(1.6, 6.5, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax.semilogy(t_fit, E_fit, 'k-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

gamma, gamma_err, intercept = fit_gamma_envelope(t_history,E3_amp,26.0,40.0)
t_fit = np.linspace(26.0, 39.0, 200)
E_fit = np.exp(intercept + gamma * t_fit)  # 対数を戻して指数に
ax.semilogy(t_fit, E_fit, 'r-.', lw=2, label=f'{gamma:.4f}')
print(f"gamma = {gamma:.4f} ± {gamma_err:.4f} ")

ax.semilogy(t_history, E3_amp, lw=1)
ax.set_xlabel(r't  [$\omega_{pe}^{-1}$]')
ax.set_ylabel(r'AMPLITUDE $E_3$')
ax.set_ylim(1e-7, 1e-2)
ax.grid(True, which='both')
# Fig.4 相当：初期さらに強い減衰→再成長
plt.legend()
plt.savefig('picture/vlasov_A=0.1_dt0002/E3_amp_with_gamma.png', dpi=150)
"""

"""
# （参考）電場エネルギー
plt.figure()
plt.plot(t_history, Energy_history, lw=1.2)
plt.xlabel('t  [$\omega_{pe}^{-1}$]')
plt.ylabel('$\int E^2/2\,dx$  (norm.)')
plt.yscale('log'); plt.grid(True)
plt.savefig('picture/vlasov_A=0.105/Energy.png', dpi=150)

# ========= optional: space-time maps =========
def spacetime_plot(data, title, fname):
    plt.figure(figsize=(8,6))
    plt.imshow(np.array(data).T, aspect='auto', origin='lower', extent=[0, tmax, 0, L])
    plt.xlabel('t'); plt.ylabel('x'); plt.title(title)
    plt.colorbar(); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()

spacetime_plot(n_history, 'density', 'picture/vlasov_A=0.105/n.png')
spacetime_plot(u_history, 'velocity', 'picture/vlasov_A=0.105/u.png')
spacetime_plot(p_history, 'pressure', 'picture/vlasov_A=0.105/p.png')
spacetime_plot(dq_dx_history, 'gradient of q', 'picture/vlasov_A=0.105/dq_dx.png')
"""