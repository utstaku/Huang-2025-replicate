from cProfile import label
import numpy as np
import matplotlib.pyplot as plt
import cmocean

k = 0.35
tmax = 40
L = 2 * np.pi / k

# === 改良版 spacetime_plot ===
def spacetime_plot(ax, data, title="", cmap="RdBu_r", vmin=None, vmax=None):
    im = ax.imshow(np.array(data).T, aspect='auto', origin='lower',
                   extent=[0, tmax, 0, L], cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_title(title, fontsize=10, pad=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

"""
# ---- データのロード ----
"""
vlasov_data = np.load('raw_data/vlasov_A=0.1/moments_test.npz')
mlc_data    = np.load('raw_data/ml_closure_A=0.1/moments.npz')
hp_data     = np.load('raw_data/hp_closure_A=0.1/moments.npz')

t_v = vlasov_data["t"]; n_v, u_v, p_v, dq_v, Eenergy_v = \
    vlasov_data["n"], vlasov_data["u"], vlasov_data["p"], vlasov_data["dq_dx"], vlasov_data["Energy"]
t_m = mlc_data["t"]; n_m, u_m, p_m, dq_m, Eenergy_m = \
    mlc_data["n"], mlc_data["u"], mlc_data["p"], mlc_data["dq_dx"], mlc_data["Energy"]
t_h = hp_data["t"]; n_h, u_h, p_h, dq_h, Eenergy_h = \
    hp_data["n"], hp_data["u"], hp_data["p"], hp_data["dq_dx"], hp_data["Energy"]


# === Fig.2: dq/dx 比較 ===
fig, axs = plt.subplots(3, 1, figsize=(8, 8))
spacetime_plot(axs[0], dq_v, "Vlasov: ∂q/∂x (Ground Truth)")
spacetime_plot(axs[1], dq_m, "Fluid + ML (FNO): ∂q/∂x Prediction")
spacetime_plot(axs[2], np.abs(dq_v - dq_m), "Absolute Error |Δ(∂q/∂x)|", cmap=cmocean.cm.balance)
fig.suptitle("Comparison of Heat-Flux Gradient ∂q/∂x", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig("picture/comparing/A=0.1_k=0.35_ml/dq_dx.png", dpi=300)
plt.close()

# === Fig.3: 電場エネルギー ===
fig, axs = plt.subplots(2, 1, figsize=(7,6))
axs[0].plot(t_v, Eenergy_v, 'k-', label='Vlasov')
axs[0].plot(t_h, Eenergy_h, 'r--', label='Fluid + HP')
axs[0].set_yscale('log')
axs[0].set_ylabel('Electric field energy')
axs[0].set_title('Nonlinear Landau damping: HP closure vs Vlasov')
axs[0].legend()

axs[1].plot(t_v, Eenergy_v, 'k-', label='Vlasov')
axs[1].plot(t_m, Eenergy_m, 'r--', label='Fluid + ML (FNO)')
axs[1].set_yscale('log')
axs[1].set_xlabel('t')
axs[1].set_ylabel('Electric field energy')
axs[1].set_title('Nonlinear Landau damping: ML closure vs Vlasov')
axs[1].legend()

fig.suptitle("Electric Field Energy Evolution", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('picture/comparing/A=0.1_k=0.35_ml/Eenergy_compare.png', dpi=300)
plt.close()

# === Fig.5: n,u,p の比較 ===
fig, axs = plt.subplots(3, 3, figsize=(9, 8))
titles = [["Vlasov", "Fluid + ML", "Fluid + HP"],
          ["Vlasov", "Fluid + ML", "Fluid + HP"],
          ["Vlasov", "Fluid + ML", "Fluid + HP"]]
labels = ["Density n", "Velocity u", "Pressure p"]
data_sets = [(n_v, n_m, n_h), (u_v, u_m, u_h), (p_v, p_m, p_h)]



for i in range(3):
    for j in range(3):
        spacetime_plot(axs[i,j], data_sets[i][j], f"{titles[i][j]}\n{labels[i]}",cmap=cmocean.cm.balance)
fig.suptitle("Comparison of Fluid Moments (n, u, p)", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('picture/comparing/A=0.1_k=0.35_ml/moments_compare.png', dpi=300)
plt.close()



# === Fig.6: 絶対誤差 ===
fig, axs = plt.subplots(3, 2, figsize=(8, 7))
quantities = ['Density n', 'Velocity u', 'Pressure p']
for i, (v, m, h, qname) in enumerate(zip([n_v,u_v,p_v],[n_m,u_m,p_m],[n_h,u_h,p_h], quantities)):
    spacetime_plot(axs[i,0], np.abs(v-m), f'ML Error: {qname}', cmap=cmocean.cm.balance)
    spacetime_plot(axs[i,1], np.abs(v-h), f'HP Error: {qname}', cmap=cmocean.cm.balance)
fig.suptitle("Absolute Errors of Fluid Moments relative to Vlasov", fontsize=12, y=0.98)
plt.tight_layout(rect=[0,0,1,0.97])
plt.savefig('picture/comparing/A=0.1_k=0.35_ml/abserror.png', dpi=300)
plt.close()
