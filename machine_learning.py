from neuralop.models import FNO
from neuralop.training import AdamW
from neuralop.utils import count_model_params
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import Dataset, Subset
import torch
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.interpolate import interp1d
from os.path import dirname, join as pjoin
import os
import io
import scipy.io as sio
import urllib
from neuralop import Trainer
from neuralop.losses.data_losses import LpLoss

# training data
train_data = np.load("raw_data/vlasov_A=0.1/moments_training.npz")
n_data = train_data["n"]
u_data = train_data["u"]
p_data = train_data["p"]
dq_dx_data = train_data["dq_dx"]

# test data
test_data = np.load("raw_data/vlasov_A=0.1/moments_test.npz")
n_data_test = test_data["n"]
u_data_test = test_data["u"]
p_data_test = test_data["p"]
dq_dx_data_test = test_data["dq_dx"]

# parameters
batch_size = 32 # バッチサイズ
n_train = len(n_data) # 学習に使用するデータ数
n_test = len(n_data_test) # テストに使用するデータ数
num_epoch = 10 # エポック数
num_modes = 16 # フーリエ空間で使用するモードの数
num_channels = 64 # インプットとアウトプットの間の層の数
in_channels = 3 # インプット数
device = 'cpu' # use GPU if available
print("Using device:", device)
outdir = "machine_learning"
os.makedirs(outdir, exist_ok=True)

# making dataset and dataloader
def dict_collate_fn(batch):
    xs, ys = zip(*batch)
    return {"x": torch.stack(xs).to(device), "y": torch.stack(ys).to(device)}

def create_dataset_and_loader_train(n_train, batch_size):

    n = torch.tensor(n_data, dtype=torch.float32)[0:n_train]
    u = torch.tensor(u_data, dtype=torch.float32)[0:n_train]
    p = torch.tensor(p_data, dtype=torch.float32)[0:n_train]
    dq_dx = torch.tensor(dq_dx_data, dtype=torch.float32)[0:n_train]
    N = dq_dx.shape[-1]
    x_train = torch.zeros((n_train, 3, N))
    y_train = torch.zeros((n_train, 1, N))

    x_train[:, 0, :] =(n-n.mean())/n.std()
    x_train[:, 1, :] =(u-u.mean())/u.std()
    x_train[:, 2, :] =(p-p.mean())/p.std()
    y_train[:, 0, :] =(dq_dx-dq_dx.mean())/dq_dx.std()
    np.savez("machine_learning/scaler.npz", mu_n=n.mean(), sig_n=n.std(), mu_u=u.mean(), sig_u=u.std(), mu_p=p.mean(), sig_p=p.std(), mu_dq=dq_dx.mean(), sig_dq=dq_dx.std())
    print("length of x_train[:,0,:] = ",len(x_train[:,0,:]))
    print("shape of x_train[:,0,:] = ",x_train[:,0,:].shape)
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        collate_fn=dict_collate_fn,
    )
    return dataset, loader

def create_dataset_and_loader_test(n_test, batch_size):
    n = torch.tensor(n_data_test, dtype=torch.float32)[0:n_test]
    u = torch.tensor(u_data_test, dtype=torch.float32)[0:n_test]
    p = torch.tensor(p_data_test, dtype=torch.float32)[0:n_test]
    dq_dx = torch.tensor(dq_dx_data_test, dtype=torch.float32)[0:n_test]
    N = dq_dx.shape[-1]
    x_test = torch.zeros((n_test, 3, N))
    y_test = torch.zeros((n_test, 1, N))

    scaler = np.load("machine_learning/scaler.npz")
    mu_n, sig_n = scaler["mu_n"], scaler["sig_n"]
    mu_u, sig_u = scaler["mu_u"], scaler["sig_u"]
    mu_p, sig_p = scaler["mu_p"], scaler["sig_p"]
    mu_dq, sig_dq = scaler["mu_dq"], scaler["sig_dq"]

    x_test[:, 0, :] =(n-mu_n)/sig_n
    x_test[:, 1, :] =(u-mu_u)/sig_u
    x_test[:, 2, :] =(p-mu_p)/sig_p
    y_test[:, 0, :] =(dq_dx-mu_dq)/sig_dq
    print("length of x_test[:,0,:] = ",len(x_test[:,0,:]))
    print("shape of x_test[:,0,:] = ",x_test[:,0,:].shape)
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        persistent_workers=False,
        collate_fn=dict_collate_fn,
    )
    return dataset, loader

# loss function
if 1:
    def l2loss(pred, **sample):
        criterion = torch.nn.MSELoss()
        return criterion(pred, sample["y"])

else:
    #
    l2loss = LpLoss(d=1, p=2, reduction="mean")


# model
train_dataset, train_loader = create_dataset_and_loader_train(
    n_train, batch_size
)
test_dataset, test_loader = create_dataset_and_loader_test(
    n_test, batch_size
)
test_loaders = {'test': test_loader }

model = FNO(
    n_modes=(num_modes,), n_layers=4,hidden_channels=num_channels, in_channels=in_channels, out_channels=1
).to(device)

optimizer = AdamW(model.parameters(), lr=5e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

train_loss = l2loss
eval_losses = {"l2": train_loss}

print("\n### MODEL ###\n", model)
print("\n### OPTIMIZER ###\n", optimizer)
print("\n### SCHEDULER ###\n", scheduler)
print("\n### LOSSES ###")
print(f"\n * Train: {train_loss}")
print(f"\n * Test: {eval_losses}")
sys.stdout.flush()

# trainer
trainer = Trainer(
    model=model,
    n_epochs=num_epoch,
    device=device,
    wandb_log=False,
    eval_interval=1,
    use_distributed=False,
    verbose=True,
)


trainer.train(train_loader=train_loader,
              test_loaders=test_loaders,
              optimizer=optimizer,
              scheduler=scheduler,
              regularizer=False,
              training_loss=train_loss,
              eval_losses=eval_losses)


# save model
# データセットのheat flux
q_true = train_dataset.tensors[1].cpu().numpy()
# MLで予想したheat flux
q_pred = model(train_dataset.tensors[0].to(device))
q_pred = q_pred.detach().cpu().numpy()
#チェック
print("length of q_pred = ",len(q_pred))
print("shape of q_pred = ",q_pred.shape)
#学習済みのモデルの保存
torch.save(model.to(device).state_dict(), "machine_learning/learnedmodel_from_vlasov.pth")

# plot
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# --- Ground Truth ---
im0 = axs[0].imshow(q_true[:, 0, :], aspect="auto", origin="lower")
axs[0].set_title("Ground Truth")
fig.colorbar(im0, ax=axs[0])

# --- Model Prediction ---
im1 = axs[1].imshow(q_pred[:, 0, :], aspect="auto", origin="lower")
axs[1].set_title("Model Prediction")
fig.colorbar(im1, ax=axs[1])

# --- Error ---
error = np.abs(q_true - q_pred)
im2 = axs[2].imshow(error[:, 0, :], aspect="auto", origin="lower")
axs[2].set_title("Error")
fig.colorbar(im2, ax=axs[2])

# --- Label all axes ---
for ax in axs:
    ax.set_xlabel("x")
    ax.set_ylabel("step")

plt.tight_layout()
plt.show()
