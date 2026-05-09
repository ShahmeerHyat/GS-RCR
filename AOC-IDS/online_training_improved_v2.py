import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split
from utils_improved_v2 import (
    load_data, SplitData, setup_seed, AE, GSCRCLoss,
    UncertaintyWeightedLoss, evaluate, symmetric_kl_gmm,
    BOTTLENECK_DIMS, HIDDEN_DIMS, K_COMPONENTS
)

import argparse
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='GS-CRC v2 online training')
parser.add_argument("--dataset",          type=str,   default='nsl')
parser.add_argument("--epochs",           type=int,   default=4)
parser.add_argument("--epoch_1",          type=int,   default=1)
parser.add_argument("--percent",          type=float, default=0.8)
parser.add_argument("--flip_percent",     type=float, default=0.05)
parser.add_argument("--sample_interval",  type=int,   default=2000)
parser.add_argument("--cuda",             type=str,   default="0")
parser.add_argument("--bs",               type=int,   default=1024)
parser.add_argument("--no_compile",       action="store_true")
parser.add_argument("--no_amp",           action="store_true")
# GS-CRC hyperparameters
parser.add_argument("--gamma",       type=float, default=2.0)
parser.add_argument("--alpha_symm",  type=float, default=0.5)
parser.add_argument("--beta_temp",   type=float, default=0.5)
parser.add_argument("--l_max",       type=float, default=5.0)
# v2 new args
parser.add_argument("--k_abnorm",    type=int,   default=-1,
                    help="K for abnormal GMM. -1 = use K_COMPONENTS[dataset] default.")
parser.add_argument("--eps_drift",   type=float, default=1e-5,
                    help="KL-drift trigger threshold (Theorem 5.1). 0 = retrain every interval.")
parser.add_argument("--alpha_ema",   type=float, default=0.05,
                    help="EMA step for normal centroid update (Theorem 5.2). 0 = no EMA.")

args = parser.parse_args()
dataset         = args.dataset
epochs          = args.epochs
epoch_1         = args.epoch_1
percent         = args.percent
flip_percent    = args.flip_percent
sample_interval = args.sample_interval
cuda_num        = args.cuda

bs          = args.bs
seed        = 5009
seed_round  = 5
use_amp     = not args.no_amp and torch.cuda.is_available()
use_compile = not args.no_compile

k_abnorm = args.k_abnorm if args.k_abnorm > 0 else K_COMPONENTS.get(dataset, 1)
eps_drift = args.eps_drift
alpha_ema = args.alpha_ema

if dataset == 'nsl':
    input_dim = 121
else:
    input_dim = 196

bottleneck_dim = BOTTLENECK_DIMS.get(dataset)
hidden_dim     = HIDDEN_DIMS.get(dataset)

if dataset == 'nsl':
    KDDTrain = load_data("NSL_pre_data/PKDDTrain+.csv")
    KDDTest  = load_data("NSL_pre_data/PKDDTest+.csv")
    splitter = SplitData(dataset='nsl')
    x_train, y_train = splitter.transform(KDDTrain, labels='labels2')
    x_test,  y_test  = splitter.transform(KDDTest,  labels='labels2')
else:
    UNSWTrain = load_data("UNSW_pre_data/UNSWTrain.csv")
    UNSWTest  = load_data("UNSW_pre_data/UNSWTest.csv")
    splitter  = SplitData(dataset='unsw')
    x_train, y_train = splitter.transform(UNSWTrain, labels='label')
    x_test,  y_test  = splitter.transform(UNSWTest,  labels='label')

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test  = torch.FloatTensor(x_test)
y_test  = torch.LongTensor(y_test)

device = torch.device("cuda:" + cuda_num if torch.cuda.is_available() else "cpu")

print(f"Dataset: {dataset} | bottleneck: {bottleneck_dim} | hidden: {hidden_dim}")
print(f"GS-CRC: gamma={args.gamma}, alpha_symm={args.alpha_symm}, "
      f"beta_temp={args.beta_temp}, l_max={args.l_max}")
print(f"v2: k_abnorm={k_abnorm}, eps_drift={eps_drift}, alpha_ema={alpha_ema}")
print(f"Perf: bs={bs} | compile={'on' if use_compile else 'off'} | "
      f"BF16={'on' if use_amp else 'off'}")

criterion = GSCRCLoss(
    device,
    gamma=args.gamma,
    alpha_symm=args.alpha_symm,
    beta_temp=args.beta_temp,
    l_max=args.l_max,
)

for i in range(seed_round):
    setup_seed(seed + i)

    online_x_train, online_x_test, online_y_train, online_y_test = train_test_split(
        x_train, y_train, test_size=percent, random_state=seed + i)

    train_ds     = TensorDataset(online_x_train, online_y_train)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=bs, shuffle=True,
        pin_memory=True, num_workers=4, persistent_workers=True)

    model = AE(input_dim, bottleneck_dim=bottleneck_dim, hidden_dim=hidden_dim).to(device)

    if use_compile:
        model = torch.compile(model)

    uncertainty_loss = UncertaintyWeightedLoss().to(device)

    optimizer = torch.optim.SGD(
        list(model.parameters()) + list(uncertainty_loss.parameters()),
        lr=0.001
    )

    amp_ctx = (torch.autocast(device_type='cuda', dtype=torch.bfloat16)
               if use_amp else torch.autocast(device_type='cpu', enabled=False))

    # ---- Initial training ----
    model.train()
    for epoch in range(epochs):
        print(f'seed = {seed+i} , first round: epoch = {epoch}')
        for j, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with amp_ctx:
                features, recon_vec = model(inputs)
                loss_e = criterion(features,  labels)
                loss_d = criterion(recon_vec, labels)
                loss   = uncertainty_loss(loss_e, loss_d)

            loss.backward()
            optimizer.step()

    x_train_gpu  = x_train.to(device)
    x_test_gpu   = x_test.to(device)
    online_x_train = online_x_train.to(device)
    online_y_train = online_y_train.to(device)

    x_train_this_epoch  = online_x_train.clone()
    x_test_left_epoch   = online_x_test.clone().to(device)
    y_train_this_epoch  = online_y_train.clone()
    y_test_left_epoch_labels = online_y_test.clone()

    # ---- Compute initial normal centroid (EMA seed) ----
    with torch.no_grad():
        online_normal_idx    = torch.where(online_y_train.squeeze().cpu() == 0)[0].to(device)
        online_x_train_normal = torch.index_select(online_x_train, 0, online_normal_idx)
        normal_temp       = torch.mean(F.normalize(model(online_x_train_normal)[0], p=2, dim=1), dim=0)
        normal_recon_temp = torch.mean(F.normalize(model(online_x_train_normal)[1], p=2, dim=1), dim=0)

    # ---- Online training loop ----
    count = 0
    y_train_detection = online_y_train
    current_gmms = None  # will hold GMMs from last evaluate() call for KL-drift

    while len(x_test_left_epoch) > 0:
        print(f'seed = {seed+i} , i = {count}')
        count += 1

        if len(x_test_left_epoch) < sample_interval:
            x_test_this_epoch = x_test_left_epoch.clone()
            x_test_left_epoch = x_test_left_epoch[:0]
        else:
            x_test_this_epoch = x_test_left_epoch[:sample_interval].clone()
            x_test_left_epoch = x_test_left_epoch[sample_interval:]

        torch.cuda.empty_cache()

        # Predict pseudo-labels for current batch
        predict_result = evaluate(
            normal_temp, normal_recon_temp,
            x_train_this_epoch, y_train_detection,
            x_test_this_epoch, 0, model,
            k_abnorm=k_abnorm, return_gmms=True
        )
        y_test_pred_this_epoch = predict_result[0].numpy()
        batch_gmms = predict_result[1]

        y_train_detection = torch.cat([
            y_train_detection.to(device),
            torch.tensor(y_test_pred_this_epoch).to(device)
        ])

        # ---- EMA normal centroid update (Theorem 5.2) ----
        # Update centroid with predicted normals from the current batch
        if alpha_ema > 0:
            normal_pred_idx_np = np.where(y_test_pred_this_epoch == 0)[0]
            if len(normal_pred_idx_np) > 0:
                normal_pred_idx_t = torch.from_numpy(normal_pred_idx_np).long().to(device)
                conf_normal_x = torch.index_select(x_test_this_epoch, 0, normal_pred_idx_t)
                with torch.no_grad():
                    new_feat  = F.normalize(model(conf_normal_x)[0], p=2, dim=1)
                    new_recon = F.normalize(model(conf_normal_x)[1], p=2, dim=1)
                    batch_mu_en = new_feat.mean(0)
                    batch_mu_de = new_recon.mean(0)
                normal_temp       = (1 - alpha_ema) * normal_temp       + alpha_ema * batch_mu_en
                normal_recon_temp = (1 - alpha_ema) * normal_recon_temp + alpha_ema * batch_mu_de
                # Re-normalise to unit sphere
                normal_temp       = F.normalize(normal_temp.unsqueeze(0), p=2, dim=1).squeeze(0)
                normal_recon_temp = F.normalize(normal_recon_temp.unsqueeze(0), p=2, dim=1).squeeze(0)

        # ---- KL-drift trigger (Theorem 5.1) ----
        # Decide whether to retrain the model this round
        should_retrain = True  # default: always retrain

        if eps_drift > 0 and current_gmms is not None:
            # Compute symmetric KL between last GMM and current GMM
            kl_en = symmetric_kl_gmm(current_gmms['gmm_en_abnorm'], batch_gmms['gmm_en_abnorm'])
            kl_de = symmetric_kl_gmm(current_gmms['gmm_de_abnorm'], batch_gmms['gmm_de_abnorm'])
            drift = 0.5 * (kl_en + kl_de)
            print(f'  KL drift = {drift:.2e} (threshold = {eps_drift:.2e}) '
                  f'| retrain = {drift > eps_drift}')
            should_retrain = (drift > eps_drift)

        current_gmms = batch_gmms  # always update stored GMMs

        # Controlled noise injection (limits confirmation-bias drift)
        num_flip = int(flip_percent * y_test_pred_this_epoch.shape[0])
        flip_idx = np.random.choice(y_test_pred_this_epoch.shape[0], num_flip, replace=False)
        y_test_pred_this_epoch[flip_idx] = 1 - y_test_pred_this_epoch[flip_idx]

        x_train_this_epoch = torch.cat([
            x_train_this_epoch.to(device),
            x_test_this_epoch.to(device)
        ])
        y_train_this_epoch = torch.cat([
            y_train_this_epoch.to(device),
            torch.tensor(y_test_pred_this_epoch).to(device)
        ])

        if should_retrain:
            train_ds     = TensorDataset(x_train_this_epoch, y_train_this_epoch)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)

            model.train()
            for epoch in range(epoch_1):
                print(f'epoch = {epoch}')
                for inputs, labels in train_loader:
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad()

                    with amp_ctx:
                        features, recon_vec = model(inputs)
                        loss_e = criterion(features,  labels)
                        loss_d = criterion(recon_vec, labels)
                        loss   = uncertainty_loss(loss_e, loss_d)

                    loss.backward()
                    optimizer.step()
        else:
            print(f'  Skipping model retrain (drift below threshold)')

    # ---- Final evaluation ----
    torch.cuda.empty_cache()
    with torch.no_grad():
        online_normal_idx     = torch.where(online_y_train.squeeze().cpu() == 0)[0].to(device)
        online_x_train_normal = torch.index_select(online_x_train, 0, online_normal_idx)
        normal_temp_final       = torch.mean(F.normalize(model(online_x_train_normal)[0], p=2, dim=1), dim=0)
        normal_recon_temp_final = torch.mean(F.normalize(model(online_x_train_normal)[1], p=2, dim=1), dim=0)

    res_en, res_de, res_final = evaluate(
        normal_temp_final, normal_recon_temp_final,
        x_train_this_epoch, y_train_detection,
        x_test_gpu, y_test, model,
        k_abnorm=k_abnorm, return_gmms=False
    )
