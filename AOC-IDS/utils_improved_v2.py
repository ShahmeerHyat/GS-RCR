import torch
import numpy as np
import random
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import scipy.optimize as opt
import torch.distributions as dist
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.mixture import GaussianMixture


def load_data(data_path):
    return pd.read_csv(data_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class SplitData(BaseEstimator, TransformerMixin):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def fit(self, X, y=None):
        return self

    def transform(self, X, labels, one_hot_label=True):
        if self.dataset == 'nsl':
            y = X[labels]
            X_ = X.drop(['labels5', 'labels2'], axis=1)
            y = (y != 'normal')
            y_ = np.asarray(y).astype('float32')
        elif self.dataset == 'unsw':
            y_ = X[labels]
            X_ = X.drop('label', axis=1)
        else:
            raise ValueError("Unsupported dataset type")
        normalize = MinMaxScaler().fit(X_)
        return normalize.transform(X_), y_


# Optimal bottleneck: m* = ceil(d* + log2(K)), d*=intrinsic dim, K=attack subclasses
BOTTLENECK_DIMS = {'nsl': 21, 'unsw': 32}
HIDDEN_DIMS     = {'nsl': 64, 'unsw': 128}
# K-component GMM: K = number of attack subtypes (Theorem 3.1)
K_COMPONENTS    = {'nsl': 5,  'unsw': 10}


class AE(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=None, hidden_dim=None):
        super().__init__()
        if bottleneck_dim is None:
            nearest = 2 ** round(math.log2(input_dim))
            hidden_dim     = nearest // 2
            bottleneck_dim = nearest // 4
        elif hidden_dim is None:
            hidden_dim = bottleneck_dim * 4

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode


class GSCRCLoss(nn.Module):
    """GS-CRC loss — identical to v1, no changes needed here."""
    def __init__(self, device, gamma=2.0, alpha_symm=0.5, beta_temp=0.5,
                 l_max=5.0, tau_min=0.01, tau_max=0.5, l_a_min_frac=0.2):
        super().__init__()
        self.device = device
        self.gamma = gamma
        self.alpha_symm = alpha_symm
        self.beta_temp = beta_temp
        self.l_max = l_max
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.l_a_min_frac = l_a_min_frac

    def _estimate_tau(self, sim_na, l_n, l_a):
        sigma_h = sim_na.detach().std().item()
        if sigma_h < 1e-8:
            return self.tau_min
        l_a_min = max(1, int(self.l_a_min_frac * l_a))
        log_ratio = math.log(max(l_n * l_a / l_a_min, 2.0))
        log_product = math.log(max(l_n * l_a, 2.0))
        tau = (sigma_h / (1.0 + log_ratio)) * math.sqrt(2.0 * log_product)
        return float(np.clip(tau, self.tau_min, self.tau_max))

    def _anchor_loss(self, sim_pp, sim_pn, tau_0, rho):
        l_p = sim_pp.shape[0]
        l_n = sim_pn.shape[1]
        if l_p < 2 or l_n < 1:
            return torch.tensor(0.0, device=self.device)

        h_bar_pos = sim_pp.mean().detach()
        h_bar_neg = sim_pn.mean().detach()

        tau_pp = (tau_0 * (1.0 + self.beta_temp * (sim_pp - h_bar_pos).abs())).clamp(min=self.tau_min)
        tau_pn = (tau_0 * (1.0 + self.beta_temp * (sim_pn - h_bar_neg).abs())).clamp(min=self.tau_min)

        scaled_pp = sim_pp / tau_pp
        diag_mask = torch.eye(l_p, device=self.device) * 1e9
        scaled_pp = scaled_pp - diag_mask

        log_sum_pos = torch.logsumexp(scaled_pp, dim=1)

        neg_softmax = F.softmax(sim_pn / tau_0, dim=1)
        focal_log_w = self.gamma * torch.log((1.0 - neg_softmax).clamp(min=1e-8))

        scaled_pn = sim_pn / tau_pn
        log_neg_terms = focal_log_w + scaled_pn
        log_sum_neg = torch.logsumexp(log_neg_terms, dim=1)

        log_rho_neg = math.log(max(rho, 1e-8)) + log_sum_neg
        log_denom = torch.logaddexp(log_sum_pos, log_rho_neg)
        loss = (log_denom - log_sum_pos).clamp(max=self.l_max)

        return loss.mean()

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        labels_cpu = labels.squeeze().cpu()

        normal_idx   = torch.where(labels_cpu == 0)[0].to(self.device)
        abnormal_idx = torch.where(labels_cpu >  0)[0].to(self.device)
        l_n = len(normal_idx)
        l_a = len(abnormal_idx)

        if l_n < 2 or l_a < 1:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        z_n = torch.index_select(features, 0, normal_idx)
        z_a = torch.index_select(features, 0, abnormal_idx)

        sim_nn = torch.mm(z_n, z_n.T)
        sim_na = torch.mm(z_n, z_a.T)
        sim_aa = torch.mm(z_a, z_a.T)

        tau_0 = self._estimate_tau(sim_na, l_n, l_a)

        rho_n = l_n / l_a
        loss_normal = self._anchor_loss(sim_nn, sim_na, tau_0, rho_n)

        if self.alpha_symm > 0 and l_a >= 2:
            rho_a = l_a / l_n
            loss_abnormal = self._anchor_loss(sim_aa, sim_na.T, tau_0, rho_a)
        else:
            loss_abnormal = torch.tensor(0.0, device=self.device)

        return loss_normal + self.alpha_symm * loss_abnormal


class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_sigma_e = nn.Parameter(torch.zeros(1))
        self.log_sigma_d = nn.Parameter(torch.zeros(1))

    def forward(self, loss_e, loss_d):
        sigma_e_sq = torch.exp(2.0 * self.log_sigma_e)
        sigma_d_sq = torch.exp(2.0 * self.log_sigma_d)
        return (loss_e / (2.0 * sigma_e_sq)
                + loss_d / (2.0 * sigma_d_sq)
                + self.log_sigma_e
                + self.log_sigma_d)


def score_detail(y_test, y_test_pred, if_print=False):
    if if_print:
        print("Confusion matrix")
        print(confusion_matrix(y_test, y_test_pred))
        print('Accuracy ',  accuracy_score(y_test, y_test_pred))
        print('Precision ', precision_score(y_test, y_test_pred, zero_division=0))
        print('Recall ',    recall_score(y_test, y_test_pred, zero_division=0))
        print('F1 score ',  f1_score(y_test, y_test_pred, zero_division=0))
    return (accuracy_score(y_test, y_test_pred),
            precision_score(y_test, y_test_pred, zero_division=0),
            recall_score(y_test, y_test_pred, zero_division=0),
            f1_score(y_test, y_test_pred, zero_division=0))


def _fit_gmm_v2(scores_normal, scores_abnorm, k_max=10):
    """
    v2 discriminator fitting (Theorem 3.1):
      - Normal class  : single Gaussian (scores concentrate near centroid)
      - Abnormal class: K-component GMM, K selected by BIC up to k_max

    Returns (g_normal, gmm_abnorm) where
      g_normal   : torch.distributions.Normal
      gmm_abnorm : sklearn GaussianMixture (use .score_samples(X) for log-likelihood)
    """
    # Normal: single Gaussian
    mu_n = float(np.mean(scores_normal))
    s_n  = max(float(np.std(scores_normal)), 1e-6)
    g_normal = dist.Normal(mu_n, s_n, validate_args=False)

    # Abnormal: BIC-selected K-GMM
    X_a = np.asarray(scores_abnorm, dtype=np.float64).reshape(-1, 1)
    k_max_safe = min(k_max, max(1, len(X_a) // 5))  # need >= 5 pts per component

    best_bic = np.inf
    best_gmm = None

    for k in range(1, k_max_safe + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                   max_iter=200, n_init=2, random_state=42)
            gmm.fit(X_a)
            bic = gmm.bic(X_a)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm
        except Exception:
            break

    if best_gmm is None:
        best_gmm = GaussianMixture(n_components=1, covariance_type='full',
                                    max_iter=200, random_state=42)
        best_gmm.fit(X_a)

    return g_normal, best_gmm


def symmetric_kl_gmm(gmm_p, gmm_q, n_samples=4000):
    """
    Estimate symmetric KL divergence between two sklearn GMMs by Monte Carlo.
    Used for KL-drift trigger (Theorem 5.1).
    Returns 0.0 if either GMM cannot sample (degenerate).
    """
    try:
        X_p, _ = gmm_p.sample(n_samples)
        X_q, _ = gmm_q.sample(n_samples)
        kl_pq = float(np.mean(gmm_p.score_samples(X_p) - gmm_q.score_samples(X_p)))
        kl_qp = float(np.mean(gmm_q.score_samples(X_q) - gmm_p.score_samples(X_q)))
        return max(0.0, 0.5 * (kl_pq + kl_qp))
    except Exception:
        return 0.0


def compute_score_gmms(normal_temp, normal_recon_temp, x_train, y_train, model, k_abnorm):
    """
    Compute cosine similarity scores on training data and fit K-GMMs.
    Lightweight call (no gradient, no full evaluate): used for KL-drift check.

    Returns dict with keys:
      'g_en_normal', 'gmm_en_abnorm', 'g_de_normal', 'gmm_de_abnorm'
    """
    model.eval()
    with torch.no_grad():
        y_cpu = y_train.squeeze().cpu()
        normal_idx   = torch.where(y_cpu == 0)[0].to(x_train.device)
        abnormal_idx = torch.where(y_cpu  > 0)[0].to(x_train.device)

        if len(normal_idx) < 2 or len(abnormal_idx) < 2:
            return None

        x_n = torch.index_select(x_train, 0, normal_idx)
        x_a = torch.index_select(x_train, 0, abnormal_idx)

        ref_en = normal_temp.reshape(1, -1)
        ref_de = normal_recon_temp.reshape(1, -1)

        feat_n   = F.normalize(model(x_n)[0], p=2, dim=1)
        feat_a   = F.normalize(model(x_a)[0], p=2, dim=1)
        recon_n  = F.normalize(model(x_n)[1], p=2, dim=1)
        recon_a  = F.normalize(model(x_a)[1], p=2, dim=1)

        sc_en_n = F.cosine_similarity(feat_n,  ref_en).cpu().numpy()
        sc_en_a = F.cosine_similarity(feat_a,  ref_en).cpu().numpy()
        sc_de_n = F.cosine_similarity(recon_n, ref_de).cpu().numpy()
        sc_de_a = F.cosine_similarity(recon_a, ref_de).cpu().numpy()

    model.train()

    g_en_n, gmm_en_a = _fit_gmm_v2(sc_en_n, sc_en_a, k_max=k_abnorm)
    g_de_n, gmm_de_a = _fit_gmm_v2(sc_de_n, sc_de_a, k_max=k_abnorm)

    return {
        'g_en_normal':  g_en_n,
        'gmm_en_abnorm': gmm_en_a,
        'g_de_normal':  g_de_n,
        'gmm_de_abnorm': gmm_de_a,
    }


def evaluate(normal_temp, normal_recon_temp, x_train, y_train, x_test, y_test, model,
             k_abnorm=1, return_gmms=False):
    """
    v2 evaluate: uses K-component GMM for abnormal class (Theorem 3.1).
    LLR fusion with class priors (Proposition 3.2) — same as v1.

    New params vs v1:
      k_abnorm    : max K for BIC-selected abnormal GMM (1 = same as v1)
      return_gmms : if True, append fitted GMM dict to return value
    """
    model.eval()
    with torch.no_grad():
        y_train_cpu = y_train.squeeze().cpu()
        normal_idx   = torch.where(y_train_cpu == 0)[0].to(x_train.device)
        abnormal_idx = torch.where(y_train_cpu == 1)[0].to(x_train.device)
        x_train_normal   = torch.index_select(x_train, 0, normal_idx)
        x_train_abnormal = torch.index_select(x_train, 0, abnormal_idx)

        # --- Encoder branch ---
        feat_normal  = F.normalize(model(x_train_normal)[0],   p=2, dim=1)
        feat_abnorm  = F.normalize(model(x_train_abnormal)[0], p=2, dim=1)
        test_feat    = F.normalize(model(x_test)[0],           p=2, dim=1)

        ref_en = normal_temp.reshape(1, -1)
        scores_normal_en = F.cosine_similarity(feat_normal,  ref_en).cpu().detach().numpy()
        scores_abnorm_en = F.cosine_similarity(feat_abnorm,  ref_en).cpu().detach().numpy()
        scores_test_en   = torch.nan_to_num(
            F.cosine_similarity(test_feat, ref_en), nan=0.0).cpu()

        # --- Decoder branch ---
        recon_normal = F.normalize(model(x_train_normal)[1],   p=2, dim=1)
        recon_abnorm = F.normalize(model(x_train_abnormal)[1], p=2, dim=1)
        test_recon   = F.normalize(model(x_test)[1],           p=2, dim=1)

        ref_de = normal_recon_temp.reshape(1, -1)
        scores_normal_de = F.cosine_similarity(recon_normal, ref_de).cpu().detach().numpy()
        scores_abnorm_de = F.cosine_similarity(recon_abnorm, ref_de).cpu().detach().numpy()
        scores_test_de   = torch.nan_to_num(
            F.cosine_similarity(test_recon, ref_de), nan=0.0).cpu()

    model.train()

    # Fit K-component GMMs (v2) — K_abnorm components for abnormal class
    g_en_normal, gmm_en_abnorm = _fit_gmm_v2(scores_normal_en, scores_abnorm_en, k_max=k_abnorm)
    g_de_normal, gmm_de_abnorm = _fit_gmm_v2(scores_normal_de, scores_abnorm_de, k_max=k_abnorm)

    # Class prior: log(pi_n / pi_a) from training labels
    y_np = y_train_cpu.numpy()
    pi_a = max(1e-6, float(np.mean(y_np > 0)))
    pi_n = 1.0 - pi_a
    prior_llr = math.log(pi_n / pi_a)

    # LLR fusion — now using K-GMM for abnormal log-likelihood
    sc_en_np = scores_test_en.numpy().reshape(-1, 1).astype(np.float64)
    sc_de_np = scores_test_de.numpy().reshape(-1, 1).astype(np.float64)

    log_p_en_normal = g_en_normal.log_prob(scores_test_en).numpy()
    log_p_en_abnorm = gmm_en_abnorm.score_samples(sc_en_np)       # K-GMM log-likelihood
    llr_en = torch.from_numpy((log_p_en_normal - log_p_en_abnorm).astype(np.float32))

    log_p_de_normal = g_de_normal.log_prob(scores_test_de).numpy()
    log_p_de_abnorm = gmm_de_abnorm.score_samples(sc_de_np)
    llr_de = torch.from_numpy((log_p_de_normal - log_p_de_abnorm).astype(np.float32))

    combined_llr = llr_en + llr_de + prior_llr
    y_pred = (combined_llr < 0.0).numpy().astype("int32")

    gmms = {
        'g_en_normal':   g_en_normal,
        'gmm_en_abnorm': gmm_en_abnorm,
        'g_de_normal':   g_de_normal,
        'gmm_de_abnorm': gmm_de_abnorm,
    }

    # Online mode: return predictions only
    if isinstance(y_test, int) or (hasattr(y_test, '__len__') and len(y_test) == 0):
        pred_tensor = torch.from_numpy(y_pred)
        return (pred_tensor, gmms) if return_gmms else pred_tensor

    if hasattr(y_test, 'cpu'):
        y_test = y_test.cpu().numpy()

    result_encoder = score_detail(y_test, (llr_en < 0.0).numpy().astype("int32"))
    result_decoder = score_detail(y_test, (llr_de < 0.0).numpy().astype("int32"))
    result_final   = score_detail(y_test, y_pred, if_print=True)

    if return_gmms:
        return result_encoder, result_decoder, result_final, gmms
    return result_encoder, result_decoder, result_final
