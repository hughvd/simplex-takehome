"""
Analysis functions for the trained Mess3 transformer.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm

from mess3 import (
    COMPONENT_PARAMS, COMPONENT_WEIGHTS,
    generate_sequences, compute_belief_states, make_observation_matrices,
)
from train import get_device


def collect_activations(model, n_seq=10000, batch_size=256, device=None):
    """Collect activations and beliefs for evaluation sequences."""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()
    model.store_activations = True

    seq_len = model.config.context_length
    rng = np.random.default_rng(0)

    all_sequences, all_component_ids = [], []
    all_activations = {k: [] for k in ['post_embed', 'post_attn', 'post_mlp', 'post_ln']}
    all_attn_weights = []

    n_batches = (n_seq + batch_size - 1) // batch_size
    for _ in tqdm(range(n_batches), desc='Collecting activations'):
        seqs, comp_ids = generate_sequences(batch_size, seq_len, rng=rng)
        all_sequences.append(seqs)
        all_component_ids.append(comp_ids)

        tokens = torch.tensor(seqs, dtype=torch.long, device=device)
        with torch.no_grad():
            _, acts = model(tokens)

        for k in ['post_embed', 'post_attn', 'post_mlp', 'post_ln']:
            all_activations[k].append(acts[k].cpu().numpy())
        all_attn_weights.append(acts['attn_weights'].cpu().numpy())

    sequences = np.concatenate(all_sequences)[:n_seq]
    component_ids = np.concatenate(all_component_ids)[:n_seq]
    activations = {k: np.concatenate(v)[:n_seq] for k, v in all_activations.items()}
    attn_weights = np.concatenate(all_attn_weights)[:n_seq]
    beliefs = compute_belief_states(sequences, component_ids)

    model.store_activations = False

    return {
        'sequences': sequences,
        'component_ids': component_ids,
        'beliefs': beliefs,
        'activations': activations,
        'attn_weights': attn_weights,
    }


def pca_analysis(activations, n_components=20):
    """PCA of activations at each position. Returns variance ratios."""
    act = activations['post_mlp']
    N, T, d = act.shape

    results = {'variance_ratio': [], 'cumulative_variance': []}
    for t in range(T):
        pca = PCA(n_components=min(n_components, d))
        pca.fit(act[:, t, :])
        results['variance_ratio'].append(pca.explained_variance_ratio_)
        results['cumulative_variance'].append(np.cumsum(pca.explained_variance_ratio_))

    return results


def linear_regression_beliefs(activations, beliefs, component_ids):
    """
    Ridge regression from activations to belief representations.
    Returns dict of R^2 values per position.
    """
    act = activations['post_mlp']
    N, T, d = act.shape

    n_train = int(0.8 * N)
    idx = np.random.default_rng(42).permutation(N)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    targets = {
        'joint': beliefs['joint'],
        'within_component': beliefs['within_component'],
        'component_posterior': beliefs['component_posterior'],
    }

    results = {}
    for name, target in targets.items():
        r2_per_pos = []
        for t in range(T):
            reg = Ridge(alpha=10.0)
            reg.fit(act[train_idx, t, :], target[train_idx, t, :])
            y_pred = reg.predict(act[test_idx, t, :])
            r2 = r2_score(target[test_idx, t, :], y_pred, multioutput='uniform_average')
            r2_per_pos.append(r2)
        results[name] = np.array(r2_per_pos)

    return results


# ---------- Simplex helpers ----------

def barycentric_to_cartesian(coords):
    """Convert barycentric coordinates (N, 3) to 2D cartesian for equilateral triangle."""
    v0 = np.array([0, 0])
    v1 = np.array([1, 0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    return coords[:, 0:1] * v0 + coords[:, 1:2] * v1 + coords[:, 2:3] * v2


def draw_simplex_outline(ax):
    """Draw the 2-simplex triangle."""
    v = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2], [0, 0]])
    ax.plot(v[:, 0], v[:, 1], 'k-', linewidth=0.5)
    ax.set_aspect('equal')
    ax.axis('off')


def rgb_from_belief(beliefs):
    """Map 3D belief to RGB color."""
    return np.clip(beliefs, 0, 1)


# ---------- Per-component theoretical loss ----------

def compute_per_component_theoretical_loss(seq_len=12, n_samples=10000):
    """
    Bayes-optimal CE per component (oracle predictor that knows the component).

    Aligned with model convention: at position t, predict x_{t+1} given x_0..x_t.
    """
    results = []
    for k, (alpha, x) in enumerate(COMPONENT_PARAMS):
        obs_mat = make_observation_matrices(alpha, x)
        pi = np.ones(3) / 3
        rng = np.random.default_rng(42 + k)
        seqs, _ = generate_sequences(
            n_samples, seq_len + 1,
            component_params=[(alpha, x)], weights=np.array([1.0]), rng=rng
        )
        ce = np.zeros(seq_len)
        for i in range(n_samples):
            eta = pi.copy()
            for t in range(seq_len):
                # Update belief with observed token at position t
                z_obs = seqs[i, t]
                new_eta = eta @ obs_mat[z_obs]
                eta = new_eta / new_eta.sum()

                # Predict x_{t+1}
                pred = np.array([(eta @ obs_mat[z]).sum() for z in range(3)])
                pred = np.clip(pred, 1e-15, None)
                pred /= pred.sum()
                ce[t] -= np.log(pred[seqs[i, t + 1]])

        results.append(ce / n_samples)
    return results
