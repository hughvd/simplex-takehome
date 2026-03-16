"""
Training loop for the Mess3 transformer.

Uses online data generation (fresh sequences each batch) and logs
per-position loss for comparison with Bayes-optimal bounds.
"""

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mess3 import (
    COMPONENT_PARAMS, COMPONENT_WEIGHTS,
    generate_sequences, make_observation_matrices,
)
from model import Transformer, ModelConfig


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def compute_theoretical_loss(component_params=None, weights=None, seq_len=12,
                             n_samples=50000, rng=None):
    """
    Estimate the Bayes-optimal cross-entropy at each position by Monte Carlo.

    Aligned with the model's convention: at position t, the model has seen
    tokens x_0..x_t and predicts x_{t+1}. So we update belief with x_t first,
    then compute the predictive CE for x_{t+1}.

    Returns: (seq_len,) array of per-position cross-entropy in nats
    """
    if component_params is None:
        component_params = COMPONENT_PARAMS
    if weights is None:
        weights = COMPONENT_WEIGHTS
    if rng is None:
        rng = np.random.default_rng(42)

    K = len(component_params)
    obs_matrices = [make_observation_matrices(a, x) for a, x in component_params]
    pi = np.ones(3) / 3

    # Need seq_len+1 tokens: model sees 0..seq_len-1, predicts 1..seq_len
    sequences, _ = generate_sequences(n_samples, seq_len + 1, component_params, weights, rng)

    ce_per_pos = np.zeros(seq_len)

    for i in range(n_samples):
        etas = [pi.copy() for _ in range(K)]
        comp_w = weights.copy().astype(np.float64)

        for t in range(seq_len):
            # Update beliefs with observed token at position t
            z_obs = sequences[i, t]
            for k in range(K):
                new_eta = etas[k] @ obs_matrices[k][z_obs]
                lik = new_eta.sum()
                comp_w[k] *= lik
                if lik > 0:
                    etas[k] = new_eta / lik
            if comp_w.sum() > 0:
                comp_w /= comp_w.sum()

            # Predict x_{t+1} from belief conditioned on x_0..x_t
            cw = comp_w / comp_w.sum()
            pred = np.zeros(3)
            for k in range(K):
                eta = etas[k] / etas[k].sum()
                for z in range(3):
                    pred[z] += cw[k] * (eta @ obs_matrices[k][z]).sum()
            pred = np.clip(pred, 1e-15, None)
            pred /= pred.sum()

            target = sequences[i, t + 1]
            ce_per_pos[t] -= np.log(pred[target])

    ce_per_pos /= n_samples
    return ce_per_pos


def train(model=None, config=None, n_steps=5000, batch_size=64, lr=1e-3,
          log_interval=100, device=None, verbose=True):
    """
    Train the transformer on non-ergodic Mess3 data.

    Uses online generation: fresh sequences each batch.

    Returns:
        history: dict with 'loss' (per step), 'per_position_loss' (logged periodically)
    """
    if config is None:
        config = ModelConfig()
    if model is None:
        model = Transformer(config)
    if device is None:
        device = get_device()

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.default_rng()
    seq_len = config.context_length

    history = {
        'loss': [],
        'per_position_loss': [],
        'steps_logged': [],
    }

    pbar = tqdm(range(n_steps), disable=not verbose, desc='Training')
    for step in pbar:
        # Generate fresh batch
        sequences, _ = generate_sequences(batch_size, seq_len + 1, rng=rng)
        tokens = torch.tensor(sequences, dtype=torch.long, device=device)
        inputs = tokens[:, :-1]   # (B, seq_len)
        targets = tokens[:, 1:]   # (B, seq_len)

        # Forward pass
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        history['loss'].append(loss_val)

        if step % log_interval == 0 or step == n_steps - 1:
            # Compute per-position loss
            with torch.no_grad():
                eval_seqs, _ = generate_sequences(512, seq_len + 1, rng=rng)
                eval_tokens = torch.tensor(eval_seqs, dtype=torch.long, device=device)
                eval_logits, _ = model(eval_tokens[:, :-1])
                per_pos_loss = []
                for t in range(seq_len):
                    pos_loss = F.cross_entropy(eval_logits[:, t, :], eval_tokens[:, t + 1]).item()
                    per_pos_loss.append(pos_loss)
                history['per_position_loss'].append(per_pos_loss)
                history['steps_logged'].append(step)

            if verbose:
                pbar.set_postfix({'loss': f'{loss_val:.4f}', 'pos0': f'{per_pos_loss[0]:.3f}',
                                  'pos-1': f'{per_pos_loss[-1]:.3f}'})

    return model, history


if __name__ == '__main__':
    print("Computing theoretical Bayes-optimal loss...")
    theo_loss = compute_theoretical_loss(n_samples=10000)
    print(f"Theoretical loss per position: {np.round(theo_loss, 4)}")
    print(f"Mean: {theo_loss.mean():.4f}, log(3)={np.log(3):.4f}")

    print("\nTraining model...")
    model, history = train(n_steps=2000, verbose=True)
    print(f"Final loss: {history['loss'][-1]:.4f}")
    print(f"Final per-position: {[f'{x:.3f}' for x in history['per_position_loss'][-1]]}")
