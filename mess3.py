"""
Data generation and belief state computation for non-ergodic Mess3 HMMs.

A Mess3 process is a 3-state, 3-emission HMM parameterized by (alpha, x):
  - Transition: stays in state with prob alpha, uniform over others with (1-alpha)/2
  - Emission: emits symbol matching state with prob (1-2x), others with prob x each

The non-ergodic setting has K=3 components with different (alpha, x) pairs.
Each sequence is generated entirely from one component.
"""

import numpy as np


# Component parameters: (alpha, x)
COMPONENT_PARAMS = [
    (0.9, 0.1),  # Sharp emissions, slow switching
    (0.7, 0.2),  # Moderate
    (0.5, 0.3),  # Diffuse emissions, fast switching
]
COMPONENT_WEIGHTS = np.array([1/3, 1/3, 1/3])


def make_transition_matrix(alpha):
    """Transition matrix for Mess3: T[i,j] = P(state j | state i)."""
    T = np.full((3, 3), (1 - alpha) / 2)
    np.fill_diagonal(T, alpha)
    return T


def make_emission_matrix(x):
    """Emission matrix: E[s,z] = P(emit z | state s)."""
    E = np.full((3, 3), x)
    np.fill_diagonal(E, 1 - 2 * x)
    return E


def make_observation_matrices(alpha, x):
    """
    Combined observation matrices T_z[i,j] = P(state i -> state j) * P(emit z | state j).

    Returns: (3, 3, 3) array where T_z[z, i, j] = T[i,j] * E[j,z]
    """
    T = make_transition_matrix(alpha)
    E = make_emission_matrix(x)
    T_z = T[np.newaxis, :, :] * E.T[:, np.newaxis, :]
    return T_z


def generate_sequences(n_seq, seq_len, component_params=None, weights=None, rng=None):
    """
    Generate sequences from the non-ergodic Mess3 mixture.

    Each sequence is generated entirely from one component.

    Returns:
        sequences: (n_seq, seq_len) int array of tokens
        component_ids: (n_seq,) int array of which component generated each sequence
    """
    if component_params is None:
        component_params = COMPONENT_PARAMS
    if weights is None:
        weights = COMPONENT_WEIGHTS
    if rng is None:
        rng = np.random.default_rng()

    K = len(component_params)
    sequences = np.empty((n_seq, seq_len), dtype=np.int64)
    component_ids = rng.choice(K, size=n_seq, p=weights)

    # Precompute observation matrices for each component
    obs_matrices = [make_observation_matrices(a, x) for a, x in component_params]

    for i in range(n_seq):
        k = component_ids[i]
        alpha, x = component_params[k]
        T = make_transition_matrix(alpha)
        E = make_emission_matrix(x)

        # Initial state from stationary distribution (uniform)
        state = rng.choice(3)

        for t in range(seq_len):
            # Emit token from current state
            sequences[i, t] = rng.choice(3, p=E[state])
            # Transition to next state
            if t < seq_len - 1:
                state = rng.choice(3, p=T[state])

    return sequences, component_ids


def compute_belief_states(sequences, component_ids, component_params=None, weights=None):
    """
    Compute Bayesian belief states for each sequence at each position.

    Belief at position t is conditioned on tokens 0..t-1 (BEFORE observing token t).
    At position 0, belief = prior (stationary distribution).

    Returns dict with:
        within_component: (N, seq_len, 3) — belief using true component's matrices
        joint: (N, seq_len, 9) — [w0*eta0, w1*eta1, w2*eta2] full joint belief
        component_posterior: (N, seq_len, 3) — posterior P(component | tokens seen)
    """
    if component_params is None:
        component_params = COMPONENT_PARAMS
    if weights is None:
        weights = COMPONENT_WEIGHTS

    N, seq_len = sequences.shape
    K = len(component_params)

    # Precompute observation matrices for each component
    obs_matrices = [make_observation_matrices(a, x) for a, x in component_params]
    pi = np.ones(3, dtype=np.float64) / 3  # Uniform by symmetry (doubly stochastic T)

    within_component = np.zeros((N, seq_len, 3), dtype=np.float64)
    joint = np.zeros((N, seq_len, K * 3), dtype=np.float64)
    component_posterior = np.zeros((N, seq_len, K), dtype=np.float64)

    for i in range(N):
        # Initialize beliefs for each component hypothesis
        # eta_k = belief over states assuming component k
        etas = [pi.copy() for _ in range(K)]
        comp_weights = weights.copy().astype(np.float64)

        true_k = component_ids[i]

        for t in range(seq_len):
            # Record beliefs BEFORE observing token t
            # Normalize component weights
            cw = comp_weights / comp_weights.sum()
            component_posterior[i, t] = cw

            for k in range(K):
                eta_norm = etas[k] / etas[k].sum()
                joint[i, t, k*3:(k+1)*3] = cw[k] * eta_norm

            within_component[i, t] = etas[true_k] / etas[true_k].sum()

            # Update beliefs after observing token t
            if t < seq_len:
                z = sequences[i, t]
                for k in range(K):
                    T_z = obs_matrices[k]
                    # Update: eta_k' = eta_k @ T_z[z]  (row vector @ matrix)
                    new_eta = etas[k] @ T_z[z]
                    # The likelihood of this token under component k
                    likelihood_k = new_eta.sum()
                    comp_weights[k] *= likelihood_k
                    # Normalize eta to be a distribution (or keep unnormalized and normalize when reading)
                    if likelihood_k > 0:
                        etas[k] = new_eta / likelihood_k
                    # else: etas[k] stays (won't matter, weight is 0)

                # Prevent underflow in component weights
                if comp_weights.sum() > 0:
                    comp_weights /= comp_weights.sum()

    return {
        'within_component': within_component,
        'joint': joint,
        'component_posterior': component_posterior,
    }


def verify_data_generation(n_seq=10000, seq_len=50, verbose=True):
    """Run sanity checks on data generation and belief computation."""
    results = {}

    # Check observation matrices
    for k, (alpha, x) in enumerate(COMPONENT_PARAMS):
        T_z = make_observation_matrices(alpha, x)
        # Sum over emissions should give transition matrix
        T_sum = T_z.sum(axis=0)
        T_expected = make_transition_matrix(alpha)
        assert np.allclose(T_sum, T_expected), f"Component {k}: T_z sum != T"
        # All entries non-negative
        assert (T_z >= 0).all(), f"Component {k}: negative entries in T_z"
    results['observation_matrices'] = 'OK'

    # Check sequence statistics
    sequences, component_ids = generate_sequences(n_seq, seq_len)

    # Token frequencies should be roughly uniform
    token_freqs = np.bincount(sequences.flatten(), minlength=3) / sequences.size
    assert np.allclose(token_freqs, 1/3, atol=0.02), f"Token frequencies not uniform: {token_freqs}"
    results['token_frequencies'] = token_freqs

    # Component frequencies should be roughly uniform
    comp_freqs = np.bincount(component_ids, minlength=3) / len(component_ids)
    assert np.allclose(comp_freqs, 1/3, atol=0.05), f"Component frequencies not uniform: {comp_freqs}"
    results['component_frequencies'] = comp_freqs

    # Check belief states on a small subset
    subset = 100
    beliefs = compute_belief_states(sequences[:subset], component_ids[:subset])

    # Beliefs should sum to 1
    wc_sums = beliefs['within_component'].sum(axis=-1)
    assert np.allclose(wc_sums, 1.0, atol=1e-10), "Within-component beliefs don't sum to 1"
    joint_sums = beliefs['joint'].sum(axis=-1)
    assert np.allclose(joint_sums, 1.0, atol=1e-10), "Joint beliefs don't sum to 1"
    cp_sums = beliefs['component_posterior'].sum(axis=-1)
    assert np.allclose(cp_sums, 1.0, atol=1e-10), "Component posteriors don't sum to 1"
    results['beliefs_sum_to_1'] = 'OK'

    # Beliefs should be non-negative
    assert (beliefs['within_component'] >= -1e-15).all(), "Negative within-component beliefs"
    assert (beliefs['joint'] >= -1e-15).all(), "Negative joint beliefs"
    assert (beliefs['component_posterior'] >= -1e-15).all(), "Negative component posteriors"
    results['beliefs_nonneg'] = 'OK'

    # At position 0, within-component belief should be uniform (stationary = 1/3 each)
    assert np.allclose(beliefs['within_component'][:, 0, :], 1/3, atol=1e-10), \
        "Position 0 beliefs not uniform"
    results['initial_belief_uniform'] = 'OK'

    if verbose:
        print("Data generation verification:")
        for k, v in results.items():
            print(f"  {k}: {v}")

    return results


if __name__ == '__main__':
    verify_data_generation()
