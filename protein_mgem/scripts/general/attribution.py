"""
Functions for attribution aka relevance analysis
"""
import hashlib
from pathlib import Path

import h5py
import numpy as np
from numpy.random import default_rng
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import hamming
from scipy import stats
import tape
import torch

VOCAB_AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'


def mutate_seq_random_n_residues(seq_wt: str, n: int) -> str:
    """
    Mutate n residues to random values, excluding first residue (M normally).
    The mutation is reproducible and specific to sequence.
    """
    # https://stackoverflow.com/a/42089311
    seed_for_seq = int(hashlib.sha256(seq_wt.encode('utf-8')).hexdigest(), 16) % 10**8
    rng = default_rng(seed=seed_for_seq)
    seq_rand = list(seq_wt)
    rand_positions = rng.integers(1, len(seq_wt), size=n)
    for pos in rand_positions:
        replacement_set = set(list(VOCAB_AMINO_ACIDS)) - {seq_rand[pos]}
        replacement_set = list(sorted(replacement_set))
        seq_rand[pos] = rng.choice(replacement_set)
    seq_rand = ''.join(seq_rand)
    return seq_rand


def test_mutate_seq_random_n_residues():
    seq_wt = 'MTESTTEST'
    assert mutate_seq_random_n_residues(seq_wt, n=2) != 'MTESTTEST'
    assert mutate_seq_random_n_residues(seq_wt, n=2) == 'MTEGTTEPT'


def get_attention_values_for_sequence(sequence: str,
                                      model: tape.ProteinBertForValuePrediction,
                                      arch: dict) -> np.ndarray:
    """
    Evaluate model on sequence and return the
    (attention weights tensor, prediction gradient w.r.t. attention weights)

    Note: Excludes demarcation tokens at first and last position of tokenized sequence.
    """
    tokenizer = tape.TAPETokenizer(vocab='iupac')
    input_ids = torch.tensor([tokenizer.encode(sequence)])
    pred, attention = model(input_ids)
    pred.backward(retain_graph=True)
    grad_attention = torch.autograd.grad(pred, attention)

    attention_tensor = np.zeros((arch['num_hidden_layers'],
                                 arch['num_attention_heads'],
                                 len(sequence), len(sequence)),
                                dtype=np.float32)
    grad_attention_tensor = np.zeros((arch['num_hidden_layers'],
                                      arch['num_attention_heads'],
                                      len(sequence), len(sequence)),
                                     dtype=np.float32)

    for layer_num in range(arch['num_hidden_layers']):
        for head_num in range(arch['num_attention_heads']):
            attention_matrix = attention[layer_num][0][head_num].detach().numpy()
            attention_tensor[layer_num][head_num] = attention_matrix[1:-1, 1:-1]
            grad_attention_matrix = grad_attention[layer_num][0][head_num].detach().numpy()
            grad_attention_tensor[layer_num][head_num] = grad_attention_matrix[1:-1, 1:-1]

    return attention_tensor, grad_attention_tensor


def remove_redundant_attention_gradients(grad_attention: np.ndarray,
                                         low_corr_threshold: float) -> (np.ndarray, pd.DataFrame):
    """
    BERT has a tendency for redundancy.
    Keep only attention gradients that are most different for a given input sequence
        - below the correlation threshold
        - dissimilar (low corr) to at least 90% of other attention gradients

    Return:
         - filtered attention gradient tensor
         - metainformation DataFrame: layer and head of each gradient matrix in the returned tensor
    """
    num_hidden_layers = grad_attention.shape[0]
    num_attention_heads = grad_attention.shape[1]

    grad_attention_matrices = []
    for layer_num in range(num_hidden_layers - 1):
        for head_num in range(num_attention_heads):
            grad_attention_matrices.append(grad_attention[layer_num][head_num])

    grad_attention_corr = _calc_corr(grad_attention_matrices,
                                     num_hidden_layers,
                                     num_attention_heads)

    similarity_counts = (np.triu(grad_attention_corr > low_corr_threshold, k=1)).sum(axis=1)
    similarity_count_threshold = np.floor(0.1 * grad_attention_corr.shape[0]).astype(int)
    most_dissimilar_idx = set(np.where(similarity_counts < similarity_count_threshold)[0])

    grad_attention_patterns = []
    grad_attention_patterns_meta = []
    idx = 1
    for layer_num in range(num_hidden_layers - 1):
        for head_num in range(num_attention_heads):
            if idx in most_dissimilar_idx:
                grad_attention_patterns.append(grad_attention_matrices[idx - 1])
                grad_attention_patterns_meta.append((layer_num, head_num, idx - 1))
            idx += 1

    grad_attention_patterns = np.array(grad_attention_patterns, dtype=np.float32)
    grad_attention_patterns_meta = pd.DataFrame.from_records(
        grad_attention_patterns_meta, columns=['layer', 'head', 'idx']
    )
    return grad_attention_patterns, grad_attention_patterns_meta


def _calc_corr(grad_attention_matrices, num_hidden_layers, num_attention_heads):
    """Function extracted to optimize with Numba"""
    grad_attention_corr = np.zeros(((num_hidden_layers - 1) * num_attention_heads,
                                    (num_hidden_layers - 1) * num_attention_heads),
                                   dtype=np.float32)

    for i in range(len(grad_attention_matrices) - 1):
        for j in range(i + 1, len(grad_attention_matrices)):
            m1 = grad_attention_matrices[i].flatten()
            m2 = grad_attention_matrices[j].flatten()

            # Z-score (code pulled from scipy to work with numba)
            m1_zscore = (m1 - m1.mean()) / m1.std()
            m2_zscore = (m2 - m2.mean()) / m2.std()

            # Pearson correlation (code pulled from scipy to work with numba)
            m1_norm = np.sqrt(np.dot(m1_zscore, m1_zscore))
            m2_norm = np.sqrt(np.dot(m2_zscore, m2_zscore))
            r = np.dot(m1_zscore / m1_norm, m2_zscore / m2_norm)
            r = max(min(r, 1.0), -1.0)
            grad_attention_corr[i][j] = r

    return grad_attention_corr


def keep_only_significant_values(profile: np.ndarray, num_sd: int = 1) -> np.ndarray:
    """
    Filter profile to keep only values that fall outside of num_sd standard deviations.
    If the profile contains negative values (as is the case for attention gradients),
    also keep values that fall below -num_sd (i.e. both + an - extreme values).
    """
    relevant_profile = np.copy(profile)
    relevant_profile[np.isnan(relevant_profile)] = 0.0
    profile_zscores = stats.zscore(profile)
    if np.any(relevant_profile < 0):
        profile_zscores = np.abs(profile_zscores)
    relevant_profile[profile_zscores < num_sd] = 0.0
    return relevant_profile


def extract_sequence_pattern(profile: np.ndarray, sequence: str) -> str:
    pattern = np.array(list(sequence))
    pattern[np.isclose(profile.flatten(), 0.0)] = '-'
    pattern = ''.join(pattern)
    return pattern


def test_extract_sequence_pattern():
    from numpy.random import default_rng
    rng = default_rng(seed=42)

    seq = 'MUPPETSSNEAKYCOLLEGE'
    profile = np.array(
        [0.04396905, 0.05272429, 0.04317478, 0.03382302, 0.04052551,
         0.05308934, 0.04885484, 0.03226313, 0.04341354, 0.04672655,
         0.05638742, 0.02947875, 0.03331515, 0.05202734, 0.04764602,
         0.06288441, 0.04888312, 0.04411819, 0.05370312, 0.04099097],
        dtype=np.float32)
    profile_gradient = profile * rng.choice([-1, 1], len(seq))

    attention_pattern = extract_sequence_pattern(keep_only_significant_values(profile), seq)
    gradient_pattern = extract_sequence_pattern(keep_only_significant_values(profile_gradient),
                                                seq)
    assert attention_pattern == '----------A----L----'
    assert gradient_pattern == 'M--PE-S-NEA----L-E-E'


def test_matrix_corr():
    m1 = np.array(
        [[1.2215275e-03, -6.9469976e-04, -3.9704479e-04, -2.7952870e-04],
         [2.3739174e-04, -3.4528414e-03, 1.8863307e-03, 5.0837547e-03],
         [2.8563730e-04, -9.1095024e-04, 5.7829299e-04, 1.2153299e-03],
         [1.2982343e-03, -1.1938314e-03, -5.3871379e-05, 4.3954095e-04]],
        dtype=np.float32)
    m2 = np.array([[0.00095118, -0.00065179, -0.00070068, 0.00179623],
                   [-0.00212504, -0.00500771, -0.00228349, -0.00308231],
                   [0.00076039, -0.00054865, 0.00018743, 0.00046584],
                   [0.00089027, -0.00050132, -0.00032141, 0.00179909]],
                  dtype=np.float32)

    grad_attention_matrces = [m1, m2]
    grad_attention_corr = _calc_corr(grad_attention_matrces, 2, 2)

    np.isclose(
        grad_attention_corr[0][1],
        0.09981027607815099
    )


def load_patterns(prot_id: str, attention_dir: str, grouping: str = 'pairs') -> tuple:
    """
    Return a list of (attention, focus) pattern pairs if grouping == 'pairs'
    or lists [attention_patterns, focus_patterns] otherwise

    from the H5 store of the given protein.
    If these cannot be retrieved, None is returned.

    Note: Function gets both to reduce I/O
    """
    if isinstance(attention_dir, str):
        attention_dir = Path(attention_dir)
    try:
        with h5py.File(attention_dir / f'attention_patterns/attention_{prot_id}.h5', 'r') as pfile:
            attention_patterns = np.array(pfile[f'{prot_id}/attended_patterns'])
            focus_patterns = np.array(pfile[f'{prot_id}/grad_attended_patterns'])
    except Exception as e:
        print('WARN: Could not load pattern for ' + prot_id + ' : ' + str(e))
        if grouping == 'pairs':
            return [(None, None)]
        else:
            return [None, None]

    if grouping == 'pairs':
        attention_focus_pairs = [
            (p_a.decode('ascii'), p_f.decode('ascii'))
            for p_a, p_f in zip(attention_patterns, focus_patterns)
        ]
        return attention_focus_pairs

    else:
        attention_focus_lists = [
            [p_a.decode('ascii') for p_a in attention_patterns],
            [p_f.decode('ascii') for p_f in focus_patterns],
        ]
        return attention_focus_lists


def load_patterns_and_meta(prot_id: str, attention_dir: str) -> tuple:
    """
    Return attention patterns for protein ID (swissprot_ac) and metainformation
    about the origin (layer, head) of the patterns.
    """
    if isinstance(attention_dir, str):
        attention_dir = Path(attention_dir)
    try:
        store_fname = attention_dir / f'attention_patterns/attention_{prot_id}.h5'
        with h5py.File(store_fname, 'r') as pfile:
            attention_patterns = np.array(pfile[f'{prot_id}/attended_patterns'])
        meta = pd.read_hdf(store_fname, key=f'{prot_id}/meta')

    except Exception as e:
        print('WARN: Could not load pattern for ' + prot_id + ' : ' + str(e))
        return [None, None]

    attention_focus_lists = [
        [p_a.decode('ascii') for p_a in attention_patterns],
        meta
    ]
    return attention_focus_lists


def load_profiles(prot_id: str, attention_dir: str, include_meta=False) -> tuple:
    """
    Return lists [attention_profiles, focus_profiles]
    from the H5 store of the given protein.

    *_profiles is an array of shape (n_profiles, profile_length)

    If these cannot be retrieved, [None, None] is returned.

    Note: Function gets both to reduce I/O
    """
    try:
        store_fname = Path(attention_dir) / f'attention_patterns/attention_{prot_id}.h5'
        with h5py.File(store_fname, 'r') as pfile:
            attention_profiles = np.array(pfile[f'{prot_id}/attended_profiles'], dtype=np.float32)
            focus_profiles = np.array(pfile[f'{prot_id}/grad_attended_profiles'], dtype=np.float32)
        if include_meta:
            meta = pd.read_hdf(store_fname, key=f'{prot_id}/meta')

    except Exception as e:
        print('WARN: Could not load profiles for ' + prot_id + ' from ' +
              str(Path(attention_dir) / f'attention_patterns/attention_{prot_id}.h5'))
        print(str(e))
        if include_meta:
            return None, None, None
        else:
            return None, None

    if include_meta:
        return attention_profiles, focus_profiles, meta
    else:
        return attention_profiles, focus_profiles


def coverage_fract_in_window(pattern: str, start: int, end: int) -> float:
    """Start and end given as Python slice indices"""
    return (end - start - pattern[start:end].count('-')) / (end - start)


def mean_coverage_outside_window(pattern: str, start: int, end: int) -> float:
    """
    Return fraction of coverage outisde of the interval,
    averaged by dividing the total coverage count by the size of the window.

    Equivalent asymptotically to averaging fractions from random samples of same-length windows.

    Start and end given as Python slice indices.
    """
    pattern_outside = pattern[:start] + pattern[end:]
    n_segments = len(pattern_outside) / (end - start)
    n_letters = len(pattern_outside) - pattern_outside.count('-')
    mean_n_letters = n_letters / n_segments
    return mean_n_letters / (end - start)


def test_coverage_fract_in_window():
    test_seq = '-AB-D--X----Y---X----Y---X----Y---X----Y---X----Y'
    assert np.isclose(coverage_fract_in_window(test_seq, 1, 5), 0.75)


def test_mean_coverage_outside_window():
    test_seq = '-AB-D--X----Y---X----Y---X----Y---X----Y---X----Y'
    assert np.isclose(mean_coverage_outside_window(test_seq, 1, 5), 0.22, atol=1e-2)

    test_seq = 'ABCD--X-----'
    assert np.isclose(mean_coverage_outside_window(test_seq, 0, 6), 1/6)


def mean_total_attention_outside_window(total_attention: float,
                                        total_attention_in_window: float,
                                        seq_len: int,
                                        window_len: int) -> float:
    """
    Return fraction of attention profile outisde of the interval,
    averaged by dividing the total value by the size of the window.

    Equivalent asymptotically to averaging fractions from random samples of same-length windows.
    """
    total_attention_outside_window = total_attention - total_attention_in_window
    n_segments = (seq_len - window_len) / window_len
    return total_attention_outside_window / n_segments


def dist_pattern_structure(pattern: str, structure: str) -> float:
    """
    Measure how specific the match between an attention pattern
    and a secondary structure annotation is.

    Quantify as the Hamming distance between agreements of:
     - any matching residue  (encoded as 1)
     - empty / uninformative position  (encoded as 0

    Examples:

    good_match = (
        '-----------PEP----------------------',
        '           HHH                      '
    )

    bad_match = (
        '------------------PEPTI-------------',
        '           HHH                      '
    )

    decent_match = (
        '-------------PEP--------------------',
        '           HHH                      '
    )

    lopsided_pattern_match = (
        '--PEPTIDE--PEP--PEPTIDE-------------',
        '           HHH                      '
    )

    lopsided_structure_match = (
        '-------------PEP--------------------',
        '       HHHHHHHHHHHHHHHHH    HHHHHHH '
    )
    """
    return hamming(
        list(map(lambda char: 0 if char in ['-',' '] else 1, pattern)),
        list(map(lambda char: 0 if char in ['-',' '] else 1, structure)),
    )


def get_residue_attention_by_pos_bin(attention_profile: np.ndarray, seq: str, bins=10) -> pd.DataFrame:
    """
    Divide the profile in bins and count the total attention each residue receives.
    Return counts as a DataFrame with column [residue, pos_bin, attention]
    """
    res_attention = pd.DataFrame.from_records(zip(list(seq), attention_profile),
                                              columns=['residue', 'attention'])
    res_attention = res_attention.assign(
        pos_bin = pd.cut(res_attention.index, bins=bins, labels=False)
    )
    res_attention = res_attention.groupby(['residue', 'pos_bin'])['attention'].sum()
    return res_attention.reset_index()


def test_get_residue_attention_by_pos_bin():
    seq = 'MUPPETS'
    bins = 2
    attention_profile = np.ones(len(seq)) * 0.1

    expected_counts = pd.DataFrame.from_records(
        zip(
            ['M', 'P', 'U', 'E', 'S', 'T'],
            [0, 0, 0, 1, 1, 1],
            [0.1, 0.2, 0.1, 0.1, 0.1, 0.1]
        ),
        columns = ['residue', 'pos_bin', 'attention']
    )

    assert expected_counts.compare(
        get_residue_attention_by_pos_bin(attention_profile, seq, bins)
        .sort_values('pos_bin', ignore_index=True)
    ).empty


def upsample(vector: np.ndarray, max_len: int) -> np.array:
    """
    Return upsampled vector to max_len (using linear interpolation)
    """
    vector_interpolation = interp1d(np.arange(0, vector.shape[0]),
                                    vector, kind='linear')
    new_range = np.linspace(0, vector.shape[0] - 1, num=max_len)
    upsampled_vector = vector_interpolation(new_range)
    return upsampled_vector
