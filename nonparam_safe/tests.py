import numpy as np
import math
from .utils import safe_prepare_data
from .math_utils import binomial_cdf, normal_cdf

def get_ranks(data):
    """
    Calculates fractional ranks and returns tie counts for variance correction.
   
    """
    temp = np.argsort(data)
    ranks = np.empty_like(temp, dtype=float)
    ranks[temp] = np.arange(1, len(data) + 1)
    
    unique_values, inverse_indices, counts = np.unique(
        data, return_inverse=True, return_counts=True
    )
    
    for i, count in enumerate(counts):
        if count > 1:
            mask = (inverse_indices == i)
            ranks[mask] = np.mean(ranks[mask])
            
    return ranks, counts

def quantile_test(x, q=0.5, test_value=0.0, alternative='two-sided'):
    """
    1-Sample Quantile Test (from scratch).
    Tests H0: The q-th quantile of the population is test_value.
    """
    x, _ = safe_prepare_data(x)
    if not (0 < q < 1):
        raise ValueError("Quantile 'q' must be strictly between 0 and 1.")
        
    n = len(x)
    k = np.sum(x <= test_value)
    
    p_lower = binomial_cdf(k, n, q)
    p_upper = 1 - binomial_cdf(k - 1, n, q)
    
    if alternative == 'less':
        p_val = p_lower
    elif alternative == 'greater':
        p_val = p_upper
    else:
        p_val = min(1.0, 2 * min(p_lower, p_upper))
    
    return {
        'test': 'Quantile Test',
        'statistic': int(k),
        'p_value': float(p_val),
        'n_samples': n,
        'quantile': q
    }

def sign_test(x, y=None, test_value=0.0, alternative='two-sided'):
    """
    Robust Sign Test for 1-sample or paired samples.
    Tests the direction of differences rather than magnitude.
    """
    if y is not None:
        x, y = safe_prepare_data(x, y, paired=True)
        diff = x - y
    else:
        x, _ = safe_prepare_data(x)
        diff = x - test_value
        
    diff = diff[diff != 0]
    n = len(diff)
    
    if n == 0:
        return {'test': 'Sign Test', 'p_value': 1.0, 'statistic': 0, 'n_samples': 0}
        
    k = np.sum(diff > 0)
    p_lower = binomial_cdf(k, n, 0.5)
    p_upper = 1 - binomial_cdf(k - 1, n, 0.5)
    
    if alternative == 'less':
        p_val = p_lower
    elif alternative == 'greater':
        p_val = p_upper
    else:
        p_val = min(1.0, 2 * min(p_lower, p_upper))
        
    return {
        'test': 'Sign Test (Paired)' if y is not None else 'Sign Test (1-Sample)',
        'statistic': int(k),
        'p_value': float(p_val),
        'n_samples': n
    }

def mann_whitney_test(x, y, alternative='two-sided'):
    """
    Robust Mann-Whitney U Test with tie-corrected variance and continuity correction.
    """
    x, y = safe_prepare_data(x, y, paired=False)
    n1, n2 = len(x), len(y)
    n = n1 + n2
    
    combined = np.concatenate([x, y])
    ranks, tie_counts = get_ranks(combined)
    
    r1 = np.sum(ranks[:n1])
    u1 = r1 - (n1 * (n1 + 1)) / 2
    u2 = (n1 * n2) - u1
    
    if alternative == 'less':
        u_stat = u1
    elif alternative == 'greater':
        u_stat = u2
    else:
        u_stat = min(u1, u2)
        
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    std_u = math.sqrt((n1 * n2 / (n * (n - 1))) * (((n**3 - n) - tie_sum) / 12))
    mu_u = (n1 * n2) / 2
    
    # Continuity correction
    z = (u_stat - mu_u + 0.5) / std_u if u_stat < mu_u else (u_stat - mu_u - 0.5) / std_u
    p_val = normal_cdf(z)
    
    if alternative == 'two-sided':
        p_val = 2 * min(p_val, 1 - p_val)
        
    return {
        'test': 'Mann-Whitney U Test',
        'statistic': float(u_stat),
        'p_value': float(p_val),
        'n_samples_x': n1,
        'n_samples_y': n2
    }

def wilcoxon_test(x, y=None, alternative='two-sided'):
    """
    Robust Wilcoxon Signed-Rank Test with tie-corrected variance.
    """
    if y is not None:
        x, y = safe_prepare_data(x, y, paired=True)
        diff = x - y
    else:
        x, _ = safe_prepare_data(x)
        diff = x
        
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return {'test': 'Wilcoxon', 'p_value': 1.0, 'statistic': 0}
        
    abs_diff = np.abs(diff)
    ranks, tie_counts = get_ranks(abs_diff)
    
    w_plus = np.sum(ranks[diff > 0])
    w_minus = np.sum(ranks[diff < 0])
    
    if alternative == 'less':
        w_stat = w_plus
    elif alternative == 'greater':
        w_stat = w_minus
    else:
        w_stat = min(w_plus, w_minus)
        
    mu_w = n * (n + 1) / 4
    tie_sum = np.sum(tie_counts**3 - tie_counts)
    std_w = math.sqrt((n * (n + 1) * (2 * n + 1) / 24) - (tie_sum / 48))
    
    z = (w_stat - mu_w + 0.5) / std_w if w_stat < mu_w else (w_stat - mu_w - 0.5) / std_w
    p_val = normal_cdf(z)
    
    if alternative == 'two-sided':
        p_val = 2 * min(p_val, 1 - p_val)
        
    return {
        'test': 'Wilcoxon Signed-Rank',
        'statistic': float(w_stat),
        'p_value': float(p_val),
        'n_samples': n
    }

def paired_test(x, y, method='wilcoxon', alternative='two-sided'):
    """Convenience router for paired tests."""
    if method.lower() == 'wilcoxon':
        return wilcoxon_test(x, y, alternative=alternative)
    elif method.lower() == 'sign':
        return sign_test(x, y, alternative=alternative)
    else:
        raise ValueError("Method must be 'wilcoxon' or 'sign'.")