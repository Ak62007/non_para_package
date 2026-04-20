import numpy as np
from scipy import stats
from .utils import safe_prepare_data

def quantile_test(x, q=0.5, test_value=0.0, alternative='two-sided'):
    """
    1-Sample Quantile Test. Tests if the q-th quantile of x equals test_value.
    Defaults to the Median Test (q=0.5).
    """
    x, _ = safe_prepare_data(x)
    if not (0 < q < 1):
        raise ValueError("Quantile 'q' must be strictly between 0 and 1.")
        
    # Count how many observations are less than or equal to the test_value
    k = np.sum(x <= test_value)
    n = len(x)
    
    # Under the null hypothesis, this follows a Binomial distribution B(n, q)
    res = stats.binomtest(k, n, p=q, alternative=alternative)
    
    return {
        'test': 'Quantile Test',
        'statistic': k,
        'p_value': res.pvalue,
        'n_samples': n
    }

def sign_test(x, y=None, test_value=0.0, alternative='two-sided'):
    """
    Sign test for 1-sample or paired samples.
    """
    if y is not None:
        x, y = safe_prepare_data(x, y, paired=True)
        diff = x - y
    else:
        x, _ = safe_prepare_data(x)
        diff = x - test_value
        
    # Exclude zero differences (ties)
    diff = diff[diff != 0]
    n = len(diff)
    
    if n == 0:
        raise ValueError("All differences are zero. Cannot perform sign test.")
        
    positive_diffs = np.sum(diff > 0)
    
    # Under null, positive differences follow B(n, 0.5)
    res = stats.binomtest(positive_diffs, n, p=0.5, alternative=alternative)
    
    return {
        'test': 'Sign Test (Paired)' if y is not None else 'Sign Test (1-Sample)',
        'statistic': positive_diffs,
        'p_value': res.pvalue,
        'n_samples': n
    }

def wilcoxon_test(x, y=None, alternative='two-sided'):
    """
    Wilcoxon signed-rank test for 1-sample or paired samples.
    """
    if y is not None:
        x, y = safe_prepare_data(x, y, paired=True)
        # scipy wilcoxon handles paired arrays directly
        res = stats.wilcoxon(x, y, alternative=alternative)
        test_name = 'Wilcoxon Signed-Rank (Paired)'
    else:
        x, _ = safe_prepare_data(x)
        res = stats.wilcoxon(x, alternative=alternative)
        test_name = 'Wilcoxon Signed-Rank (1-Sample)'
        
    return {
        'test': test_name,
        'statistic': res.statistic,
        'p_value': res.pvalue,
        'n_samples': len(x)
    }

def mann_whitney_test(x, y, alternative='two-sided'):
    """
    Mann-Whitney U test for two independent samples.
    """
    x, y = safe_prepare_data(x, y, paired=False)
    
    res = stats.mannwhitneyu(x, y, alternative=alternative)
    
    return {
        'test': 'Mann-Whitney U Test',
        'statistic': res.statistic,
        'p_value': res.pvalue,
        'n_samples_x': len(x),
        'n_samples_y': len(y)
    }

def paired_test(x, y, method='wilcoxon', alternative='two-sided'):
    """
    A convenience router for paired tests.
    Methods available: 'wilcoxon', 'sign'
    """
    if method.lower() == 'wilcoxon':
        return wilcoxon_test(x, y, alternative=alternative)
    elif method.lower() == 'sign':
        return sign_test(x, y, alternative=alternative)
    else:
        raise ValueError("Method must be 'wilcoxon' or 'sign'.")
