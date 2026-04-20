import math

def combinations(n, k):
    """Calculate nCr."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k
    
    numerator = 1
    for i in range(k):
        numerator = numerator * (n - i) // (i + 1)
    return numerator

def binomial_cdf(k, n, p):
    """Calculate the probability of getting k or fewer successes in n trials."""
    cdf = 0.0
    for i in range(k + 1):
        prob = combinations(n, i) * (p**i) * ((1 - p)**(n - i))
        cdf += prob
    return min(cdf, 1.0)

def normal_cdf(x):
    """Standard Normal CDF (Approximation) using the error function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))