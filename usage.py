import numpy as np
import nonparam_safe.tests as nps

# 1. Create messy data (includes NaNs to test safety wrappers)
#
group_a = [12.5, 14.1, np.nan, 15.2, 11.9, 13.8, 14.5]
group_b = [10.1, 11.2, 9.8, 12.0, np.nan, 10.5, 11.1]

# 2. Independent 2-Sample Test (Mann-Whitney)
# The wrapper automatically drops NaNs independently for A and B
print("--- Mann-Whitney U Test ---")
mw_result = nps.mann_whitney_test(group_a, group_b)
print(mw_result)

# 3. Paired Tests (Wilcoxon and Sign Test)
# Automatically drops the entire PAIR if either A or B is NaN
print("\n--- Paired Wilcoxon Test ---")
paired_wilcoxon = nps.paired_test(group_a, group_b, method='wilcoxon')
print(paired_wilcoxon)

print("\n--- Paired Sign Test ---")
paired_sign = nps.paired_test(group_a, group_b, method='sign')
print(paired_sign)

# 4. 1-Sample Quantile Test
# Testing if the median (q=0.5) of group_a is significantly different from 13.0
print("\n--- Quantile (Median) Test ---")
median_test = nps.quantile_test(group_a, q=0.5, test_value=13.0)
print(median_test)

# 5. Robustness Check: Tie-Heavy Data
# Validates the custom tie-correction logic
tie_data_1 = [10, 10, 11, 12, 12, 12]
tie_data_2 = [12, 12, 13, 14, 14, 15]
print("\n--- Mann-Whitney U with Heavy Ties ---")
tie_test = nps.mann_whitney_test(tie_data_1, tie_data_2)
print(tie_test)