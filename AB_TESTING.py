##################

#####################################################
# Business Problem
#####################################################
# Facebook recently introduced a new bidding type called "average bidding" as an alternative to the existing "maximum bidding."
# One of our clients, bombabomba.com, has decided to test this new feature and wants to conduct an A/B test to determine
# whether average bidding generates more conversions than maximum bidding. The A/B test has been ongoing for 1 month,
# and bombabomba.com now expects you to analyze the results of this A/B test. The ultimate success metric for bombabomba.com is Purchase.
# Therefore, the focus should be on the Purchase metric for statistical tests.

#####################################################
# Dataset Story
#####################################################
# This dataset contains information about a company's website, including data on the number of ads viewed and clicked by users,
# as well as revenue data generated from these ads. There are two separate datasets: Control and Test groups, which are available
# on separate sheets in an Excel file named ab_testing.xlsx. Maximum Bidding was applied to the Control group, while Average Bidding
# was applied to the Test group.

# impression: Number of ad impressions
# Click: Number of clicks on the displayed ads
# Purchase: Number of products purchased after clicking on the ads
# Earning: Revenue generated after product purchases

#####################################################
# Project Tasks
#####################################################

######################################################
# AB Testing (Independent Two-Sample T-Test)
######################################################
# 1. Formulate Hypotheses
# 2. Assumption Checks
#   - 1. Normality Assumption (Shapiro-Wilk Test)
#   - 2. Homogeneity of Variances (Levene's Test)
# 3. Application of Hypothesis Test
#   - 1. If assumptions are met, use independent two-sample t-test
#   - 2. If assumptions are not met, use Mann-Whitney U test
# 4. Interpret the results based on the p-value
#   Note:
#    - If normality is not met, proceed to step 2 directly. If homogeneity of variances is not met, provide an argument for step 1.
#    - Prior to normality assessment, it may be useful to perform outlier detection and correction.




#####################################################
# MISSION 1 : DATA PREPROCESSING
#####################################################
import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
from scipy.stats import shapiro, ttest_1samp, levene,\
    ttest_ind, mannwhitneyu, pearsonr,\
    spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest



# Step 1:
control = pd.read_excel("Modul 3- Measurement Problems/case study/ABTesti/ab_testing.xlsx", sheet_name="Control Group")
test = pd.read_excel("Modul 3- Measurement Problems/case study/ABTesti/ab_testing.xlsx", sheet_name="Test Group")

control.head()
test.head()


# Step 2:
control.describe().T
test.describe().T

# Missing Value Analysis - Function
def missing_values(dataframe):
    na_columns_ = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns_].isnull().sum().sort_values(ascending=True)
    ratio_ = (dataframe[na_columns_].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio_, 2)], axis=1, keys=['Total Missing Values', 'Ratio'])
    missing_df = pd.DataFrame(missing_df)
    return missing_df

def check_df(dataframe, head=5, column="Purchase"):
    print("--------------------- Shape ---------------------")
    print(dataframe.shape)

    print("---------------------- Types --------------------")
    print(dataframe.dtypes)

    print("--------------------- Head ---------------------")
    print(dataframe.head(head))

    print("--------------------- Missing Value Analysis ---------------------")
    print(missing_values(dataframe))

    print("--------------------- Quantiles ---------------------")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(control)
check_df(test)

# Step 3:

control["Group"] = "control"
test["Group"] = "test"
df = pd.concat([control, test], ignore_index = True)
df.head()

#####################################################
# MISSION 2 :  DEFINE THE HYPOTHESIS OF A/B TEST   
#####################################################

# Step 1: Define the hypothesis
# H0: M1 = M2
# (There is no statistically significant difference between the purchase averages for the Control and Test groups.)

# H1: M1 != M2
# (There is Statistically Significant difference between the purchase averages for the Control and Test groups.)

# Step 2: 
df.groupby(["Group"])["Purchase"].mean()

# kontrol   550.89406
# test      582.10610

#####################################################
# MISSION 3 : PERFORMING THE HYPOTHESIS TEST  
#####################################################

# Step 1: Assumption Control

# Normal Assumption for Kontrol Group #
test_stat, p_value = shapiro(df.loc[df["Group"] == "control", "Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, p_value))
if p_value < 0.05:
    print(f"H0 can be rejected since pvalue {p_value} is smaller than 0.05")
else:
    print(f"H0 can NOT be rejected since pvalue {p_value} is greater than 0.05")

# Normal Assumption for Test Group
test_stat, p_value = shapiro(df.loc[df["Group"] == "test", "Purchase"])
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, p_value))
if p_value < 0.05:
    print(f"H0 can be rejected since pvalue {p_value} is smaller than 0.05")
else:
    print(f"H0 can NOT be rejected since pvalue {p_value} is greater than 0.05")
    
# # Conclusion:
# # H0 cannot be rejected since p value > 0.05 in both groups.
# # The assumption of normal distribution is provided.

# Variance Homogenity (Levene Test) #
test_stat, pvalue = levene(df.loc[df["Group"] == "control", "Purchase"],
                           df.loc[df["Group"] == "test", "Purchase"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
if p_value < 0.05:
    print(f"H0 can be rejected since pvalue {p_value} is smaller than 0.05")
else:
    print(f"H0 can NOT be rejected since pvalue {p_value} is greater than 0.05")

# Conclusion:
# H0 cannot be rejected since p value > 0.05.
# The variances are homogenous.

# Step 2 : Implementation of Appropriate Hypothesis Testing

# The two independent sample T-tests that is parametric tests can be applied because the assumptions are provided.
test_stat, pvalue = ttest_ind(df.loc[df["Group"] == "control", "Purchase"],
                           df.loc[df["Group"] == "test", "Purchase"], equal_var=True)
print("Test Stat = %.4f, pvalue = %.4f" % (test_stat, p_value))
if p_value < 0.05:
    print(f"H0 can be rejected since pvalue {p_value} is smaller than 0.05")
else:
    print(f"H0 can NOT be rejected since pvalue {p_value} is greater than 0.05")

# Conclusion
# H0 hypothesis can not be rejected because p value > 0.05.
# With an accuracy rate of 95%, M1 = M2 .So we can state that there is no statistically significant difference
# between the purchase averages for the Control and Test Groups.(M1=M2)


