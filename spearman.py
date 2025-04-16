import pandas as pd
from scipy.stats import spearmanr

# read file with data
# file_path = './5yu3_Results of tests.xlsx'
# file_path = './7vtk_Results of tests.xlsx'
# file_path = './4d05_Results of tests.xlsx'
# file_path = 'P1-P2.xlsx'
# file_path = 'P1-P3.xlsx'
# file_path = 'P2-P1.xlsx'
# file_path = 'P2-P3.xlsx'
# file_path = 'P3-P1.xlsx'
file_path = 'P3-P2.xlsx'
data = pd.read_excel(file_path)
df = pd.DataFrame(data)

# Calculate Spearman correlation
spearman_corr = df.corr(method='spearman')

# Display the Spearman correlation matrix
print(spearman_corr)

# put into excel file (for 18 tests)
# output_file = 'spearman_correlation.xlsx'
# spearman_corr.to_excel(output_file)

# Spearman correlation for P-P model
spearman_corr = spearman_corr.loc['Actual', 'Predicted']
print(spearman_corr)

