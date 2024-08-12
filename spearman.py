import pandas as pd
import numpy as np
from scipy.stats import spearmanr

file_path = 'D:/Documents/Bioinformatics/Final project paper/final1.xlsx'
data = pd.read_excel(file_path)
print(data.head())
df = pd.DataFrame(data)

# Calculate Spearman correlation
correlation, p_value = spearmanr(df.iloc[:, 0].values, df.iloc[:, 1].values)

# Display the result
print(f'Spearman correlation: {correlation}')
print(f'P-value: {p_value}')
