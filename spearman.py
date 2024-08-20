import pandas as pd

file_path = 'C:/Users/ashle/PycharmProjects/diploma/Real and Predicted.xlsx'
data = pd.read_excel(file_path)
df = pd.DataFrame(data)

# Calculate Spearman correlation
spearman_corr = df.corr(method='spearman')

# Display the Spearman correlation matrix
print(spearman_corr)

# put into excel file
output_file = 'spearman_correlation_output.xlsx'
spearman_corr.to_excel(output_file)

