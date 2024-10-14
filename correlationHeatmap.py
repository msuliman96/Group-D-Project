import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'Bacteria.xlsx'
bacteria_data = pd.read_excel(file_path)

plt.figure(figsize=(10, 8))
correlation_matrix = bacteria_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of All Variables')
plt.savefig('Correlation_Matrix_All_Variables.png')
plt.show()
