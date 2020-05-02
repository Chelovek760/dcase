
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import roc_auc_score,f1_score
X=pd.read_csv(r'D:\Ботать\Работа\dcase\feature\X_fits_max_from_wavlet.csv',index_col=0)
X_test=pd.read_csv(r'D:\Ботать\Работа\dcase\feature\X_test_fits_max_from_wavlet.csv',index_col=0)
sns.pairplot(X)
plt.show()