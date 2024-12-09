#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[19]:


#Input Data
src =pd.read_excel("C:/Users/asus/Downloads/Stunt dataset.xlsx")
src.info()


# In[20]:


#Describe
src.describe


# In[21]:


#Cek missing value
src.isna().sum()


# Tidak ada missing value pada dataset

# In[55]:


#Binning menurut WHO
categories = ['Rendah','Menengah', 'Tinggi','Sangat Tinggi']
src['Stunt Category'] = pd.cut(src['Prevalensi Stunting (TB/U) %'], bins=[-float('inf'), 10, 20, 30, float('inf')], labels=categories)


#Encoding
category_mapping = {'Rendah': 1, 'Menengah': 2, 'Tinggi': 3, 'Sangat Tinggi': 4}
src['Stunt CatNum'] = src['Stunt Category'].map(category_mapping)
print(src[['Kabupaten/Kota Prov Indonesia','Prevalensi Stunting (TB/U) %', 'Stunt Category', 'Stunt CatNum']])


# In[53]:


sns.countplot(x='Stunt Category', data=src, palette=sns.color_palette('pastel')[0:5])

# Adding data labels
for i, value in enumerate(src['Stunt Category'].value_counts()):
    plt.text(i, value + 0.1, str(value), ha='center', va='bottom', fontsize=10)

plt.title('Stunting Category Distribution')
plt.show()


# In[24]:


srcnokab = src.drop(['Kabupaten/Kota Prov Indonesia','Stunt Category','Stunt CatNum'], axis=1)
print(srcnokab.head(5))


# In[26]:


#cek outlier
Q1 = srcnokab.quantile(q=.25)
Q3 = srcnokab.quantile(q=.75)
IQR = Q3-Q1

data_iqr = srcnokab[-((srcnokab < (Q1-1.5*IQR)) | (srcnokab >(Q3+1.5*IQR))).any(axis=1)]
data_iqr.shape

print("Dimensi dataset awal", srcnokab.shape)
print("Dimensi dataset setelah pengecekan outlier", data_iqr.shape)


# In[49]:


#cek distribusi

nrows = 11
ncols = 2

fig, axes = plt.subplots(nrows, ncols, figsize=(15, 20))

for i, var in enumerate(srcnokab):
    row = i // ncols
    col = i % ncols
    sns.kdeplot(data=srcnokab, x=var, ax=axes[row, col], fill=True)
    axes[row, col].set_title(f'Distribution of {var}')

plt.tight_layout()
plt.show()


# In[38]:


#cek outlier menggunakan boxplot

# Membuat boxplot untuk kolom 'UMK'
data_iqr[['UMK']].boxplot(figsize=(10, 5))
plt.title('Boxplot of UMK')
plt.show()

# Memilih semua kolom kecuali 'UMK'
data_iqr_without_UMK = data_iqr.drop('UMK', axis=1)

# Membuat boxplot untuk seluruh kolom (selain 'UMK')
data_iqr_without_UMK.boxplot(figsize=(15, 7), vert=True)  # vert=False agar boxplot horizontal
plt.title('Boxplot untuk Seluruh Kolom (Selain UMK)')
plt.show()


# Terdapat outlier yang terdeteksi pada masing-masing fitur jika menggunakan perhitungan IQR sehingga perlu dilakukan penanganan outlier. Sebelum itu akan dilakukan pengecekan distribusi data

# In[50]:


#cek skewness
for var_name in srcnokab:
    skewness = round(srcnokab[var_name].skew(), 3)
    print(f'Skewness of {var_name}: {skewness}')


# Fitur-fitur dengan skewness negatif menunjukkan persebaran data yang besar ke arah kanan, menunjukkan bahwa nilai-nilai tersebut cenderung besar. Oleh karena itu, untuk mengatasi adanya pencilan (outlier), akan dilakukan standardisasi.

# In[51]:


#standardization
from sklearn import preprocessing
srcz = preprocessing.scale(srcnokab)
srcz


# In[59]:


#feature selection using correlation
srcz_df = pd.DataFrame(srcz, columns=['Prevalensi Stunting (TB/U) %', 'K4',
                                      'Persalinan FASYANKES', 'KF Lengkap', 'Vit A Ibu', 'bumil TTD', 'BBLR',
                                      'IMD', 'ASI', 'CPKB', 'IDL', 'A 6-11', 'A 12-59', 'A 6-59', 'mCPR',
                                      'Air Minum Layak', 'Sanitasi Layak', 'IKP', 'BNPT 40%', 'KKS 40%',
                                      'APK PAUD','UMK'])

correlation = srcz_df.corr()

# Plot heatmap
plt.figure(figsize=(20, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation between Numeric Variables")
plt.show()


# Pemilihan fitur didasarkan pada korelasi yang mendekati nilai 0,5, menunjukkan hubungan yang kuat. Fitur lainnya tidak dipertimbangkan karena memiliki korelasi yang mendekati 0.

# In[62]:


# Plotting scatterplots in a grid

fig, axes = plt.subplots(11, 2, figsize=(10, 20))

for i, ax in enumerate(axes.flatten()):
    if i == 0:
        continue  # Skip the first subplot

    x = srcz_df.iloc[:, i]
    y = srcz_df['Prevalensi Stunting (TB/U) %']

    ax.scatter(x, y, marker='o')
    ax.grid()
    ax.set_ylim(ymin=0)
    ax.set_xlabel(srcz_df.columns[i])
    ax.set_ylabel('Prevalensi Stunting')

plt.tight_layout()
plt.show()


# In[61]:


#K-means Clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[16]:


import os

# Set OMP_NUM_THREADS to 1 to avoid memory leak warning
os.environ['OMP_NUM_THREADS'] = '1'

silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, n_init=100)
    kmeans.fit(srcz)
    silhouette_scores.append(silhouette_score(srcz, kmeans.labels_))

plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Jumlah Cluster')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score untuk Berbagai Jumlah Cluster')
plt.show()

for i, score in enumerate(silhouette_scores, 2):
    print(f"Silhouette Score for {i} clusters: {score:.3f}")


# Berdasarkan grafik di atas dapat diketahui bahwa jumlah cluster yang optimal adalah 2 (dua). Hal ini disebabkan oleh nilai silhouette yang paling tinggi terjadi ketika jumlah cluster = 2.

# In[63]:


#feature selected based on corr
X = srcz[:, [1, 2, 3, 6, 11, 14]]
Y = srcz[:, 0]


# In[64]:


# Suppress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[71]:


k_means = KMeans(init="k-means++", n_clusters=2, n_init=100)
labels = k_means.fit_predict(X)
print((labels))


# In[72]:


#menambahkan labels sebagai kolom baru
src['Cluster']=labels
src.head()


# In[73]:


# Plotting the results
plt.scatter(srcz[labels==0, 0], srcz[labels==0, 1], s=100, c='red', label ='Cluster 1')
plt.scatter(srcz[labels==1, 0], srcz[labels==1, 1], s=100, c='blue', label ='Cluster 2')

plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=100, c='yellow', label = 'Centroids')
plt.show()


# In[74]:


#Keanggotaan Kabupaten/Kota berdasarkan Cluster
grouped_data = src.groupby('Cluster')

for cluster, group_data in grouped_data:
    print(f"Cluster {cluster}:")
    print(group_data.iloc[:, 0])
    print("\n")


# In[75]:


#karakteristik tiap kluster
grouped_data.mean()


# Cluster 0 memiliki rata-rata prevalensi stunting yang lebih tinggi daripada cluster 1, sehingga Kabupaten/Kota yang tergabung pada cluster 0 memiliki rata-rata prevalensi stunting yang lebih tinggi daripada Kabupaten/Kota di Cluster 1.

# In[76]:


# Cluster performance
import sklearn
results = {}

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, random_state=100)
    labels = kmeans.fit_predict(srcz)
    db_index = sklearn.metrics.calinski_harabasz_score(srcz, labels)
    results.update({i: db_index})

# Menampilkan hasil dan menambahkan data label
for k, v in results.items():
    print(f"Number of clusters: {k}, Calinski-Harabasz Index: {v:.2f}")

# Plotting
plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Calinski-Harabasz Index")
plt.show()


# Kualitas pengelompokkan Kabupaten/Kota berdasarkan variabel prediktor menjadi 2 klaster dapat dinilai melalui nilai Silhouette dan Calinski-Harabasz Index. Kedua metrik ini menunjukkan bahwa pemilihan 2 klaster adalah keputusan yang optimal, karena keduanya mencapai titik tertinggi pada jumlah klaster tersebut.

# In[77]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


# In[78]:


#splitting training & testing + all scalling
srcz_df['Cluster']=labels
srcz_df['StuntCatNum']= src['Stunt CatNum']

Xc = srcz_df.iloc[:, list(range(1, 21))]
Yc = srcz_df.iloc[:, 22]
xc_train, xc_test, yc_train, yc_test = train_test_split(Xc, Yc, test_size=0.2, random_state=1)
print ('Train set:',xc_train.shape, yc_train.shape)
print ('Test set:', xc_test.shape, yc_test.shape)


# In[79]:


from sklearn import svm
model_SVM = svm.SVC(kernel='linear')
model_SVM.fit(xc_train, yc_train)
yc_pred_SVM = model_SVM.predict(xc_test)
print(classification_report(yc_test, yc_pred_SVM))


# In[80]:


from sklearn import svm
model_SVM = svm.SVC(kernel='rbf')
model_SVM.fit(xc_train, yc_train)
yc_pred_SVM = model_SVM.predict(xc_test)
print(classification_report(yc_test, yc_pred_SVM))


# In[81]:


from sklearn import svm
model_SVM = svm.SVC(kernel='sigmoid')
model_SVM.fit(xc_train, yc_train)
yc_pred_SVM = model_SVM.predict(xc_test)
print(classification_report(yc_test, yc_pred_SVM))


# In[82]:


from sklearn import svm
model_SVM = svm.SVC(kernel='poly')
model_SVM.fit(xc_train, yc_train)
yc_pred_SVM = model_SVM.predict(xc_test)
print(classification_report(yc_test, yc_pred_SVM))


# In[ ]:




