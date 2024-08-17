# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
df=pd.read_csv("/content/SAMPLEIDS.csv")
df
```
![alt text](image.png)
```
df.isnull().sum()

```
![alt text](image-1.png)
```
  df.isnull().any()
```
![alt text](image-2.png)
```
df.dropna()
```
![alt text](image-3.png)
```
df.fillna(0)
```
![alt text](image-4.png)
```
df.fillna(method='ffill')

```
![alt text](image-5.png)
```
df.fillna(method='bfill')

```
![alt text](image-6.png)
```
df_dropped=df.dropna()
df_dropped
```
![alt text](image-7.png)
```
df.fillna({'GENDER':'MALE','NAME':'SRI','ADDRESS':'POONAMALEE','M1':98,'M2':87,'M3':76,'M4':92,'TOTAL':305,'AVG':89.999999})
```
![alt text](image-8.png)
```
import pandas as pd
import seaborn as sns
```

```
ir=pd.read_csv('iris.csv')
ir
```
![alt text](image-9.png)
```
ir.describe()
```
![alt text](image-10.png)
```
sns.boxplot(x='sepal_width',data=ir)
```
![alt text](image-11.png)
```
c1=ir.sepal_width.quantile(0.25)
c3=ir.sepal_width.quantile(0.75)
iq=c3-c1
print(c3)
```
![alt text](image-12.png)
```
rid=ir[((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
rid['sepal_width']
```
![alt text](image-13.png)
```
delid=ir[~((ir.sepal_width<(c1-1.5*iq))|(ir.sepal_width>(c3+1.5*iq)))]
delid
```
![alt text](image-14.png)
```
sns.boxplot(x='sepal_width',data=delid)
```
![alt text](image-15.png)
```

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats

```

```
dataset=pd.read_csv("heights.csv")
dataset
```
![alt text](image-16.png)
```
df = pd.read_csv("heights.csv")
q1 = df['height'].quantile(0.25)
q2 = df['height'].quantile(0.5)
q3 = df['height'].quantile(0.75)
```
```
iqr = q3-q1
iqr

```
![alt text](image-17.png)
```
low = q1 - 1.5*iqr
low
```
![alt text](image-18.png)
```
high = q3 + 1.5*iqr
high
```
![alt text](image-19.png)
```
df1 = df[((df['height'] >=low)& (df['height'] <=high))]
df1
```
![alt text](image-20.png)
```
z = np.abs(stats.zscore(df['height']))
z
```
![alt text](image-21.png)
```
df1 = df[z<3]
df1
```
![alt text](image-22.png)

# Result
```
Thus we have cleaned the data and removed the outliers by detection using IQR and Z-score method
```
