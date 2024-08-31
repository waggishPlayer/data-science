import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import os
import importlib
warnings.filterwarnings('ignore')
sns.set_theme(color_codes=True)

# user-defined function to check library is installed or not, if not installed then it will install automatically at runtime.
def check_and_install_library(library_name):
    try:
        importlib.import_module(library_name)
        print(f"{library_name} is already installed.")
    except ImportError:
        print(f"{library_name} is not installed. Installing...")
        try:
            import pip
            pip.main(['install', library_name])
        except:
            print("Error: Failed to install the library. Please install it manually.")
if 'amazon-product-reviews' not in os.listdir():
  check_and_install_library('opendatasets')
  import opendatasets as od
  od.download('https://github.com/waggishPlayer/Data-Science/blob/main/Devtown/capstone-project/E-commerce-recommendation-system/ratings_Electronics.csv')
  
#load the dataframe and set column name
df=pd.read_csv('amazon-product-reviews/ratings_Electronics.csv',names=['userId', 'productId','rating','timestamp'])
electronics_data=df.sample(n=10000,ignore_index=True)

#after taking samples drop df to release the memory occupied by entire dataframe
del df
#print top 5 records of the dataset
electronics_data.head()
#print the concise information of the dataset
electronics_data.info()
#drop timestamp column
electronics_data.drop('timestamp',axis=1,inplace=True)
#handling duplicate records
electronics_data[electronics_data.duplicated()].shape[0]
#plot the data in chart
plt.figure(figsize=(8,4))
sns.countplot(x='rating',data=electronics_data)
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid()
plt.show()
#recommending
data=electronics_data.groupby('productId').filter(lambda x:x['rating'].count()>=50)
no_of_rating_per_product=data.groupby('productId')['rating'].count().sort_values(ascending=False)
#top 20 product
no_of_rating_per_product.head(20).plot(kind='bar')
plt.xlabel('Product ID')
plt.ylabel('num of rating')
plt.title('top 20 procduct')
plt.show()
#average rating product
mean_rating_product_count=pd.DataFrame(data.groupby('productId')['rating'].mean())
#plot the rating distribution of average rating product
plt.hist(mean_rating_product_count['rating'],bins=100)
plt.title('Mean Rating distribution')
plt.show()
#check the skewness of the mean rating data
mean_rating_product_count['rating'].skew()
#it is highly negative skewed
mean_rating_product_count['rating_counts'] = pd.DataFrame(data.groupby('productId')['rating'].count())
#highest mean rating product
mean_rating_product_count[mean_rating_product_count['rating_counts']==mean_rating_product_count['rating_counts'].max()]
#joint plot of rating and rating counts
sns.jointplot(x='rating',y='rating_counts',data=mean_rating_product_count)
plt.title('Joint Plot of rating and rating counts')
plt.tight_layout()
plt.show()