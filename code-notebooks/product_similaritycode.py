# -*- coding: utf-8 -*-
"""Product_SimilarityCode.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jB-i3EvA0A-fhHtcLvNA8jZT6xBUE2Dw

PRODUCT DATABASE AND COUNTERFEIT SCORES
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

df_one = pd.read_csv("/content/product_database.csv")

pd.set_option('display.max_columns', None)

df_one.head(5)

df_one.columns

print(f"The Number of Rows are {df_one.shape[0]}, and columns are {df_one.shape[1]}.")

df_one.info()

df_one.isnull().sum()

df = df_one.drop_duplicates(subset='Product_ID')
df.info()

# Changing the data type of discounted price and actual price

df['Discounted_Price'] = df['Discounted_Price'].str.replace("₹",'')
df['Discounted_Price'] = df['Discounted_Price'].str.replace(",",'')
df['Discounted_Price'] = df['Discounted_Price'].astype('float64')

df['Actual_Price'] = df['Actual_Price'].str.replace("₹",'')
df['Actual_Price'] = df['Actual_Price'].str.replace(",",'')
df['Actual_Price'] = df['Actual_Price'].astype('float64')
# Changing Datatype and values in Discount Percentage

df['Discount_Percentage'] = df['Discount_Percentage'].str.replace('%','').astype('float64')

df['Discount_Percentage'] = df['Discount_Percentage'] / 100
# Finding unusual string in rating column
df['Rating'].value_counts()

df.query('Rating == "|"')

df['Rating'] = df['Rating'].str.replace('|', '3.9').astype('float64')
# Changing 'rating_count' Column Data Type

df['Rating_Count'] = df['Rating_Count'].str.replace(',', '').astype('float64')
df.info()

df.describe()

df.isnull().sum().sort_values(ascending = False)

round(df.isnull().sum() / len(df) * 100, 2).sort_values(ascending=False)

df[df['Rating_Count'].isnull()].head(5)

df['Rating_Count'] = df.Rating_Count.fillna(value=df['Rating_Count'].median())

df.isnull().sum().sort_values(ascending = False)

plt.scatter(df['Actual_Price'], df['Rating'])
plt.xlabel('Actual_price')
plt.ylabel('Rating')
plt.show()

def identify_popular_products(month):
    popular_products = df[(df[month] > df[month].quantile(0.75)) &
                            (df['Rating_Count'] > df['Rating_Count'].quantile(0.75))]
    return popular_products[['Product_ID', 'Product_Name', month, 'Rating_Count']]

# Identify popular products for each month
popular_products_january = identify_popular_products('January')
popular_products_february = identify_popular_products('February')
popular_products_march = identify_popular_products('March')
popular_products_april = identify_popular_products('April')
popular_products_may = identify_popular_products('May')
popular_products_june = identify_popular_products('June')
popular_products_july = identify_popular_products('July')
popular_products_august = identify_popular_products('August')
popular_products_september = identify_popular_products('September')
popular_products_october = identify_popular_products('October')
popular_products_november = identify_popular_products('November')
popular_products_december = identify_popular_products('December')

# Print the popular products for each month
print("Popular products in January:")
print(popular_products_january)
print("\nPopular products in February:")
print(popular_products_february)
print("\nPopular products in March:")
print(popular_products_march)
print("\nPopular products in April:")
print(popular_products_april)
print("\nPopular products in May:")
print(popular_products_may)
print("\nPopular products in June:")
print(popular_products_june)
print("\nPopular products in July:")
print(popular_products_july)
print("\nPopular products in August:")
print(popular_products_august)
print("\nPopular products in September:")
print(popular_products_september)
print("\nPopular products in October:")
print(popular_products_october)
print("\nPopular products in November:")
print(popular_products_november)
print("\nPopular products in December:")
print(popular_products_december)

# Visualization of popular products for each month
months = ['January', 'February', 'March', 'April', 'May','June','July','August','September','October','November','December']
for month in months:
    popular_products = identify_popular_products(month)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=month, y='Rating_Count', data=popular_products)
    plt.title(f'Popular Products in {month}: Sales vs Rating Count')
    plt.xlabel(f'{month} Sales')
    plt.ylabel('Rating Count')
    plt.show()

def identify_potential_counterfeit_products(df):
    # Low rating with high sales
    low_rating_high_sales = df[(df['rating'] < df['rating'].quantile(0.25)) &
                               (df[['January', 'February', 'March', 'April', 'May', 'June',
                                   'July', 'August', 'September', 'October', 'November', 'December']].sum(axis=1) > df[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].sum(axis=1).quantile(0.75))]

    # High discount percentage
    high_discount = df[df['discount_percentage'] > df['discount_percentage'].quantile(0.75)]

    # Inconsistent rating patterns (e.g., sudden spikes)
    inconsistent_ratings = df[(df['rating'] - df['rating'].shift()).abs() > 2]

    potential_counterfeit = pd.concat([low_rating_high_sales, high_discount, inconsistent_ratings]).drop_duplicates()
    return potential_counterfeit[['product_id', 'product_name', 'rating', 'discount_percentage']]

potential_counterfeit_products = identify_potential_counterfeit_products(df)
print("Potential Counterfeit Products:")
print(potential_counterfeit_products)

def find_consistently_popular_products(df, threshold=0.75):
    popular_products = set()
    for month in months:
        month_popular = set(df[df[month] > df[month].quantile(threshold)]['Product_ID'])
        popular_products.update(month_popular)
    popular_products = list(popular_products)

    consistency_df = df[df['Product_ID'].isin(popular_products)]
    consistency_df['popular_months_count'] = consistency_df.apply(lambda row: sum([row[month] > df[month].quantile(threshold) for month in months]), axis=1)
    consistent_products = consistency_df[consistency_df['popular_months_count'] > 1]

    return consistent_products[['Product_ID', 'Product_Name', 'popular_months_count']]

consistent_products = find_consistently_popular_products(df)
print("Products with high sales for more than one month:")
print(consistent_products)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def calculate_counterfeit_score(row):
    rating = row['Rating']
    discount_percentage = row['Discount_Percentage']
    sales_sum = row[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].sum()

    # Assign weights to different factors
    rating_weight = 0.4
    discount_weight = 0.3
    sales_weight = 0.3

    # Calculate the counterfeit score based on the factors
    rating_score = 1 - rating / 5  # Lower rating, higher score
    discount_score = discount_percentage
    sales_score = sales_sum / df[['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']].sum(axis=1).max()

    counterfeit_score = rating_weight * rating_score + discount_weight * discount_score + sales_weight * sales_score

    return counterfeit_score

# Calculate the counterfeit score for each product
df['counterfeit_score'] = df.apply(calculate_counterfeit_score, axis=1)

# Normalize the counterfeit scores between 0 and 1
scaler = MinMaxScaler()
df['counterfeit_score'] = scaler.fit_transform(df['counterfeit_score'].values.reshape(-1, 1))
print(df['counterfeit_score'])

# Save the product ID and counterfeit score to a new CSV file
counterfeit_scores = df[['Product_ID', 'counterfeit_score']]
counterfeit_scores.to_csv('counterfeit_scores.csv', index=False)

dfs = pd.read_csv('/content/counterfeit_scores.csv')
sorted_df = dfs.sort_values(by='counterfeit_score', ascending=False)
print(sorted_df)

# the higher the score - more likely to be counterfeited

high_scores = dfs[dfs['counterfeit_score'] > 0.65]

print(f"Number of values greater than 0.6: {len(high_scores)}")

"""SIMILARITY SCORE FOR REVIEWS"""

!pip install sentence_transformers

from tqdm import tqdm, trange
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
import torch

# Load your dataset
ama = pd.read_csv('/content/product_database1.csv')

# Step 1: Load and prepare your data
descriptions = ama['About_Product'].tolist()

# Step 2: Text Preprocessing (example, you might need more preprocessing steps)
# Note: TfidfVectorizer includes basic preprocessing such as lowercasing and stop word removal.
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Step 3: Feature Extraction
embeddings = model.encode(descriptions, convert_to_tensor=True)

# Step 4: Similarity Calculation
def calculate_similarity(new_description, embeddings, model):
    new_embedding = model.encode(new_description, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(new_embedding, embeddings)
    return similarity_scores

# Step 5: Determine if a new description is too similar
new_description = "Flexible, thin HDMI cable for connecting to playback display such as HDTVs, projectors, and more. Compatible with Apple and Samsung Devices"
similarity_scores = calculate_similarity(new_description, embeddings, model)

# Define a similarity threshold
threshold = 0.8  # Adjust based on your requirement

# Check if the new description is too similar to any existing description
is_too_similar = (similarity_scores > threshold).any().item()

print(f"Similarity scores: {similarity_scores}")
print(f"Is the new description too similar? {'Yes' if is_too_similar else 'No'}")

# Optionally, find the most similar existing description
most_similar_index = similarity_scores.argmax().item()
most_similar_score = similarity_scores[0, most_similar_index].item()
if most_similar_index < len(descriptions):
    most_similar_description = descriptions[most_similar_index]
    print(f"Most similar description: {most_similar_description}")
    print(f"Similarity score with the most similar description: {most_similar_score:.2f}")
else:
    print("Error: Most similar index is out of range.")
model_save_path = "paraphrase-MiniLM-L6-v2-model.pth"
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")

import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd

# Load the pre-trained model
model_save_path = "paraphrase-MiniLM-L6-v2-model.pth"
model = torch.load(model_save_path)

# Load your dataset
ama = pd.read_csv('/content/product_database1.csv')
descriptions = ama['About_Product'].tolist()

# Encode all existing descriptions
embeddings = model.encode(descriptions, convert_to_tensor=True)

def calculate_similarity(new_description, embeddings, model):
    new_embedding = model.encode(new_description, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(new_embedding, embeddings)
    return similarity_scores

def check_similarity(new_description, threshold=0.7):
    similarity_scores = calculate_similarity(new_description, embeddings, model)
    is_too_similar = (similarity_scores > threshold).any().item()

    print(f"Is the new description too similar? {'Yes' if is_too_similar else 'No'}")

    most_similar_index = similarity_scores.argmax().item()
    most_similar_score = similarity_scores[0, most_similar_index].item()

    if most_similar_index < len(descriptions):
        most_similar_description = descriptions[most_similar_index]
        print(f"Most similar description: {most_similar_description}")
        print(f"Similarity score with the most similar description: {most_similar_score:.2f}")
    else:
        print("Error: Most similar index is out of range.")

# User input loop
while True:
    new_description = input("Enter a new product description (or 'quit' to exit): ")
    if new_description.lower() == 'quit':
        break
    check_similarity(new_description)
    print()  # Add a blank line for readability

print("Thank you for using the similarity checker!")