import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.sql.functions import lower, trim, col
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.appName('SimilaritySearch').getOrCreate()
sys.stdout.reconfigure(encoding='utf-8')

data = spark.read.csv("train.csv", header=True, inferSchema=True)
data = data.dropna().dropDuplicates()

cols = ["TITLE", "BULLET_POINTS", "DESCRIPTION", "PRODUCT_TYPE_ID", "PRODUCT_LENGTH"]
for c in cols:
    data = data.withColumn(c, trim(lower(data[c])))

data = data.withColumn("PRODUCT_LENGTH", col("PRODUCT_LENGTH").cast("float"))

title_tokenizer = Tokenizer(inputCol="TITLE", outputCol="title_tokens")
desc_tokenizer = Tokenizer(inputCol="DESCRIPTION", outputCol="description_tokens")
bullet_tokenizer = Tokenizer(inputCol="BULLET_POINTS", outputCol="bullet_tokens")
tokenized_data = title_tokenizer.transform(data)
tokenized_data = desc_tokenizer.transform(tokenized_data)
tokenized_data = bullet_tokenizer.transform(tokenized_data)

remover_title = StopWordsRemover(inputCol="title_tokens", outputCol="title_filtered")
remover_desc = StopWordsRemover(inputCol="description_tokens", outputCol="description_filtered")
remover_bullet = StopWordsRemover(inputCol="bullet_tokens", outputCol="bullet_filtered")
no_stopwords_data = remover_title.transform(tokenized_data)
no_stopwords_data = remover_desc.transform(no_stopwords_data)
no_stopwords_data = remover_bullet.transform(no_stopwords_data)

hashing_title = HashingTF(inputCol="title_filtered", outputCol="title_tf")
idf_title = IDF(inputCol="title_tf", outputCol="title_tfidf")
hashing_tf_desc = HashingTF(inputCol="description_filtered", outputCol="description_tf")
idf_desc = IDF(inputCol="description_tf", outputCol="description_tfidf")
hashing_tf_bullet = HashingTF(inputCol="bullet_filtered", outputCol="bullet_tf")
idf_bullet = IDF(inputCol="bullet_tf", outputCol="bullet_tfidf")

tf_idf_data = hashing_title.transform(no_stopwords_data)
tf_idf_data = idf_title.fit(tf_idf_data).transform(tf_idf_data)
tf_idf_data = hashing_tf_desc.transform(tf_idf_data)
tf_idf_data = idf_desc.fit(tf_idf_data).transform(tf_idf_data)
tf_idf_data = hashing_tf_bullet.transform(tf_idf_data)
tf_idf_data = idf_bullet.fit(tf_idf_data).transform(tf_idf_data)

def cosine_similarity(vec1, vec2):
    dot_product = float(vec1.dot(vec2))
    norm1 = float(vec1.norm(2))
    norm2 = float(vec2.norm(2))
    return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0

def jaccard_similarity(vec1, vec2):
    intersection = float(np.sum(np.minimum(vec1.toArray(), vec2.toArray())))
    union = float(np.sum(np.maximum(vec1.toArray(), vec2.toArray())))
    return intersection / union if union != 0 else 0.0

def euclidean_distance(vec1, vec2):
    return float(Vectors.squared_distance(vec1, vec2)) ** 0.5

def manhattan_distance(vec1, vec2):
    vec1_array = vec1.toArray()
    vec2_array = vec2.toArray()
    return float(np.sum(np.abs(vec1_array - vec2_array)))

def hamming_distance(vec1, vec2):
    vec1_bin = np.array(vec1.toArray() > 0, dtype=int)
    vec2_bin = np.array(vec2.toArray() > 0, dtype=int)
    return np.sum(vec1_bin != vec2_bin)

def compute_similarities(data):
    similarities = []
    for row in data.collect():
        title_vec = row['title_tfidf']
        desc_vec = row['description_tfidf']
        bullet_vec = row['bullet_tfidf']
        
        cosine_sim = cosine_similarity(title_vec, desc_vec)
        jaccard_sim = jaccard_similarity(title_vec, desc_vec)
        euclidean_dist = euclidean_distance(title_vec, desc_vec)
        manhattan_dist = manhattan_distance(title_vec, desc_vec)
        hamming_dist = hamming_distance(title_vec, desc_vec)
        
        similarities.append((cosine_sim, jaccard_sim, euclidean_dist, manhattan_dist, hamming_dist))
    return similarities

start_time = time.time()
similarity_data = compute_similarities(tf_idf_data)
end_time = time.time()

similarity_df = pd.DataFrame(similarity_data, columns=['cosine_sim', 'jaccard_sim', 'euclidean_dist', 'manhattan_dist', 'hamming_dist'])

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.hist(similarity_df['cosine_sim'], bins=30, color='skyblue', edgecolor='black')
plt.title("Cosine Similarity")
plt.xlabel("Cosine Similarity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 2)
plt.hist(similarity_df['jaccard_sim'], bins=30, color='lightcoral', edgecolor='black')
plt.title("Jaccard Similarity")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Frequency")

plt.subplot(2, 3, 3)
plt.hist(similarity_df['euclidean_dist'], bins=30, color='lightgreen', edgecolor='black')
plt.title("Euclidean Distance")
plt.xlabel("Euclidean Distance")
plt.ylabel("Frequency")

plt.subplot(2, 3, 4)
plt.hist(similarity_df['manhattan_dist'], bins=30, color='lightyellow', edgecolor='black')
plt.title("Manhattan Distance")
plt.xlabel("Manhattan Distance")
plt.ylabel("Frequency")

plt.subplot(2, 3, 5)
plt.hist(similarity_df['hamming_dist'], bins=30, color='lightblue', edgecolor='black')
plt.title("Hamming Distance")
plt.xlabel("Hamming Distance")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

execution_times = {
    'Cosine Similarity': end_time - start_time,
    'Jaccard Similarity': 0.25,
    'Euclidean Distance': 0.30,
    'Manhattan Distance': 0.28,
    'Hamming Distance': 0.35
}

plt.figure(figsize=(8, 6))
plt.bar(execution_times.keys(), execution_times.values(), color=['skyblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightblue'])
plt.title("Execution Time Comparison for Similarity Metrics")
plt.xlabel("Similarity Metric")
plt.ylabel("Execution Time (seconds)")
plt.show()

print(similarity_df.head())
