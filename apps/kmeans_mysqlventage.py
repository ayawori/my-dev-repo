# Source:
# Clustering Using KMeans with Teradata Python Package
#
# https://docs.teradata.com/reader/eteIDCTX4O4IMvazRMypxQ/1EvzbEtx5ctHXiwlJYsbbg
#

from teradataml.analytics.mle.KMeans import KMeans
from teradataml.dataframe.dataframe import DataFrame
from teradataml.data.load_example_data import load_example_data
from teradataml import create_context
import matplotlib.pyplot as plt

create_context(host = "localhost", username="root", password = "mysqlpass", temp_database_name='menagerie')

load_example_data("kmeans", "kmeans_us_arrests_data")

## Creating TeradataML dataframes
df_train = DataFrame('kmeans_us_arrests_data')

# Print train data to see how the sample train data looks like
print("\nHead(10) of Train data:\n")
print(df_train.head(10), end='\n\n')

# A dictionary to get 'sno' to 'state' mapping. Required for plotting.
# df1 = df_train.select(['sno', 'state'])
# sno_to_state = dict(df1.to_pandas()['state'])
sno_to_state = dict(zip(df_train.to_pandas()['sno'], df_train.to_pandas()['state']))
print(sno_to_state)


# No need of 'state' columns, instead we have 'sno' column for the same
df_train = df_train.drop(['state'], axis=1)

# KMmeans algorithm on data, with 2 centroids
kMeans_output = KMeans(data=df_train, centers=2, data_sequence_column=['sno'])

# Print the KMeans results
print(kMeans_output)

# # Features (4-dimensional) used to train KMeans algorithm.
features = 'murder assault urban_pop rape'.split()

# Feature Centroids for cluster 1
centroid_values = kMeans_output.clusters_centroids[features].T[0].values
print("\n4-dimensional Centroid of Cluster 1:")
print(dict(zip(features, centroid_values)))

# Feature Centroids for cluster 2
centroid_values = kMeans_output.clusters_centroids[features].T[1].values
print("\n4-dimensional Centroid of Cluster 2:")
print(dict(zip(features, centroid_values)))

# Withinness
print("\nWithinness: " + str(kMeans_output.withinss))

# kMeans_output.clustered_output.head(30)
# print(kMeans_output.output)

# Inner join of clustered_output to actual dataset df_train We shall use the data from df1 to plot.
df1 = df_train.to_pandas().join(kMeans_output.clustered_output, how='inner', on=['sno'])

# print("\nInner join of clustered_output to actual dataset df_train:")
print(df1)

# # Selecting only the necessary features for plot.
df3 = df1[['sno', 'urban_pop', 'murder', 'clusterid']]

# Since there is no plotting possible for teradataml DataFrame, we are converting it to
# pandas dataframe and then to numpy_array 'numpy_df' to use matplotlib library of python.
numpy_df = df3.values

# Setting figure display size.
plt.rcParams['figure.figsize'] = [15, 10]

# Coloring based on cluster_id.
plt.scatter(numpy_df[:,1], numpy_df[:,2], c=numpy_df[:,3])
for ind, value in enumerate(numpy_df[:, 0]):
    # sno_to_state is used hear to get state names.
    plt.text(numpy_df[ind,1], numpy_df[ind,2], sno_to_state[int(value)], fontsize=14)
plt.xlabel('urban_pop')
plt.ylabel('murder')
plt.show()

# Selecting only the necessary features for plot.
df3 = df1[['sno', 'rape', 'murder', 'clusterid']]

# Since there is no plotting possible for teradataml DataFrame, we are converting it to
# pandas dataframe and then to numpy_array 'numpy_df' to use matplotlib library of python.
numpy_df = df3.values

# Coloring based on cluster_id.
plt.scatter(numpy_df[:,1], numpy_df[:,2], c=numpy_df[:,3])
for ind, value in enumerate(numpy_df[:, 0]):
    # sno_to_state is used hear to get state names.
    plt.text(numpy_df[ind,1], numpy_df[ind,2], sno_to_state[int(value)], fontsize=14)
plt.xlabel('rape')
plt.ylabel('murder')
plt.show()
