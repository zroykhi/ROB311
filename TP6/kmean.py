from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.metrics import confusion_matrix

# read data
train_data = pd.read_csv('data/optdigits.tra')
train_labels = train_data.iloc[:,-1]
n_digits = len(np.unique(train_labels))
train_data.drop(train_data.columns[len(train_data.columns)-1], axis=1, inplace=True)
train_data = np.array(train_data)

test_data = pd.read_csv('data/optdigits.tes')
test_labels = test_data.iloc[:,-1]
test_data.drop(test_data.columns[len(test_data.columns)-1], axis=1, inplace=True)

n_samples, n_features = train_data.shape
sample_size = 300

print("n_digits: %d, \t n_training_samples %d, \t n_features %d"
	  % (n_digits, n_samples, n_features))

fig, axes = plt.subplots(2, 4)
fig.suptitle("Example digital numbers", fontsize="x-large")
for ax in axes.flat:
	isample = np.random.randint(train_data.shape[0])
	ax.imshow(train_data[isample].reshape(8, 8), cmap='gray')
	ax.set_title("chiffre = {}".format(train_labels[isample]))
	ax.axis('off')

def bench_k_means(estimator, init_method, train_data, test_data, test_labels):
	t0 = time()
	estimator.fit(train_data)
	k_labels = estimator.predict(test_data)
	print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
	print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
	      % (init_method, (time() - t0), estimator.inertia_,
	         metrics.homogeneity_score(test_labels, k_labels),
	         metrics.completeness_score(test_labels, k_labels),
	         metrics.v_measure_score(test_labels, k_labels),
	         metrics.adjusted_rand_score(test_labels, k_labels),
	         metrics.adjusted_mutual_info_score(test_labels, k_labels,
	                                            average_method='arithmetic'),
	         metrics.silhouette_score(test_data, test_labels,
	                                  metric='euclidean',
	                                  sample_size=sample_size)))
	# initialize a array with same dimension of k_lables
	k_labels_matched = np.empty_like(k_labels)
	# For each cluster label, find and assign the best-matching truth label
	for k in np.unique(k_labels):
		match_nums = [np.sum((k_labels==k)&(test_labels==t)) for t in np.unique(test_labels)]
		k_labels_matched[k_labels==k] = np.unique(test_labels)[np.argmax(match_nums)]
	nerr_test = (k_labels_matched != test_labels).sum()
	print("recognition rate of test data = {:.1f}%".format(100 - 100*float(nerr_test)/test_data.shape[0]))
	cm = confusion_matrix(test_labels, k_labels_matched)
	print("Confusion matrix:\n%s" % cm)
	plt.figure(figsize=(10,10))
	class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
	plot_confusion_matrix(cm, classes=class_names, normalize=True,
					title='Normalized confusion matrix of initialization method ' + init_method)

# run
bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              init_method="random", train_data=train_data, test_data=test_data, test_labels=test_labels)

# Visualize the results on PCA-reduced data
reduced_data = PCA(n_components=2).fit_transform(train_data)
kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure()
plt.clf()
plt.imshow(Z, interpolation='nearest',
		   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
		   cmap=plt.cm.Paired,
		   aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)

# Plot centroids as white X and label next to it
# -----------------------------------------------
centroids = kmeans.cluster_centers_
# centroids are in label-wise order (0 to 9), we just plot those cluster labels
for i in range(len(centroids)):
	plt.text(centroids[i][0] + 1, centroids[i][1] + 1, str(i),bbox=dict(facecolor='white', alpha=0.8))
plt.scatter(centroids[:, 0], centroids[:, 1],
			marker='x', s=169, linewidths=3,
			color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
		  'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
