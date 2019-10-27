# ROB 311 - TP6 - 21/10/2019

Authors: Simon Queyrut, Zhi Zhou

## Gist
In this 6th TP we implement a k-means unsupervised classification algorithm on ICU archives 
Optical Recognition of Handwritten Digits Data Set based on `sklearn` library.

#### Note
- This dataset has lower resolution than MNIST dataset so we do not expect better accuracy for models that try to classify these data in general.
- The provided source code is widely inspired by `sklearn` documentation examples.

## Implementation
Running the algorithm relies solely on the execution of the `bench_k_means` function defined on line 27.

Its parameters are 
- `estimator` which is the core `sklearn` component on which the `bench_k_means` will call the `.fit(train_data)` method to have it train on our dataset.
It will afterwards undergo the `.predict(test_data)` method so we get its version of the labels. The latter are a priori wrongly assigned since the unsupervised learning has, by definition, none of them during the learning process. Line 44 to 48 endeavours to correctly replace those labels with the correct ones (wrongly assigned `3` cluster would therefore be renamed to, say, `5` for instance)
- `init_method` is set to `'random'` since centroids are initially chosen randomly.
- `train_data`, `test_data` and `test_labels` are the dataset features and labels.

We run this function with

```bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10), init_method="random", train_data=train_data, test_data=test_data, test_labels=test_labels)```

### KMeans Estimator
First parameter is, as mentionned, a core `sklearn` component which take the following parameters:
- `n_cluster` is the number of clusters to form as well as the number of centroids to generate.
- `n_init` Number of time the k-means algorithm will be run with different centroid seeds. Final results are the best outputs from this pool from a `sklearn` computing method.

## Plot
We performe PCA dimension reduction on the data to plot the clusters (the plot doesn't have much utility but gives an overall idea of the clustering geometry)


![](https://markdown.data-ensta.fr/uploads/upload_1abd2da3e185cd4eb961bcedb199424a.png)
     
## Results    

We obtain this confusion matrix:

![](https://markdown.data-ensta.fr/uploads/upload_192ec8ad06a0026d01132ef990827263.png)

We note that there is confusion for correctly labeling `9`. We assume this is due to resemblance betweeen `9` and `3` in the dataset due to low resolution:

This three looks like a nine |
:----:|
![](https://markdown.data-ensta.fr/uploads/upload_5298474b549389c18c5a6028eb7b9343.png)
