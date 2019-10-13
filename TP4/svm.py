# coding: utf-8
from sklearn import datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import timeit
from sklearn.metrics import confusion_matrix
from plotcm import plot_confusion_matrix
import argparse
from tqdm import tqdm

"""
usage: 
    python svm.py --num_epoch 1 -k linear -g 1 -C 0.001 --train_size 1000
""" 
def int2float_grey(x):
    x = x / 255
    return x

train_data_path = "./data/mnist_train.csv"
test_data_path = "./data/mnist_test.csv"

def main(epoch=1, C=0.001, gamma=1, kernel='linear', train_size=-1, search_parameters=False):
    # Read dataset to pandas dataframe
    train_dataset = pd.read_csv(train_data_path)
    trainY = np.array(train_dataset.iloc[:, 0])
    trainX = np.array(train_dataset.iloc[:, 1:])
    if train_size != -1:
        trainX = trainX[:train_size,:]
        trainY = trainY[:train_size]
    test_dataset = pd.read_csv(test_data_path)
    testY = np.array(test_dataset.iloc[:, 0])
    testX = np.array(test_dataset.iloc[:, 1:])

    # pre-processing data
    trainX = int2float_grey(trainX)
    testX = int2float_grey(testX)

    print('training data size: ', trainX.shape, trainY.shape)
    print('test data size: ', testX.shape, testY.shape)

    # display exmaple images
    fig, axes = plt.subplots(2, 4)
    plt.title('example images')
    for ax in axes.flat:
        isample = np.random.randint(trainX.shape[0])
        ax.imshow(trainX[isample].reshape(28,28),cmap='gray')
        ax.set_title("Chiffre = {}".format(trainY[isample]))
        ax.axis('off')

    tic = timeit.default_timer()
    if search_parameters:
        # search the best parameters
        print('searching for the best parameters will take long time, please wait ...')
        svc = svm.SVC(shrinking=True, max_iter=500) # max_iter = 500 pour limiter les non convergences de l'optimiseur 
        Clist=np.logspace(0,2,10)
        Glist=np.logspace(0,3,10)
        Dlist=[1,2,3]
        Kernellist = ('linear', 'poly')
        parameters = {'kernel': Kernellist, 'C':Clist, 'gamma':Glist, "degree":Dlist}
        clf = GridSearchCV(svc, parameters)
    else:
        clf = svm.SVC(C=C, gamma=gamma, kernel=kernel)
    print('start training ...')
    for i in tqdm(range(epoch)):
        clf.fit(trainX, trainY)
    toc = timeit.default_timer()
    print("Execution time = {:.3g} s".format(toc-tic)) 
    # if search for the best parameter, use below code to print out the results found
    if search_parameters:
        print("best score is {}".format(clf.best_score_))
        print("the best parametres are: ")
        print(clf.best_params_)
    print('predicting ...')
    y_test_predic = clf.predict(testX)
    nerr_test = (y_test_predic != testY).sum()
    print("recognition rate of test data = {:.1f}%".format(100 - 100*float(nerr_test)/testX.shape[0]))

    cm = confusion_matrix(testY, y_test_predic)
    print("Confusion matrix:\n%s" % cm)

    plt.figure(figsize=(10,10))
    plt.title('confusion matrix')
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plot_confusion_matrix(cm, classes=class_names, normalize=True,
                        title='Normalized Confusion Matrix')

    plt.show() 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train svm')
    parser.add_argument('-g', help='parameter gamma', default=1, type=float)
    parser.add_argument('-C', help='parameter C', default=1, type=float)
    parser.add_argument('--num_epochs', help='Number of training epoch', default=3, type=int)
    parser.add_argument('--train_size', help='training size', default=5000, type=int)
    parser.add_argument('-k', help='parameter kernel, could be linear, rbf, poly, etc', default='linear', type=str)
    parser.add_argument('-s', help='whether search for parameters', default=False)
    args = parser.parse_args()
    print("""
    usage: 
        python svm.py --num_epoch 3 -k linear -g 1 -C 1 --train_size 5000
    see more:
        python svm.py -h""")

    main(epoch=args.num_epochs, C=args.C, gamma=args.g, kernel=args.k, train_size=args.train_size, search_parameters=args.s)
