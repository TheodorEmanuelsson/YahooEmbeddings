import numpy as np

def normalize_vec(vec):
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    else:
        return vec / norm
    
def preprocess(load=False, representation=None):
        
    if load:
        print('Loading the DBOW vectors')
        sum_vectors_train = np.load('sum_vectors_train.npy')
        sum_vectors_test = np.load('sum_vectors_test.npy')
        print('Loading the MeanPool vectors')
        mean_vectors_train = np.load('mean_vectors_train.npy')
        mean_vectors_test = np.load('mean_vectors_test.npy')
        print('Loading the PCA projected vectors')
        pca_vectors_train = np.load('pca_vectors_train.npy')
        pca_vectors_test = np.load('pca_vectors_test.npy')
        
    else: # Takes roughly 25h to process the data.
        # Initialize the DBOW class and the MeanPool class that does preprocessing and vectorization
        DBOW = DistributedBagOfWords(lemmatize=True, lowercase=True, remove_stopwords=True)
        MeanPool = MeanPooling(lemmatize=True, lowercase=True, remove_stopwords=True)
        PCA = PCA_Projection(lemmatize=True, lowercase=True, remove_stopwords=True)

        # Run the DBOW on all the data and store it as numpy arrays
        print('Performing DBOW transformation')
        sum_vectors_train = DBOW.transform(X_train)
        sum_vectors_test = DBOW.transform(X_test)
        print('Finished DBOW transformation')
        np.save('sum_vectors_train.npy', sum_vectors_train)
        np.save('sum_vectors_test.npy', sum_vectors_test)

        # Run the MeanPooling on all data and store it as numpy arrays
        print('Performing MeanPool transformation')
        mean_vectors_train = MeanPool.transform(X_train)
        mean_vectors_test = MeanPool.transform(X_test)
        np.save('mean_vectors_train.npy', mean_vectors_train)
        np.save('mean_vectors_test.npy', mean_vectors_test)
        print('Finished MeanPool transformation')
        
        # Run the PCA_Projection on all the training data and store it as a numpy array
        print('Performing PCA transformation')
        pca_vectors_train = PCA.transform(X_train)
        pca_vectors_test = PCA.transform(X_test)
        np.save('pca_vectors_train.npy', pca_vectors_train)
        np.save('pca_vectors_test.npy', pca_vectors_test)
        print('Finished PCA transformation')
        
        return sum_vectors_train, sum_vectors_test, mean_vectors_train, mean_vectors_test, pca_vectors_train, pca_vectors_test
        
        
        