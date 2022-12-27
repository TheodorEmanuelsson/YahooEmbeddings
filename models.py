from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import spacy
from utils import normalize_vec

nlp = spacy.load("en_core_web_lg")

class ParagraphEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, lemmatize:bool=False, lowercase:bool=False, remove_stopwords:bool=False):
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
    
    def fit(self, X, y=None):
        return self
    
    def _tokenize(self, sentence):
        if self.remove_stopwords:
            return self._remove_stopwords(sentence)
        return [token.text for token in nlp(sentence)]
    
    def _lemmatize(self, sentence):
        if self.lemmatize:
            return self._remove_stopwords(sentence)
        return [token.lemma_ for token in nlp(sentence)]

    def _remove_stopwords(self, sentence):
        if self.lemmatize:
            return [token.lemma_ for token in nlp(sentence) if not token.is_stop]

        return [token.text for token in nlp(sentence) if not token.is_stop]

class DistributedBagOfWords(ParagraphEmbedding):
    def __init__(self, lemmatize:bool=False, lowercase:bool=False, remove_stopwords:bool=False):
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords

    # Vectorize a block of text
    def _transform1(self, sentence):

        # Get the tokenized list of words
        words = self._tokenize(sentence)

        # If the sentence is empty, return a vector of zeros
        if len(words) == 0:
            return np.zeros(300)

        # Get the word vectors for each word in the sentence
        word_vectors = [normalize_vec(nlp.vocab[word].vector) for word in words]

        # Return the sum of the word vectors
        return np.sum(word_vectors, axis=0)

    # Vectorize a single row of the dataframe.
    def _transform2(self, row):

        # Concatenate the sentence vectors
        sent1_vec = self._transform1(row[0])
        sent2_vec = self._transform1(row[1])
        sent3_vec = self._transform1(row[2])
        return np.concatenate((sent1_vec, sent2_vec, sent3_vec), axis=0)
    
    def transform(self, X):
        # Use numpy's apply_along_axis function to apply _transform2 to all rows of X
        return np.apply_along_axis(self._transform2, 1, X)  
    
class MeanPooling(ParagraphEmbedding):
    def __init__(self, lemmatize:bool=False, lowercase:bool=False, remove_stopwords:bool=False):
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords
        
    # Vectorize a block of text
    def _transform1(self, sentence):

        # Get the tokenized list of words
        words = self._tokenize(sentence)

        # If the sentence is empty, return a vector of zeros
        if len(words) == 0:
            return np.zeros(300)

        # Get the normalized word vectors for each word in the sentence
        word_vectors = [normalize_vec(nlp.vocab[word].vector) for word in words]

        # Return the mean of the word vectors
        return np.mean(word_vectors, axis=0)
    
    # Vectorize a single row of the dataframe.
    def _transform2(self, row):

        # Concatenate the sentence vectors
        sent1_vec = self._transform1(row[0])
        sent2_vec = self._transform1(row[1])
        sent3_vec = self._transform1(row[2])
        return np.concatenate((sent1_vec, sent2_vec, sent3_vec), axis=0)
    
    def transform(self, X):
        # Use numpy's apply_along_axis function to apply _transform2 to all rows of X
        return np.apply_along_axis(self._transform2, 1, X)
    