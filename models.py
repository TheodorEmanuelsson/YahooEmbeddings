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
    def __init__(self, lemmatize:bool=False, lowercase:bool=False, remove_stopwords:bool=False, use_mean:bool=False):
        self.lemmatize = lemmatize
        self.lowercase = lowercase
        self.remove_stopwords = remove_stopwords,
        self.use_mean = use_mean

    # Vectorize a block of text
    def _transform1(self, sentence):

        # Sum the word vectors for each word in the sentence
        words = self._tokenize(sentence)
        
        sent_vec = normalize_vec(nlp.vocab[words[0]].vector)

        for word in words[1:]:
            sent_vec += normalize_vec(nlp.vocab[word].vector)

        if self.use_mean:
            sent_vec /= len(words)

        return sent_vec

    # Vectorize a single row of the dataframe.
    def _transform2(self, row):

        # Concatenate the sentence vectors
        sent1_vec = self._transform1(row.question_title)
        sent2_vec = self._transform1(row.question_content)
        sent3_vec = self._transform1(row.best_answer)


        return np.concatenate((sent1_vec, sent2_vec, sent3_vec), axis=0)

    def transform(self, X):
        return np.concatenate(
            [self._transform2(row).reshape(1, -1) for row in X.itertuples()]
        )        