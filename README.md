# YahooEmbeddings

Code for project in Text Mining course (732A81) at Linköping University during the fall of 2022. See folder paper for the final report.

## Abstract

Pretrained word embeddings are easily available and can be utilized to build representations of longer pieces of text. This paper investigates three simple strategies for representing paragraphs in a Q/A topic classification problem as well as for suitable common classifiers. The data used is the large scale Yahoo! Answers dataset which contains triples of question title, question content and best answer. The investigated approaches for paragraph representation are Distributed bag of words (DBOW), mean-pooling and projecting the word embeddings for an observation onto the first principal component. The DBOW and mean-pooling representations perform equally well with logistic regression (69% accuracy) and multilayer perceptron (71-72% accuracy). Other investigated models are SVM with linear and radial-basis function. The best model performs 4 percentage-points lower than the state-of-the-art on accuracy. Yet, the simplicity of the approaches show the power of pretrained word embeddings and simple solutions for representing longer pieces of text for topic classification in question and answer settings. SpaCy word embeddings are used throughout the study.

## Data

Data is available on [Google Drive](https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ). 

See the [official repo](https://github.com/LC-John/Yahoo-Answers-Topic-Classification-Dataset) for the data and the corresponding [paper](https://arxiv.org/abs/1509.01626).
