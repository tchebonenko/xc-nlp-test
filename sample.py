# python

from sklearn.feature_extraction.text import CountVectorizer
# list of text documents
# create the transform
vectorizer = CountVectorizer(ngram_range=(4, 4))
with open("./docsample.txt", "r") as f:
    vectorizer.fit(f)
    vector = vectorizer.transform(f)

# tokenize and build vocab
# summarize
print(vectorizer.vocabulary_)
# encode document
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())