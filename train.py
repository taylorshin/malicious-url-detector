import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataset import load_data, get_tokens

def train():
    data = load_data('data.csv')
    y = [d[1] for d in data]
    corpus = [d[0] for d in data]
    vectorizer = TfidfVectorizer(tokenizer=get_tokens)
    X = vectorizer.fit_transform(corpus)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lgs = LogisticRegression(solver='liblinear', verbose=1)
    lgs.fit(X_train, y_train)
    print('Test Accuracy: ', lgs.score(X_test, y_test))

def main():
    parser = argparse.ArgumentParser(description='Trains the model')
    args = parser.parse_args()

    # Training
    train()


if __name__ == '__main__':
    main()
