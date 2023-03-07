from string import punctuation, digits
import csv
import numpy as np


def load_data(path, extras=False):
    # extras: variable define to take all the fields of data
    basic_fields = {'sentiment', 'text'}
    numeric_fields = {'sentiment', 'helpfulY', 'helpfulN'}

    data = []
    f_path = open(path, encoding="latin1")
    for datum in csv.DictReader(f_path, delimiter='\t'):
        for field in list(datum.keys()):
            if not extras and field not in basic_fields:
                del datum[field]
            elif field in numeric_fields and datum[field]:
                datum[field] = int(datum[field])

        data.append(datum)
    f_path.close()

    return data


def extract_words(input_string):
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')
    return input_string.lower().split()


def bag_of_words(data, stop_words):
    dictionary = {}
    for input_string in data:
        words = extract_words(input_string)
        for word in words:
            if word in stop_words:
                del word
            elif word not in dictionary:
                dictionary[word] = len(dictionary)
    return dictionary


def feature_matrix(data, dictionary):
    features = np.zeros([len(data), len(dictionary)])
    print(features.shape)

    for i, exp in enumerate(data):
        words = extract_words(exp)
        for word in words:
            if word in dictionary:
                features[i, dictionary[word]] = 1

    return features


def classify(feature_matrix, theta, theta_0):
    predicted = np.zeros(feature_matrix.shape[0])
    for i in range(feature_matrix.shape[0]):
        if (np.dot(feature_matrix[i], theta) + theta_0) > 0:
            predicted[i] = 1
        else:
            predicted[i] = -1
    return predicted


def accuracy(preds, targets):
    return (preds == targets).mean()


def classifier_accuracy(classifier,
                        train_features,
                        test_features,
                        train_labels,
                        test_labels,
                        **kwargs):

    theta, theta_0 = classifier(train_features,train_labels, **kwargs)
    predictions = classify(train_features,theta, theta_0)
    train_accuracy = accuracy(predictions,train_labels)

    validation_predictions = classify(test_features, theta, theta_0)
    test_accuracy = accuracy(validation_predictions,test_labels)
    return train_accuracy, test_accuracy