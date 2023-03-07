# # This is a sample Python script.
#
# # Press Shift+F10 to execute it or replace it with your code.
# # Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#
#
# def print_hi(name):
#     # Use a breakpoint in the code line below to debug your script.
#     print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print_hi('PyCharm')
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

from utlis import *
from algos import perceptron


path = 'reviews_train.tsv'
test_path = 'reviews_test.tsv'

data = load_data(path)
test_data = load_data(test_path)
train_text, train_label = zip(*((sample['text'], sample['sentiment']) for sample in data))
test_text, test_label = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

with open('stopwords.txt', 'r') as fp:
    stop_words = fp.read()
    fp.close()

dictionary = bag_of_words(train_text, stop_words)

train_features = feature_matrix(train_text, dictionary)
test_features = feature_matrix(test_text, dictionary)

print(train_features.shape)
print(test_features.shape)


theta, theta_0 = perceptron(train_features, train_label, 10)
print(theta.shape)

train_accuracy, test_accuracy = classifier_accuracy(perceptron,
                                                    train_features,
                                                    test_features,
                                                    train_label,
                                                    test_label,
                                                    T=10)
print('train_accuracy : ', train_accuracy)
print('test_accuracy : ', test_accuracy)
