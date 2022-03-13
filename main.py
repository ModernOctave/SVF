import sys
from sklearn import svm
from random import shuffle


def unzip(dataset):
    dataset = list(zip(*dataset))
    return dataset[0], dataset[1]


def import_data(filename):
    x = []
    y = []
    ifile = open(filename, 'r')
    for line in ifile:
        line = line.strip('\n')
        data = line.split(',')
        x.append(data[:-1])
        y.append(data[-1])
    ifile.close()
    dataset = list(zip(x, y))
    return dataset


def create_sets(dataset, train_percent):
    shuffle(dataset)
    training_set = dataset[:int(len(dataset) * train_percent)]
    testing_set = dataset[int(len(dataset) * train_percent):]
    return training_set, testing_set


def train_model(training_set, kernel):
    x, y = unzip(training_set)
    if kernel == 'quadratic':
        kernel = 'poly'
    clf = svm.SVC(kernel=kernel, degree=2)
    clf.fit(x, y)
    return clf


def find_accuracy(testing_set, clf):
    x, y = unzip(testing_set)
    predictions = clf.predict(x)
    correct = 0
    for i in range(len(predictions)):
        if predictions[i] == y[i]:
            correct += 1
    return (correct / len(predictions)) * 100


def main():
    # Import data
    dataset = import_data(sys.argv[1])
    # Create training and testing sets
    training_set, testing_set = create_sets(dataset, 0.7)
    # For each kernel type
    kernels = ['linear', 'quadratic', 'rbf']
    for kernel in kernels:
        # Train the model
        clf = train_model(training_set, kernel)
        # Find the accuracy of the model
        accuracy = find_accuracy(testing_set, clf)
        print(f'Accuracy for {kernel} kernel is: {accuracy}')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 main.py <filename>')
        sys.exit(1)
    main()
