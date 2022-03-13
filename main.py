import itertools
import sys
from multiprocessing import Pool, Process
import concurrent
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


def train_model(training_set, kernel, c):
    x, y = unzip(training_set)
    if kernel == 'quadratic':
        kernel = 'poly'
    clf = svm.SVC(kernel=kernel, degree=2, C=c)
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


def svf(kernel, c):
    global training_set, testing_set
    # Train model
    clf = train_model(training_set, kernel, c)
    # Find accuracy
    accuracy = find_accuracy(testing_set, clf)
    return accuracy


def main():
    global training_set, testing_set
    # Import data
    dataset = import_data(sys.argv[1])
    # Create training and testing sets
    training_set, testing_set = create_sets(dataset, 0.7)
    # For each kernel type
    kernels = ['linear', 'quadratic', 'rbf']
    cs = [0.1, 0.5, 1, 5, 10, 50, 100]

    # Process
    # processes = [Process(target=svf, args=(kernel,)) for kernel in kernels]
    # for process in processes:
    #     process.start()
    #     print(process.name)
    # for process in processes:
    #     process.join()

    # Process Pool
    tasks = list(itertools.product(kernels, cs))
    with Pool() as pool:
        async_result = pool.starmap_async(svf, tasks)
        pool.close()
        accuracies = async_result.get()
    print(f"Kernel \t C \t Accuracy")
    for kernel, c in tasks:
        print(f'{kernel} {c} {accuracies[tasks.index((kernel, c))]}')
    jobs = []

    # Process Pool Executor
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     jobs.append(executor.map(svf, kernels))
    #     for job in futures.as_completed(jobs):
    #         print(job.result())

    # Thread





if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python3 main.py <filename>')
        sys.exit(1)
    main()
