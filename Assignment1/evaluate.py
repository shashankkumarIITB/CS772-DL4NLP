from sklearn.metrics import confusion_matrix
import numpy as np

predicted = []
actual = []
with open('evaluate.txt') as file:
    lines = file.readlines()
    for line in lines:
        p, a = line.split('\n')[0].split(', ')
        predicted.append(int(p))
        actual.append(int(a))

# Train => loss: 0.1772 - accuracy: 0.9510 - precision: 0.9679 - recall: 0.9321
# Test =>  loss: 0.8543 - accuracy: 0.7356 - precision: 0.7687 - recall: 0.7120
# Confusion matrix => true label along row and predicted label along column
matrix = confusion_matrix(actual, predicted)
print('Insights into validation data')
print(matrix)

for i in range(5):
    print(f'Class {i + 1} => Total entries = {matrix[i].sum()}, Accuracy = {round(matrix[i, i] / matrix[i].sum(), 2)}')

print('Insights into predicted test labels')
test = {}
with open('predict.txt') as file:
    lines = file.readlines()
    for line in lines:
        p = line.split('\n')[0]
        if p in test.keys():
            test[p] += 1
        else:
            test[p] = 1

for key in test.keys():
    print(f'Class {key} => Total predicted = {test[key]}')