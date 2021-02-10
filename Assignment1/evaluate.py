from sklearn.metrics import confusion_matrix

predicted = []
actual = []
with open('evaluate.txt') as file:
    lines = file.readlines()
    for line in lines:
        p, a = line.split('\n')[0].split(', ')
        predicted.append(int(p))
        actual.append(int(a))
        line = file.readline()

# Train => loss: 0.1772 - accuracy: 0.9510 - precision: 0.9679 - recall: 0.9321
# Test =>  loss: 0.8543 - accuracy: 0.7356 - precision: 0.7687 - recall: 0.7120
# Confusion matrix => true label along row and predicted label along column
print(confusion_matrix(actual, predicted))
