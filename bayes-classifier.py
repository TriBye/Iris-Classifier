import numpy as np
import pickle

with open('iris_model.pkl', 'rb') as f:
    data = pickle.load(f)
X_train = data['X_train']
y_train = data['y_train']
params = data['params']
target_names = data['target_names']

np.random.seed(42)
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train = y_train[indices]

classes, counts = np.unique(y_train, return_counts=True)
priors = {cls: count / len(y_train) for cls, count in zip(classes, counts)}

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

predictions = []
for sample in X_train:
    class_probs = {}
    for cls in classes:
        prob = priors[cls]
        for feature_idx, x in enumerate(sample):
            mu, var = params[cls][feature_idx]
            sigma = np.sqrt(var)
            prob *= gaussian_pdf(x, mu, sigma)
        class_probs[cls] = prob
    predicted_class = max(class_probs, key=class_probs.get)
    predictions.append(predicted_class)
predictions = np.array(predictions)

print("Index | Prediction        | Class")
print("-------------------------------------------")
for idx, (pred, actual) in enumerate(zip(predictions, y_train)):
    print("{:5d} | {:15s} | {:15s}".format(idx, target_names[pred], target_names[actual]))

accuracy = np.mean(predictions == y_train)
print("\nAccuracy: {:.2f}%".format(accuracy * 100))