import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import pickle

def load_iris_from_uci():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    response = urllib.request.urlopen(url)
    lines = response.read().decode('utf-8').strip().splitlines()
    lines = [line for line in lines if line.strip()]
 
    species_to_label = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    X, y = [], []
    for line in lines:
        parts = line.split(',')
        if len(parts) != 5:
            continue
        features = list(map(float, parts[:4]))
        species = parts[4]
        X.append(features)
        y.append(species_to_label[species])
    return np.array(X), np.array(y)

X, y = load_iris_from_uci()
feature_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']
target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test   = X[split:], y[split:]

params = {} 
classes = np.unique(y_train)
for cls in classes:
    X_cls = X_train[y_train == cls]
    means = np.mean(X_cls, axis=0)
    variances = np.var(X_cls, axis=0)
    params[cls] = list(zip(means, variances))

def gaussian_pdf(x, mu, sigma):
    return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

num_features = X.shape[1]
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i in range(num_features):
    ax = axes[i]
    x_min = np.min(X[:, i]) - 1
    x_max = np.max(X[:, i]) + 1
    x_axis = np.linspace(x_min, x_max, 100)
    for cls in classes:
        mu, var = params[cls][i]
        sigma = np.sqrt(var)
        pdf_values = gaussian_pdf(x_axis, mu, sigma)
        ax.plot(x_axis, pdf_values, label=f'{target_names[cls]} (μ={mu:.2f}, σ²={var:.2f})')
    ax.set_title(feature_names[i])
    ax.legend()
plt.tight_layout()
plt.show()

data_to_save = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'params': params,
    'target_names': target_names,
    'feature_names': feature_names
}
with open('iris_model.pkl', 'wb') as f:
    pickle.dump(data_to_save, f)

print("Data got saved in iris_model.pkl")