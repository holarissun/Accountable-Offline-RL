import itertools
from shapely.geometry import Point, Polygon
import torch
import numpy as np
from scipy.spatial import ConvexHull

# Constants
MAX_HULL_POINTS = 10

def find_min_hull_heuristic(points, target, start_k):
    min_area = np.inf
    min_hull = None
    min_indices = None
    k = start_k
    used_combinations = set()

    while min_hull is None and k <= 10:
        # Get k nearest neighbors
        dist = ((points - target) ** 2).sum(1)
        _, idx = torch.sort(torch.as_tensor(dist))
        nearest_points = points[idx[:k]]

        for combo in itertools.combinations(range(len(nearest_points)), 3):
            # Skip combinations we've already tried
            if combo in used_combinations:
                continue
            used_combinations.add(combo)
            
            poly = Polygon([nearest_points[i] for i in combo])
            if poly.contains(Point(target)):
                if poly.area < min_area:
                    min_area = poly.area
                    min_hull = [nearest_points[i] for i in combo]
                    min_indices = [idx[i] for i in combo]

        k += 1

    return min_hull, min_indices

def find_min_hull_heuristic_high_dim(points, target, start_k, max_hull_points=10):
    """
    Find the smallest convex hull containing the target point.
    If no such hull is found, return None.
    """
    min_volume = np.inf
    min_hull = None
    min_indices = None
    k = start_k
    d = len(target)  # dimension of the space

    while min_hull is None and k <= max_hull_points:
        # Get the k nearest points
        dist = ((points - target) ** 2).sum(1)
        _, idx = torch.sort(torch.as_tensor(dist))
        nearest_points = points[idx[:k]]

        for combo in itertools.combinations(range(len(nearest_points)), d+1):
            # Compute the convex hull of the points in the combo
            combo_points = nearest_points[list(combo)]
            hull = ConvexHull(combo_points)

            # Check if the target is in the convex hull
            eqs = hull.equations.T
            if np.all(np.dot(eqs[:-1].T, target) + eqs[-1] <= 0):
                # Update min_volume, min_hull, and min_indices if this hull's volume is smaller
                if hull.volume < min_volume:
                    min_volume = hull.volume
                    min_hull = combo_points
                    min_indices = [idx[i] for i in combo]

        k += 1

    return min_hull, min_indices



# def find_min_hull_heuristic(points, target, start_k):
#     """
#     Find the smallest convex hull containing the target point.
#     If no such hull is found, return None.
#     """
#     min_area = np.inf
#     min_hull = None
#     min_indices = None
#     k = start_k
#     used_combinations = set()

#     while min_hull is None and k <= MAX_HULL_POINTS:
#         dist = ((points - target) ** 2).sum(1)
#         _, idx = torch.sort(torch.as_tensor(dist))
#         nearest_points = points[idx[:k]]

#         for combo in itertools.combinations(range(len(nearest_points)), 3):
#             if combo in used_combinations:
#                 continue
#             used_combinations.add(combo)

#             poly = Polygon([nearest_points[i] for i in combo])
#             if poly.contains(Point(target)) and poly.area < min_area:
#                 min_area = poly.area
#                 min_hull = [nearest_points[i] for i in combo]
#                 min_indices = [idx[i] for i in combo]

#         k += 1

#     return min_hull, min_indices

def LP_solver_minimalhull(
    test_latent_reps: torch.Tensor,
    train_latent_reps: torch.Tensor,
    n_epoch: int = 100,
    start_k: int = 3,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Find the minimal convex hull for each test point and optimize the weights
    for each point in the hull to minimize the mean squared error.
    """
    n_test = test_latent_reps.shape[0]
    idx_list = torch.zeros((n_test, start_k), dtype=torch.long)

    for i in range(n_test):
        num_points_in_hull = start_k
        min_hull, indices = find_min_hull_heuristic(train_latent_reps.cpu().detach().numpy(), test_latent_reps[i].cpu().detach().numpy(), start_k)
        if min_hull is None:
            dist = ((train_latent_reps - test_latent_reps[i]) ** 2).sum(1)
            _, idx = torch.sort(dist)
            indices = idx[:start_k]
        idx_list[i] = torch.as_tensor(indices)

    train_latent_reps = train_latent_reps[idx_list]

    preweights = torch.zeros(
        (n_test, num_points_in_hull),
        device=device,
        requires_grad=True,
    ).to(device)

    optimizer = torch.optim.Adam([preweights])
    error_list = []

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        weights = F.softmax(preweights, dim=-1)
        corpus_latent_reps = (weights.unsqueeze(-1) * train_latent_reps).sum(1)
        error = ((corpus_latent_reps - test_latent_reps) ** 2).mean()
        error_list.append(error.item())
        loss = error
        loss.backward()
        optimizer.step()

        if (epoch + 1) % (n_epoch/10) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{n_epoch} ; Error: {error.item():.3g} ;"
            )

    final_error = ((corpus_latent_reps - test_latent_reps) ** 2).sum(1).cpu().detach()
    weights = F.softmax(preweights, dim=-1).cpu().detach()
    return weights, idx_list, final_error



import pickle
# Load
with open('Dataset/training_data.pkl', 'rb') as f:
    static_train, observations_train, actions_train = pickle.load(f)

with open('Dataset/testing_data.pkl', 'rb') as f:
    static_test, observations_test, actions_test = pickle.load(f)

import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import medkit as mk
import numpy as np
N_TRAIN = 50000
MAX_LEN = 10
HIDDEN = 2
baselines_out = []
repeat_out = []
sampling_rate = 1.0

# static_train shape is (1000, 49)
# observations_train shape is (1000, 10, 35)
# actions_train shape is (1000, 10, 1)

combined_feature_shape = static_train.shape[1] + observations_train.shape[2]

train_x = np.concatenate([static_train[:, np.newaxis, :].repeat(MAX_LEN, 1), observations_train], axis=2).reshape(-1, combined_feature_shape)
train_y = actions_train.reshape(-1, 1)
test_x = np.concatenate([static_test[:, np.newaxis, :].repeat(MAX_LEN, 1), observations_test], axis=2).reshape(-1, combined_feature_shape)
test_y = actions_test.reshape(-1, 1)

sample_size = int(sampling_rate * len(train_x))
sampled_indices = np.random.choice(len(train_x), size=sample_size, replace=False)

train_x = train_x[sampled_indices]
train_y = train_y[sampled_indices]

print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

# train a linear classifier to predict actions from observations
classifier = LogisticRegression()
classifier.fit(train_x, train_y)

# evaluate the classifier on the test set
accuracy = accuracy_score(test_y, classifier.predict(test_x))
print("LR Accuracy: {:.2f}%".format(accuracy * 100))
baselines_out.append(accuracy)
# Neural Network classifier
classifier = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000)
classifier.fit(train_x, train_y)

# evaluate the classifier on the test set
accuracy = accuracy_score(test_y, classifier.predict(test_x))
print("MLP Accuracy: {:.2f}%".format(accuracy * 100))
baselines_out.append(accuracy)
# Build a neural network to predict actions from observations with Pytorch

# convert data to torch tensors

#     train_x = np.concatenate([static_train[:, np.newaxis, :].repeat(MAX_LEN, 1), observations_train], axis=2).reshape(-1, combined_feature_shape)
#     train_y = actions_train.reshape(-1, 1)
#     test_x = np.concatenate([static_test[:, np.newaxis, :].repeat(MAX_LEN, 1), observations_test], axis=2).reshape(-1,  combined_feature_shape)
#     test_y = actions_test.reshape(-1, 1)
#     train_x = train_x[sampled_indices]
#     train_y = train_y[sampled_indices]

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).long().squeeze()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).long().squeeze()
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(combined_feature_shape, 100)
        self.fc2 = nn.Linear(100, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_features(self, x):
        x = F.relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return x

net = Net().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

# train the network
for epoch in range(5000):
    optimizer.zero_grad()
    outputs = net(train_x.cuda())
    loss = criterion(outputs, train_y.cuda())
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        # print("Epoch: {}, loss: {:.4f}".format(epoch, loss.item()))
        # accuracy
        accuracy = accuracy_score(train_y.detach().numpy(), np.argmax(outputs.cpu().detach().numpy(), axis=1))
# evaluate the network on the test set
with torch.no_grad():
    outputs = net(test_x.cuda())
    loss = criterion(outputs, test_y.cuda())

    # print("Test loss: {:.4f}".format(loss.item()))
    # accuracy
    accuracy_mlp = accuracy_score(test_y.detach().numpy(), np.argmax(outputs.cpu().detach().numpy(), axis=1))
    print("Pytorch MLP Test accuracy: {:.2f}%".format(accuracy_mlp * 100))
baselines_out.append(accuracy_mlp)
train_features = net.forward_features(train_x.cuda()).cpu().detach().numpy()
train_outputs = net(train_x.cuda()).cpu().detach().numpy()
test_features = net.forward_features(test_x.cuda()).cpu().detach().numpy()
test_outputs = net(test_x.cuda()).cpu().detach().numpy()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Assuming you have training_data and test_data with shapes (50000, 10) and (5000, 10) respectively
# training_data = np.random.rand(50000, 10)
# test_data = np.random.rand(5000, 10)

# Combine the training and test-time datasets
data = np.vstack((train_features, test_features))

# Standardize the data (optional but recommended)
scaler = StandardScaler()
data_std = scaler.fit_transform(data)

data_tsne = np.vstack((train_features, test_features))

# Plot the t-SNE clustering map
plt.scatter(data_tsne[:int(N_TRAIN*sampling_rate), 0], data_tsne[:int(N_TRAIN*sampling_rate), 1], label="Training Data", alpha=0.5)
plt.scatter(data_tsne[int(N_TRAIN*sampling_rate):, 0], data_tsne[int(N_TRAIN*sampling_rate):, 1], label="Test Data", alpha=0.5)
plt.legend()
plt.title("Latent Space Map")
plt.xlabel("Latent Dim 1")
plt.ylabel("Latent Dim 2")
plt.show()



def LP_solver_KNN(
    test_latent_reps: torch.Tensor,
    train_latent_reps: torch.Tensor,
    n_epoch: int = 100,
    n_keep: int = 20,
) -> None:
    corpus_size = train_latent_reps.shape[0]
    n_test = test_latent_reps.shape[0]
    preweights = torch.zeros(
        (n_test, n_keep),
        device=test_latent_reps.device,
        requires_grad=True,
    ).cuda()
    optimizer = torch.optim.Adam([preweights])
    idx_list = torch.zeros((n_test, n_keep), dtype=torch.long)
    # generate kNN masks
    # find the k-nearest neighbors of each test point
    for i in range(n_test):
        dist = ((train_latent_reps - test_latent_reps[i]) ** 2).sum(1)
        _, idx = torch.sort(dist)
        idx_list[i] = idx[:n_keep]

    # select index of k-nearest neighbors
    train_latent_reps = train_latent_reps[idx_list]

    for epoch in range(n_epoch):
        optimizer.zero_grad()
        weights = F.softmax(preweights, dim=-1)
        # weights shape is (n_test, n_keep)
        # train_latent_reps shape is (n_test, n_keep, representation_dim)
        # time those two together:
        corpus_latent_reps = (weights.unsqueeze(-1) * train_latent_reps).sum(1)
        error = ((corpus_latent_reps - test_latent_reps) ** 2).mean()
        weights_sorted = torch.sort(weights)[0]
#             regulator = (weights_sorted[:, : (corpus_size - n_keep)]).sum()
        loss = error #+ regulator
        loss.backward()
        optimizer.step()
        if (epoch + 1) % (n_epoch/10) == 0:
            print(
                f"Weight Fitting Epoch: {epoch+1}/{n_epoch} ; Error: {error.item():.3g} ;"
            )
    final_error = ((corpus_latent_reps - test_latent_reps) ** 2).sum(1).cpu().detach()
    weights = torch.softmax(preweights, dim=-1).cpu().detach()
    return weights, idx_list, final_error
for repeat in range(10):
    results_out = []

    for keep_number in [1, 3, 5, 10, 20, 30, 50, 100]:
        sub_result = []
        weights, idx_list, out_err = LP_solver_minimalhull(test_latent_reps=torch.as_tensor(test_features).cuda(),
                                                   train_latent_reps=torch.as_tensor(train_features).cuda(),
                                                   n_epoch=10000,
                                                   start_k=keep_number)
        weights = weights.cpu().detach().numpy()
        convex_pred = np.argmax(np.mean(weights[:, :, np.newaxis] * train_outputs[idx_list], axis=1), axis=1)
        accuracy = accuracy_score(test_y.detach().numpy(), convex_pred)
        print(f"ABC Accuracy K={keep_number:.2f}: {accuracy * 100:.2f}%")
        sub_result.append(accuracy)
        knn_pred = np.argmax(np.mean(train_outputs[idx_list], axis=1), axis=1)
        accuracy = accuracy_score(test_y.detach().numpy(), knn_pred)
        print(f"Avg Latent Accuracy K={keep_number:.2f}: {accuracy * 100:.2f}%")
        sub_result.append(accuracy)

        weights, idx_list, out_err = LP_solver_KNN(test_latent_reps=test_x.cuda(),
                                                   train_latent_reps=train_x.cuda(),
                                                   n_epoch=10000,
                                                   n_keep=keep_number)
        weights = weights.cpu().detach().numpy()
        convex_pred = np.argmax(np.mean(weights[:, :, np.newaxis] * train_outputs[idx_list], axis=1), axis=1)
        accuracy = accuracy_score(test_y.detach().numpy(), convex_pred)
        print(f"Convex KNN Accuracy K={keep_number:.2f}: {accuracy * 100:.2f}%")
        sub_result.append(accuracy)
        knn_pred = np.argmax(np.mean(train_outputs[idx_list], axis=1), axis=1)
        accuracy = accuracy_score(test_y.detach().numpy(), knn_pred)
        print(f"Avg KNN Accuracy K={keep_number:.2f}: {accuracy * 100:.2f}%")
        sub_result.append(accuracy)

        results_out.append(sub_result)
    repeat_out.append(results_out)
