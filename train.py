# Mengimport Libraries
import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

#Import Datset Json
with open("intents.json", "r") as f:
    intents = json.load(f)

#Menyiapkan data training
all_words = [] 
tags = []
xy = []
# mengulang setiap kalimat dalam pola maksud kita
for intent in intents["intenst"]:
    tag = intent["tag"]
    # Menambahkan tag list
    tags.append(tag)
    for pattern in intent["patterns"]:
        # Membagi setiap kata dalam kalimat menjadi token
        w = tokenize(pattern)
        # Menambahkan daftar kata
        all_words.extend(w)
        xy.append((w, tag))

# stem and lower Setiap kata
ignore_words = ["?", ".", "!"] #mengabaikan tanda spesial karakter
all_words = [stem(w) for w in all_words if w not in ignore_words]
# Menghapus duplikat dan urutkan
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Membuat model latih data
X_train = []
y_train = []
for pattern_sentence, tag in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss needs only class labels
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyperparameter model
# Tujuan dari hyperparameter model ini adalah untuk mencapai keseimbangan yang baik antara model yang cocok dengan data pelatihan 
# dan kemampuan model untuk melakukan prediksi yang baik pada data baru yang belum pernah dilihat sebelumnya. 
num_epochs = 1000
batch_size = 32
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 32
output_size = len(tags)
print(input_size, output_size)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Melatih  model
loss_history = []
accuracy_history = []

for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0

    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # menghitung loss
        outputs = model(words)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Menghitung Accuracy dari Model
        _, predicted = torch.max(outputs, 1)
        total_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

    if (epoch + 1) % 100 == 0:
        accuracy = total_correct / total_samples
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}"
        )

print(f"final loss: {loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags,
}

FILE = "data.pth"
torch.save(data, FILE)

print(f"training complete. file saved to {FILE}")
