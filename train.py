import numpy as np  # Importation de NumPy pour les opérations mathématiques et la manipulation des tableaux
import random  # Importation de la bibliothèque random pour les opérations aléatoires
import json  # Importation de json pour la gestion des fichiers JSON

import torch  # Importation de PyTorch pour le calcul numérique et l'apprentissage profond
import torch.nn as nn  # Importation des modules de réseau de neurones de PyTorch
from torch.utils.data import Dataset, DataLoader  # Importation des outils pour la gestion des ensembles de données

from nltk_utils import bag_of_words, tokenize, stem  # Importation de fonctions de prétraitement du texte (à partir d'un fichier externe)
from model import NeuralNet  # Importation du modèle de réseau de neurones défini dans un fichier externe

# Chargement des données d'intention depuis un fichier JSON
with open('intents.json', 'r',encoding='utf-8') as f:
    intents = json.load(f)  # Lecture et chargement du fichier JSON contenant les intentions

all_words = []   # Liste pour stocker tous les mots extraits des modèles de phrases
tags = []  # Liste pour stocker toutes les étiquettes (tags) des intentions
xy = []  # Liste pour stocker les paires de modèles de phrases et de tags

# Définition des langues supportées
languages = ['fr', 'en']

# Boucle à travers chaque intention dans les modèles d'intention
for intent in intents['intents']:
    tag = intent['tag']  # Extraction de l'étiquette de l'intention
    tags.append(tag)  # Ajout de l'étiquette à la liste des tags
    for language in languages:
        if language in intent['patterns']:
            for pattern in intent['patterns'][language]:
                # Tokenisation de chaque mot dans la phrase
                w = tokenize(pattern)
                # Ajout des mots à la liste des mots
                all_words.extend(w)
                # Ajout de la paire (modèle de phrase, tag) à la liste xy
                xy.append((w, tag))

# Stemming et conversion en minuscules de chaque mot
ignore_words = ['?', '.', '!']  # Liste de mots à ignorer lors du stemming
all_words = [stem(w) for w in all_words if w not in ignore_words]  # Appliquer le stemming et ignorer les mots à ignorer
# Suppression des doublons et tri des mots
all_words = sorted(set(all_words))
tags = sorted(set(tags))  # Suppression des doublons et tri des tags

print(len(xy), "patterns")  # Affichage du nombre total de modèles de phrases
print(len(tags), "tags:", tags)  # Affichage du nombre total de tags et des tags eux-mêmes
print(len(all_words), "unique stemmed words:", all_words)  # Affichage du nombre total de mots uniques après stemming et les mots eux-mêmes

# Création des données d'entraînement
X_train = []  # Liste pour les vecteurs de caractéristiques des phrases
y_train = []  # Liste pour les étiquettes des phrases

# Boucle à travers chaque paire (modèle de phrase, tag)
for (pattern_sentence, tag) in xy:
    # X: bag of words pour chaque modèle de phrase
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: les étiquettes doivent être des indices de classes pour PyTorch CrossEntropyLoss
    label = tags.index(tag)
    y_train.append(label)

# Conversion des listes en tableaux NumPy
X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-paramètres pour l'entraînement du modèle
num_epochs = 1000  # Nombre d'époques d'entraînement
batch_size = 8  # Taille du lot pour l'entraînement
learning_rate = 0.001  # Taux d'apprentissage
input_size = len(X_train[0])  # Taille d'entrée (nombre de caractéristiques)
hidden_size = 8  # Taille de la couche cachée
output_size = len(tags)  # Taille de la couche de sortie (nombre de tags)
print(input_size, output_size)  # Affichage de la taille d'entrée et de sortie

# Définition de la classe ChatDataset pour gérer les données d'entraînement
class ChatDataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)  # Nombre d'échantillons dans le dataset
        self.x_data = X_train  # Données d'entrée
        self.y_data = y_train  # Étiquettes de sortie

    # Fonction pour obtenir un échantillon spécifique
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # Fonction pour obtenir la taille du dataset
    def __len__(self):
        return self.n_samples

# Création d'un objet ChatDataset
dataset = ChatDataset()
# Création du DataLoader pour le dataset
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)  # Chargement des données en lot, avec mélange aléatoire et sans utilisation de threads supplémentaires

# Définition du périphérique (CPU ou GPU) pour l'entraînement
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Création du modèle de réseau de neurones
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Définition de la fonction de perte et de l'optimiseur
criterion = nn.CrossEntropyLoss()  # Fonction de perte pour la classification multi-classe
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Optimiseur Adam

# Entraînement du modèle
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)  # Transfert des données d'entrée au périphérique (GPU ou CPU)
        labels = labels.to(dtype=torch.long).to(device)  # Transfert des étiquettes au périphérique

        # Passage avant (forward pass)
        outputs = model(words)  # Obtenir les sorties du modèle
        loss = criterion(outputs, labels)  # Calcul de la perte

        # Passage arrière et optimisation
        optimizer.zero_grad()  # Réinitialiser les gradients
        loss.backward()  # Calcul des gradients
        optimizer.step()  # Mise à jour des poids du modèle
        
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')  # Affichage de la perte tous les 100 epochs

print(f'final loss: {loss.item():.4f}')  # Affichage de la perte finale après l'entraînement

# Sauvegarde de l'état du modèle et des hyper-paramètres
data = {
    "model_state": model.state_dict(),  # État des poids du modèle
    "input_size": input_size,  # Taille d'entrée
    "hidden_size": hidden_size,  # Taille de la couche cachée
    "output_size": output_size,  # Taille de la couche de sortie
    "all_words": all_words,  # Liste des mots uniques après stemming
    "tags": tags  # Liste des tags
}

FILE = "data.pth"  # Nom du fichier pour sauvegarder les données
torch.save(data, FILE)  # Sauvegarde des données dans le fichier

print(f'training complete. file saved to {FILE}')  # Message indiquant la fin de l'entraînement et la sauvegarde du fichier
