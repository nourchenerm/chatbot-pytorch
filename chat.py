import random  # Importation du module random pour la sélection aléatoire
import json  # Importation du module json pour la manipulation des fichiers JSON
import torch  # Importation de PyTorch pour les opérations de calcul numérique et d'apprentissage profond
from flask import Flask, request, jsonify  # Importation des composants Flask pour créer une API web
from flask_cors import CORS  # Importation de CORS pour permettre les requêtes cross-origin
from model import NeuralNet  # Importation de la classe NeuralNet définie dans le fichier model.py
from nltk_utils import bag_of_words, tokenize  # Importation des fonctions utilitaires de nltk_utils
from spellchecker import SpellChecker  # Importation de la bibliothèque pour la correction orthographique
from autocorrect import Speller  # Importation de la bibliothèque autocorrect pour la correction automatique
import re  # Importation du module re pour les expressions régulières

# Charger les intents et les données du modèle depuis un fichier JSON
with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)  # Chargement des données d'intents depuis le fichier JSON

# Charger les données du modèle depuis un fichier PyTorch
FILE = "data.pth"
data = torch.load(FILE)  # Chargement des données du modèle
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

# Initialisation du modèle de réseau de neurones
model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)  # Chargement de l'état du modèle sauvegardé
model.eval()  # Mise du modèle en mode évaluation (désactivation du dropout, etc.)

# Initialisation de l'application Flask
app = Flask(__name__)
CORS(app)  # Activation de CORS pour l'application Flask

# Fonction pour charger les données QA depuis un fichier JSON
def load_qa(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)  # Chargement des données QA depuis le fichier JSON

# Chargement des données QA
qa_data = load_qa("intents.json")

# Fonction pour extraire les mots-clés des intents
def extract_keywords_from_intents(qa_data, lang):
    keywords = []
    for intent in qa_data['intents']:
        if 'patterns' in intent and lang in intent['patterns']:
            keywords.extend(intent['patterns'][lang])
    return set(keywords)  # Retourner un ensemble de mots-clés uniques

# Initialisation des correcteurs orthographiques et des mots-clés
spell_fr = SpellChecker(language='fr')  # Correcteur orthographique pour le français
spell_en = Speller(lang='en')  # Correcteur orthographique pour l'anglais
# Extraction des mots-clés pour chaque langue
keywords_fr = extract_keywords_from_intents(qa_data, 'fr')
keywords_en = extract_keywords_from_intents(qa_data, 'en')
print("English keywords:", keywords_en)
# Fonction pour la correction orthographique personnalisée
def custom_spell_checker(input_text, keywords, spell_checker):
    words = input_text.split()  # Séparer le texte d'entrée en mots
    corrected_words = []
    for word in words:
        if word in keywords:
            corrected_words.append(word)  # Garder les mots qui sont déjà des mots-clés
        else:
            corrected_words.append(spell_checker(word))  # Corriger les autres mots
    return ' '.join(corrected_words)  # Rejoindre les mots corrigés en une seule chaîne





# Fonction pour diviser les messages en segments
def split_message(message):
    separators = r'[.,;!?&|]|\bet\b|\bou\b|\band\b|\bor\b'  # Séparateurs et mots-clés de séparation
    parts = re.split(separators, message)  # Diviser le message en utilisant les séparateurs
    parts = [part.strip() for part in parts if part.strip()]  # Nettoyer les parties et supprimer les chaînes vides
    return parts  # Retourner la liste des segments

# Définition de la route pour l'API chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    content = request.json  # Récupération des données JSON de la requête
    user_input = content.get('message', '')  # Récupération du message utilisateur
    langue = request.args.get('langue')  # Récupération de la langue de la requête, avec 'en' comme valeur par défaut

    print(f"Received input: {user_input}")
    print(f"Language: {langue}")

    # Correction orthographique du message utilisateur selon la langue
    if langue == 'fr':
        corrected_input = custom_spell_checker(user_input, keywords_fr, spell_fr.correction)
        print("Corrected input (FR):", corrected_input)
    else:
        corrected_input = custom_spell_checker(user_input, keywords_en, spell_en)
        print("Corrected input (EN):", corrected_input)

    # Division du message corrigé en segments
    segments = split_message(corrected_input)
    print("Segments:", segments)

    responses = []

    for segment in segments:
        # Tokenisation de chaque segment
        segment = tokenize(segment)
        print("Tokenizing:", segment)
        print("Preprocessed segment:", segment)
        X = bag_of_words(segment, all_words)  # Conversion des tokens en vecteurs de caractéristiques
        X = X.reshape(1, X.shape[0])  # Reshape pour correspondre à l'entrée du modèle
        print("Feature vector:", X)

        # Définition du périphérique (CPU ou GPU)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = torch.from_numpy(X).to(device)  # Conversion des vecteurs de caractéristiques en tenseurs PyTorch

        # Prédiction pour chaque segment
        output = model(X)  # Passage des données à travers le modèle
        _, predicted = torch.max(output, dim=1)  # Obtention du tag prédit

        probs = torch.softmax(output, dim=1)  # Calcul des probabilités
        prob = probs[0][predicted.item()]  # Probabilité associée à la classe prédite

        print(f"Predicted tag: {tags[predicted.item()]}, Probability: {prob.item()}")

        if prob.item() > 0.75:  # Seuil de confiance pour la réponse
            tag = tags[predicted.item()]
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    responses.append(random.choice(intent['responses'][langue]))  # Ajouter une réponse aléatoire

    if responses:
        answer = ' '.join(responses)  # Combiner toutes les réponses
        print("Response:", answer)
        return jsonify({"response": answer})  # Retourner la réponse au format JSON
    else:
        return jsonify({"response": "Je ne comprends pas..."})  # Réponse par défaut si aucune réponse n'est trouvée

# Exécution de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
