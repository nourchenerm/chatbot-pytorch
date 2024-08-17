import random  
import json  
import torch 
import spacy
from flask import Flask, request, jsonify  
from flask_cors import CORS  
from model import NeuralNet  
from nltk_utils import bag_of_words, tokenize  
from spellchecker import SpellChecker  
from autocorrect import Speller  
import re 
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as en_stop_words
from spacy.tokens import Doc 

nlp_fr = spacy.load('fr_core_news_md')
nlp_en = spacy.load('en_core_web_md')

# Charger les intents et les données du modèle depuis un fichier JSON
with open('intents.json', 'r',encoding='utf-8') as json_data:
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

spell_fr = SpellChecker(language='fr')  
spell_en = Speller(lang='en')  

keywords_fr = extract_keywords_from_intents(qa_data, 'fr')
keywords_en = extract_keywords_from_intents(qa_data, 'en')

def preprocess_text(text, language):
    if language == 'fr':
        doc = nlp_fr(text.lower())
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space and token.text not in fr_stop_words]
    elif language == 'en':
        doc = nlp_en(text.lower())
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space and token.text not in en_stop_words]
    else:
        doc = nlp_fr(text.lower())  # Default to French if language is not supported
        tokens = [token.text for token in doc if not token.is_punct and not token.is_space and token.text not in fr_stop_words]
    
    return ' '.join(tokens)
# Extract keywords from intents.json for custom spell checking
def extract_keywords_from_intents(qa_data, lang):
    keywords = []
    for intent in qa_data['intents']:
        if 'patterns' in intent and lang in intent['patterns']:
            keywords.extend(intent['patterns'][lang])
    return set(keywords)




def custom_spell_checker(input_text, keywords, spell_checker):
    words = input_text.split()  # Séparer le texte d'entrée en mots
    corrected_words = []
    for word in words:
        if word in keywords:
            corrected_words.append(word)  # Garder les mots qui sont déjà des mots-clés
        else:
            corrected_words.append(spell_checker(word))  # Corriger les autres mots
    return ' '.join(corrected_words)  # Rejoindre les mots corrigés en une seule chaîne
def get_closest_question(user_input, questions, language):
    best_similarity = 0.0
    closest_question = None

    user_input_processed = preprocess_text(user_input, language)

    for question in questions:
        question_processed = preprocess_text(question, language)
        if language == 'fr':
            similarity = nlp_fr(user_input_processed).similarity(nlp_fr(question_processed))
        elif language == 'en':
            similarity = nlp_en(user_input_processed).similarity(nlp_en(question_processed))
        else:
            continue  
        
        if similarity > best_similarity:
            best_similarity = similarity
            closest_question = question
    
    return closest_question




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
        questions = []
        question_answer_map = {}

        # Extract questions and corresponding answers from QA data
        for intent in qa_data["intents"]:
            if langue in intent["patterns"]:
                for pattern in intent["patterns"][langue]:
                    questions.append(pattern)
                    question_answer_map.setdefault(pattern, set()).update(intent["responses"][langue])

        # Find the closest question for this part
        closest_question = get_closest_question(segment, questions, langue)
        print('gjhnknlk,p',closest_question)
        # Tokenisation de chaque segment
        closest_question = tokenize(closest_question)
        print("Tokenizing:", closest_question)
        print("Preprocessed closest_question:", closest_question)
        X = bag_of_words(closest_question, all_words)  # Conversion des tokens en vecteurs de caractéristiques
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
        response = {"message": answer}
        return jsonify(response), 200, {'Content-Type': 'application/json; charset=utf-8'}
    else:

        return jsonify({"message": "Je ne comprends pas..."})  # Réponse par défaut si aucune réponse n'est trouvée

# Exécution de l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
