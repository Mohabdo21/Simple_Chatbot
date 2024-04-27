"""
This script runs a chatbot model that was trained using intents
from a JSON file.
The chatbot uses a neural network trained on the intents and can
classify user inputs into these intents.
The neural network is implemented using the Keras library.

Usage:
    python chatbot.py
Inputs:
    intents.json - A JSON file containing the intents for the chatbot.
    words.pkl - A pickle file containing the preprocessed words.
    classes.pkl - A pickle file containing the preprocessed classes.
    chatbot_model.keras - The trained chatbot model.
Outputs:
    Interactive chatbot that can respond to user inputs based
    on trained intents.
"""
import json
import logging
import pickle
import random

import nltk
import numpy as np
from colorama import Fore, Style, init
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

logging.basicConfig(filename="chatbot.log", level=logging.INFO)

lemmatizer = WordNetLemmatizer()


def load_data():
    """Load model and data from files."""
    with open("intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
    with open("words.pkl", "rb") as file:
        words = pickle.load(file)
    with open("classes.pkl", "rb") as file:
        classes = pickle.load(file)
    model = load_model("chatbot_model.keras")
    return intents, words, classes, model


def clean_up_sentence(sentence):
    """Tokenize and lemmatize the sentence."""
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    """Convert sentence to bag of words."""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, words, classes, model):
    """Predict class using trained model."""
    # Convert sentence to bag of words
    bow = bag_of_words(sentence, words)
    # Predict class probabilities using the model
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25
    # Filter out low-probability predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # Sort predictions by probability
    results.sort(key=lambda x: x[1], reverse=True)
    # Prepare list of prediction results
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    """Get appropriate response for the intent."""
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result


init()


def main():
    """Main function to run the chatbot."""
    intents, words, classes, model = load_data()
    print(Fore.GREEN + "Go! Skill Recognition Bot is running!" + Style.RESET_ALL)

    while True:
        try:
            message = input(Fore.WHITE + "" + Style.RESET_ALL)
            ints = predict_class(message, words, classes, model)
            res = get_response(ints, intents)
            print(Fore.GREEN + res + Style.RESET_ALL)
            if ints[0]["intent"] == "goodbye":
                print(Fore.GREEN + "Chatbot is shutting down..." + Style.RESET_ALL)
                break
        except Exception as e:
            logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
