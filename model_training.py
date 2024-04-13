"""
This script trains a chatbot model using intents from a JSON file.
The chatbot uses a neural network trained on the intents and can
classify user inputs into these intents.
The neural network is implemented using the Keras library.

Usage:
    python train_chatbot.py
Inputs:
    intents.json - A JSON file containing the intents for the chatbot.
Outputs:
    words.pkl - A pickle file containing the preprocessed words.
    classes.pkl - A pickle file containing the preprocessed classes.
    chatbot_model.keras - The trained chatbot model.
"""
import json
import logging
import pickle

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Set up logging
logging.basicConfig(filename="chatbot.log", level=logging.INFO)

# Constants
IGNORE_LETTERS = ["?", "!", ".", ","]

lemmatizer = WordNetLemmatizer()


def load_intents():
    """Load intents from json file."""
    with open("intents.json", "r", encoding="utf-8") as file:
        intents = json.load(file)
    return intents


def preprocess_data(intents):
    """Preprocess data for training."""
    words = []
    classes = []
    documents = []
    for intent in intents["intents"]:
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

        for pattern in intent["patterns"]:
            word_list = nltk.word_tokenize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent["tag"]))

    words = [
        lemmatizer.lemmatize(word.lower())
        for word in words
        if word not in IGNORE_LETTERS
    ]
    words = sorted(set(words))
    classes = sorted(set(classes))

    return words, classes, documents


def save_data(words, classes):
    """Save preprocessed data to files."""
    with open("words.pkl", "wb") as file:
        pickle.dump(words, file)
    with open("classes.pkl", "wb") as file:
        pickle.dump(classes, file)


def create_training_data(documents, words, classes):
    """Create training data."""
    train_x = []
    train_y = []
    output_empty = [0] * len(classes)

    for document in documents:
        bag = []
        word_patterns = document[0]
        word_patterns = [
                lemmatizer.lemmatize(word.lower()) for word in word_patterns
                ]
        for word in words:
            bag.append(1 if word in word_patterns else 0)

        output_raw = output_empty.copy()
        output_raw[classes.index(document[1])] = 1
        train_x.append(bag)
        train_y.append(output_raw)

    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y


def create_model(training_inputs, training_labels):
    """Create and train the model."""
    model = Sequential()
    # Input layer
    model.add(Input(shape=(len(training_inputs[0]),)))
    # First hidden layer with 128 neurons and ReLU activation function
    model.add(Dense(128, activation="relu"))
    # Dropout layer to prevent overfitting
    model.add(Dropout(0.5))
    # Second hidden layer with 64 neurons and ReLU activation function
    model.add(Dense(64, activation="relu"))
    # Another dropout layer
    model.add(Dropout(0.5))
    # Output layer with softmax activation function
    model.add(Dense(len(training_labels[0]), activation="softmax"))

    # Using SGD optimizer with Nesterov momentum for training
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    # Compiling the model with categorical crossentropy loss function
    # and accuracy metrics
    model.compile(
            loss="categorical_crossentropy",
            optimizer=sgd,
            metrics=["accuracy"]
            )

    # Training the model for 2000 epochs
    chat_model = model.fit(
        training_inputs, training_labels, epochs=2000, batch_size=5, verbose=1
    )
    return model, chat_model


def main():
    """Main function to run the chatbot."""
    intents = load_intents()
    words, classes, documents = preprocess_data(intents)
    save_data(words, classes)
    train_x, train_y = create_training_data(documents, words, classes)
    model, chat_model = create_model(train_x, train_y)
    model.save("chatbot_model.keras", chat_model)

    print("Done")


if __name__ == "__main__":
    main()
