# Chatbot Project

This repository contains the code for a simple chatbot trained on custom intents.
Inspired from youtube channel [**NeuralNine**](https://www.youtube.com/@NeuralNine), Thanks to Mr. Florian Dedov

## Files

- `chatbot.py`: This script runs the chatbot model that was trained using intents from a JSON file. The chatbot uses a neural network trained on the intents and can classify user inputs into these intents. The neural network is implemented using the Keras library.

- `intents.json`: A JSON file containing the intents for the chatbot.

- `model_training.py`: This script trains a chatbot model using intents from a JSON file. The chatbot uses a neural network trained on the intents and can classify user inputs into these intents. The neural network is implemented using the Keras library.

- `requirements.txt`: This file lists the Python dependencies that you need to install to run the project.

## Setup

1. Clone this repository.
2. Create a virtual environment: `python3 -m venv venv_bot`
3. Activate the virtual environment: `source venv_bot/bin/activate`
4. Install the dependencies: `pip install -r requirements.txt`

## Troubleshooting

- If you got error message indicating that the NLTK (Natural Language Toolkit) library is trying to use the ‘punkt’ tokenize:

1. Open your Python environment terminal.
2. Import the NLTK library by typing `import nltk`.
3. Download the ‘punkt’ package by typing `nltk.download('punkt')`.

```python
import nltk
nltk.download('punkt')
```

- Export the environment variable `TF_CPP_MIN_LOG_LEVEL` to suppress TensorFlow’s Warning messages:

```bash
export TF_CPP_MIN_LOG_LEVEL=2
```

## Usage

1. Train the model: `python model_training.py`
2. Run the chatbot: `python chatbot.py`
