import os
import string
import time

from dotenv import load_dotenv
from parea import Parea, trace
from parea.schemas import Log

from main.rnn.rnn import RNN, train_iterations, predict
from main.utils import get_category_lines, time_since

load_dotenv()

# Set up Parea client and wrap OpenAI client
p = Parea(api_key=os.getenv("PAREA_API_KEY"))

# Load data
all_categories, category_lines = get_category_lines('../data/names/*.txt')

# Evaluation dataset
dataset = [
    {"name": "Tannous", "target": "Arabic"},
    {"name": "Ferrero", "target": "Italian"},
    {"name": "Thian", "target": "Chinese"},
    {"name": "Montagne", "target": "French"},
    {"name": "Cheplakov", "target": "Russian"},
    {"name": "O'Ryan", "target": "Irish"},
    {"name": "Heidl", "target": "Czech"},
    {"name": "Skeril", "target": "Czech"},
    {"name": "Fergus", "target": "Irish"},
    {"name": "Diakogeorgiou", "target": "Greek"},
    {"name": "Vilaro", "target": "Spanish"},
    {"name": "Giles", "target": "French"},
    {"name": "Zangari", "target": "Italian"},
    {"name": "Bell", "target": "Scottish"},
    {"name": "Soseki", "target": "Japanese"},
    {"name": "Mcmillan", "target": "Scottish"},
    {"name": "Ho", "target": "Korean"},
    {"name": "Sniegowski", "target": "Polish"},
    {"name": "Accardi", "target": "Italian"},
    {"name": "Zawisza", "target": "Polish"}
]


# Train RNN model before running the experiment, but add the training latency to the metadata
@trace(metadata={"num_categories": len(all_categories)})
def train_rnn_model(n_iters: int = 100000, n_hidden: int = 128) -> tuple[RNN, str]:
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    start = time.time()
    n_categories = len(all_categories)
    rnn_model = RNN(n_letters, n_hidden, n_categories)
    train_iterations(rnn_model, all_categories, category_lines, n_iters)
    train_latency = time_since(start)
    return rnn_model, train_latency


rnn, latency = train_rnn_model()


# Set up eval function
def check_classification(log: Log) -> float:
    if log.output.lower() == log.target.lower():
        return 1
    return 0


@trace(eval_funcs=[check_classification], metadata={"all_categories": all_categories, "training_latency": latency})
def rnn_prediction(name: str) -> str:
    preds = predict(rnn, all_categories, name, n_predictions=1, verbose=False)
    return preds[0][1]


if __name__ == "__main__":
    p.experiment(
        name="RNN_Experiment",
        data=dataset,
        func=rnn_prediction,
    ).run(run_name="rnn_ml_model")
