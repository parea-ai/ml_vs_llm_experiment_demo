import os

from dotenv import load_dotenv
from parea import Parea, trace
from parea.schemas import Log

load_dotenv()

p = Parea(api_key=os.getenv("PAREA_API_KEY"))


def check_classification(log: Log) -> float:
    if log.output.lower() == log.target.lower():
        return 1
    return 0


# Directly take the outputs from the dataset CSV and return it
@trace(eval_funcs=[check_classification])
def rnn_prediction_from_csv(name: str, output: str) -> str:
    return output


# Assuming I have a CSV Dataset saved on Parea with the model outputs already filled in
if __name__ == "__main__":
    p.experiment(
        name="RNN_Experiment_With_OUTPUT",
        data="classification_test",
        func=rnn_prediction_from_csv,
    ).run()
