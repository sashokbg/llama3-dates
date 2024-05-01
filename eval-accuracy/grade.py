from promptflow.core import tool


@tool
def grade(ground_truth: str, prediction: str):
    return "Correct" if ground_truth.lower() == prediction.lower() else "Incorrect"
