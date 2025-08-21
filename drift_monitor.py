
import json
from pathlib import Path

LOG_PATH = Path("accuracy_log.json")

def log_prediction(real_digit, predicted_top3, position):
    log = load_log()
    correct = int(real_digit in predicted_top3)
    log.append({"pos": position, "correct": correct})
    save_log(log)

def load_log():
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return []

def save_log(log):
    LOG_PATH.write_text(json.dumps(log[-200:], indent=2))  # keep recent 200

def calculate_accuracy_per_position():
    log = load_log()
    pos_accuracy = {i: [] for i in range(4)}
    for entry in log:
        pos_accuracy[entry["pos"]].append(entry["correct"])
    return {k: round(np.mean(v)*100, 2) if v else 0 for k,v in pos_accuracy.items()}
