from flask import Flask, render_template
import torch
import torch.nn as nn
import joblib
import time

# Load encoder + model
encoder = joblib.load("encoder.pkl")
vocab_size = len(encoder.classes_)

class LSTMPredictor(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 16)
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embed(x.unsqueeze(1))
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMPredictor(vocab_size)
model.load_state_dict(torch.load("zln_model.pt"))
model.eval()

app = Flask(__name__)

# Track user state and metrics
user_state = {"last_page": "home"}
metrics = {
    "total_clicks": 0,
    "prefetch_hits": 0,
    "total_latency": 0
}

@app.route("/")
def home():
    start_time = time.time()
    user_state["last_page"] = "home"

    # AI prediction
    idx = torch.tensor([encoder.transform(["home"])[0]])
    with torch.no_grad():
        out = model(idx)
        pred_idx = torch.argmax(out).item()
        predicted_page = encoder.inverse_transform([pred_idx])[0]

    # Update metrics (simulate latency measurement)
    latency = time.time() - start_time
    metrics["total_clicks"] += 1
    metrics["total_latency"] += latency

    return render_template("home.html", predicted=predicted_page,
                           total_clicks=metrics["total_clicks"],
                           prefetch_hits=metrics["prefetch_hits"],
                           avg_latency=metrics["total_latency"]/metrics["total_clicks"])

@app.route("/<page>")
def route_page(page):
    start_time = time.time()
    last_page = user_state["last_page"]
    user_state["last_page"] = page

    # AI prediction
    try:
        idx = torch.tensor([encoder.transform([page])[0]])
        with torch.no_grad():
            out = model(idx)
            pred_idx = torch.argmax(out).item()
            predicted_page = encoder.inverse_transform([pred_idx])[0]
    except:
        predicted_page = "unknown"

    # Prefetch hit simulation
    if predicted_page == page:
        metrics["prefetch_hits"] += 1

    # Measure latency
    latency = time.time() - start_time
    metrics["total_clicks"] += 1
    metrics["total_latency"] += latency

    # Compute avg latency and hit ratio
    avg_latency = metrics["total_latency"] / metrics["total_clicks"]
    hit_ratio = (metrics["prefetch_hits"] / metrics["total_clicks"]) * 100

    return render_template(f"{page}.html", predicted=predicted_page,
                           total_clicks=metrics["total_clicks"],
                           prefetch_hits=metrics["prefetch_hits"],
                           hit_ratio=hit_ratio,
                           avg_latency=avg_latency)

if __name__ == "__main__":
    app.run(debug=True)
