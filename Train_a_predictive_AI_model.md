 **1️⃣ Install Required Libraries**

```python
!pip install torch torchvision torchaudio
!pip install scikit-learn pandas matplotlib
```

* Installs **PyTorch** for deep learning and **scikit-learn, pandas, matplotlib** for data processing and analysis.

---

 **2️⃣ Prepare Navigation Data**

```python
logs = [
    ["home", "products", "cart", "checkout"],
    ["home", "about", "contact"],
    ...
]
```

* Simulates **user navigation sequences** on your website.
* Converts navigation into **pairs** like `(current_page → next_page)` for training.

---

 **3️⃣ Encode Pages**

```python
encoder = LabelEncoder()
encoder.fit(all_pages)
X = encoder.transform(df["current"])
y = encoder.transform(df["next"])
```

* Converts page names (like `"home"`) into **numerical labels** for the model.

---

 **4️⃣ Define LSTM Model**

```python
class LSTMPredictor(nn.Module):
    ...
```

* **LSTM (Long Short-Term Memory)** network predicts the next page based on the current page.
* Handles sequential data efficiently.

---

 **5️⃣ Train the Model**

```python
for epoch in range(200):
    ...
```

* Trains the LSTM to **learn patterns** in user navigation.
* Outputs **loss** every 50 epochs to monitor training progress.

---

 **6️⃣ Save Model and Encoder**

```python
torch.save(model.state_dict(), "zln_model.pt")
joblib.dump(encoder, "encoder.pkl")
```

* Saves the **trained model** (`.pt`) and **label encoder** (`.pkl`) so they can be loaded later without retraining.

---

 **7️⃣ Evaluate Accuracy**

```python
with torch.no_grad():
    preds = torch.argmax(out, axis=1)
    accuracy = (preds == y).sum().item() / len(y)
print(f"Prediction Accuracy: {accuracy*100:.2f}%")
```

* Checks how often the model correctly predicts the next page in your training data.

---

 **8️⃣ Test Predictions**

```python
test_pages = ["home", "products", "cart", "about", "services"]
...
```

* Prints **predicted next page** for sample pages to validate the model.

---

 **9️⃣ Download Files from Colab**

```python
from google.colab import files
files.download("zln_model.pt")
files.download("encoder.pkl")
```

* Downloads the trained model and encoder to your computer so you can use them in your **local Flask web app** or **upload to GitHub**.

---

✅ **Purpose:**

* This code **prepares and trains an AI model** for predicting the next page a user will visit.
* Saves and downloads the model for integration into your **Zero-Latency Web project**, making it **ready for deployment or sharing**.


