 Predictive Zero-Latency Web Application 🚀

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red)
![License](https://img.shields.io/badge/License-MIT-green)

 **Project Description**

This is an AI-powered web project that predicts the next webpage a user is likely to visit and prefetches it, simulating **zero-latency browsing**.  
The project uses an **LSTM neural network** to learn navigation patterns and provides a Flask-based demo with metrics like **hit ratio, prediction accuracy, and average latency**.

---

 **Features**

- Predicts the next page using **LSTM**.
- Simulates **prefetching** for faster navigation.
- Tracks **metrics**: prediction accuracy, prefetch hits, and average latency.
- **Flask web interface** with templates for multiple pages.
- Easy to **retrain** using Google Colab notebook.
- Fully **open-source** and educational.

---

 **Project Structure**

```

Predictive-Zero-Latency-Web/
│
├── app.py                    Flask web app
├── zln\_model.pt              Trained LSTM model
├── encoder.pkl               Label encoder
├── requirements.txt          Python dependencies
├── README.md                 Project description
├── LICENSE                   MIT License
├── train\_model.ipynb         Colab notebook to train model
│
├── templates/                HTML pages
│   ├── home.html
│   ├── products.html
│   ├── cart.html
│   ├── checkout.html
│   ├── about.html
│   └── contact.html
│
└── static/                   CSS and assets
└── style.css

````

---

 **Installation**

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Predictive-Zero-Latency-Web.git
cd Predictive-Zero-Latency-Web
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

 **Usage**

1. Run the Flask app:

```bash
python app.py
```

2. Open a web browser and go to:

```
http://127.0.0.1:5000/
```

3. Navigate through pages and watch the **predicted next page** and **metrics**.

---

 **Training the Model**

* Open `train_model.ipynb` in Google Colab.
* Update navigation logs if needed.
* Train the LSTM model.
* Download the trained `zln_model.pt` and `encoder.pkl`.
* Replace existing files in the repo to use the new model.

---

 **Metrics**

* **Prediction Accuracy:** How often the model predicts the correct next page.
* **Prefetch Hit Ratio:** Percentage of times prefetch was successful.
* **Average Latency Saved:** Difference in load time with and without prefetch.

---

 **Screenshots / Demo**

*(Add screenshots or GIFs here)*

---

 **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

 **Contributors**

* Your Name – [GitHub Profile](https://github.com/yourusername)

---

 **Acknowledgements**

* [PyTorch](https://pytorch.org/) – Deep learning framework
* [Flask](https://flask.palletsprojects.com/) – Web framework
* [scikit-learn](https://scikit-learn.org/) – Preprocessing and utilities

```

---

