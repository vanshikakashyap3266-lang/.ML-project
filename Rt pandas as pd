from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, cross_val_score

app = Flask(__name__)

# 🔹 Improved & Balanced Dataset
data = {
    "email": [
        # Spam
        "Win money now",
        "Free gift offer",
        "Click here to win prize",
        "Earn money fast",
        "Congratulations you won",
        "Claim your free reward",
        "Limited time offer",
        "Get cash now",

        # Not Spam (Ham)
        "Hello how are you",
        "Let's meet tomorrow",
        "Are you coming to class",
        "Please send notes",
        "Call me when free",
        "Meeting at 10 am",
        "Project submission today",
        "Let's have lunch",
        "See you soon",
        "Good morning",
        "How was your day"
    ],
    "label": [
        "spam","spam","spam","spam","spam","spam","spam","spam",
        "ham","ham","ham","ham","ham","ham","ham","ham","ham","ham","ham"
    ]
}

df = pd.DataFrame(data)

# 🔹 Label Encoding (spam=1, ham=0)
le = LabelEncoder()
df["label"] = le.fit_transform(df["label"])

# 🔹 Text → Numeric (Better version)
cv = CountVectorizer(stop_words='english')
X = cv.fit_transform(df["email"])
y = df["label"]

# 🔹 Model
model = LogisticRegression()

# 🔹 K-Fold Cross Validation
kf = KFold(n_splits=3, shuffle=True, random_state=1)
scores = cross_val_score(model, X, y, cv=kf)
print("Model Accuracy:", scores.mean())

# 🔹 Train Model
model.fit(X, y)

# 🔹 Home Page
@app.route('/')
def home():
    return render_template("index.html")

# 🔹 Prediction
@app.route('/predict', methods=["POST"])
def predict():
    msg = request.form['message']
    data = cv.transform([msg])
    pred = model.predict(data)

    if pred[0] == 1:
        result = "Spam ❌"
    else:
        result = "Not Spam ✅"

    return render_template("index.html", prediction=result)

# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)
