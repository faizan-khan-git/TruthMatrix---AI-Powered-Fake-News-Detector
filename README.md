TruthMatrix: Full-Stack Fake News Detection System 🕵️‍♂️

TruthMatrix is a robust, full-stack machine learning web application designed to combat misinformation. It uses a Random Forest Classifier trained on over 44,000 news articles to instantly analyze and classify news content as "Real" or "Fake" with 95.91% accuracy.

✨ Features

High-Accuracy AI: Powered by a Random Forest model and TF-IDF vectorization.

Tech-Noir UI: A modern, dark-themed interface with glassmorphism effects and smooth animations.

Real-Time Analysis: Instant classification with confidence scores via a Flask REST API.

Responsive Design: Works seamlessly on desktop and mobile devices.

🛠️ Installation & Setup

Follow these steps to run the project locally on your machine.

Prerequisites

Python 3.9 or higher

pip (Python package installer)

1. Clone the Repository

git clone [https://github.com/your-username/TruthMatrix---AI-Powered-Fake-News-Detector.git](https://github.com/your-username/TruthMatrix---AI-Powered-Fake-News-Detector.git)
cd fake-news-detector


2. Create a Virtual Environment

Mac/Linux:

python3 -m venv venv
source venv/bin/activate


Windows:

python -m venv venv
venv\Scripts\activate


3. Install Dependencies

pip install pandas scikit-learn nltk flask


4. Download NLTK Data

Run this one-line command in your terminal to download the necessary text processing data:

python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt'); nltk.download('punkt_tab')"


5. Prepare the Dataset

Download the Fake and Real News Dataset from Kaggle: Link to Dataset

Unzip the downloaded file.

Create a folder named data in the root project directory.

Move True.csv and Fake.csv into the data/ folder.

6. Train the Model

Run the training script. This will preprocess the text, train the Random Forest model, and save the model files (model.pkl and vectorizer.pkl) to the saved_models/ directory.

Note: This process may take a few minutes depending on your computer's speed.

python train.py


7. Run the Application

Start the Flask web server:

python app.py


Open your browser and navigate to:
http://127.0.0.1:5000

🧠 Model Performance

The model was trained and tested on a balanced dataset of 44,898 articles.

Algorithm: Random Forest Classifier

Vectorization: TF-IDF (Top 5000 features)

Accuracy: 95.91%

Precision/Recall: 0.96 for both classes

📂 Project Structure

fake-news-detector/
├── data/               # Dataset folder (True.csv, Fake.csv)
├── saved_models/       # Trained model files (.pkl)
├── static/             # CSS and JavaScript files
│   ├── style.css
│   └── script.js
├── templates/          # HTML templates
│   └── index.html
├── app.py              # Flask backend server
├── train.py            # ML training script
├── utils.py            # Text preprocessing helper functions
└── README.md           # Project documentation


🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
