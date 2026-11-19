import os
from flask import Flask, render_template, request
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
template_dir = os.path.join(BASE_DIR, 'templates')
static_dir = os.path.join(BASE_DIR, 'static')

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Load trained model
model_path = os.path.join(BASE_DIR, "iris_model.pkl")
model = joblib.load(model_path)

species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        sl = float(request.form['sepal_length'])
        sw = float(request.form['sepal_width'])
        pl = float(request.form['petal_length'])
        pw = float(request.form['petal_width'])

        error = None

        if not (4.3 <= sl <= 7.9):
            error = "Sepal Length must be between 4.3 and 7.9 cm."
        elif not (2.0 <= sw <= 4.4):
            error = "Sepal Width must be between 2.0 and 4.4 cm."
        elif not (1.0 <= pl <= 6.9):
            error = "Petal Length must be between 1.0 and 6.9 cm."
        elif not (0.1 <= pw <= 2.5):
            error = "Petal Width must be between 0.1 and 2.5 cm."

        if error:
            return render_template(
                'index.html',
                error=error
            )

        prediction = model.predict([[sl, sw, pl, pw]])
        result = species[prediction[0]]

        return render_template(
            'index.html',
            prediction=result
        )

    except Exception as e:
        return render_template(
            'index.html',
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
