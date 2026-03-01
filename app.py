from flask import Flask,render_template,request
import pickle
import numpy as np

app=Flask(__name__)

model=pickle.load(open("walmart_sales_model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [
            float(request.form["Store"]),
            float(request.form["Holiday_Flag"]),
            float(request.form["Temperature"]),
            float(request.form["Fuel_Price"]),
            float(request.form["CPI"]),
            float(request.form["Unemployment"]),
            float(request.form["Month"])
        ]

        final_features = np.array([features])
        prediction = model.predict(final_features)

        return render_template("index.html",
                               prediction_text=f"Predicted Weekly Sales: {prediction[0]:,.2f}")

    except Exception as e:
        return str(e)
    
if __name__ == "__main__":
    app.run(debug=True)