import tkinter as tk
from tkinter import messagebox
from joblib import load

# Load the LinearSVC model and LabelEncoder
loaded_linearSVC_model = load('linearSVC_model.joblib')
enc = load('label_encoder.joblib')

def predict_emotion():
    input_text = entry.get()
    if not input_text:
        messagebox.showwarning("Warning", "Please enter text before predicting emotion.")
        return

    # Predict emotion
    predicted_label = loaded_linearSVC_model.predict([input_text])
    predicted_emotion = enc.classes_[predicted_label][0]

    # Display the result
    result_label.config(text=f'Predicted Emotion: {predicted_emotion}')

# Create the main Tkinter window
window = tk.Tk()
window.title("Text Emotion Prediction")

# Create input entry and button
entry = tk.Entry(window, width=40)
entry.grid(row=0, column=0, padx=10, pady=10)

predict_button = tk.Button(window, text="Predict Emotion", command=predict_emotion)
predict_button.grid(row=0, column=1, padx=10, pady=10)

# Create a label to display the predicted emotion
result_label = tk.Label(window, text="")
result_label.grid(row=1, column=0, columnspan=2, pady=10)

# Start the Tkinter event loop
window.mainloop()
