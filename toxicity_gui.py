import tkinter as tk
from tkinter import messagebox, PhotoImage, Toplevel, Label
import joblib
import numpy as np
import os

# Define your k-mer analyzer again (must be identical to training time)
def kmer_analyzer(sequence, k=3):
    return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

# Load vectorizer and model (ensure this function exists when unpickling)
vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("toxicity_model.pkl")

# GUI Functionality
def predict_toxicity():
    sequence = entry.get("1.0", tk.END).strip()
    if not sequence:
        result_label.config(text="❗ Please enter a protein sequence.")
        return
    try:
        X_test = vectorizer.transform([sequence])
        proba = model.predict_proba(X_test)[0]
        prediction = model.predict(X_test)[0]
        result = "🧪 Toxic" if prediction == 1 else "✅ Non-Toxic"
        confidence = round(np.max(proba) * 100, 2)
        result_label.config(
            text=f"Prediction: {result}\nConfidence: {confidence}%", fg="#ffffff"
        )
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed:\n{e}")

# About Window
def show_about():
    about_win = Toplevel(root)
    about_win.title("About")
    about_win.geometry("400x250")
    about_win.configure(bg="#e0f7fa")
    tk.Label(
        about_win,
        text="Toxicity Prediction Tool",
        font=("Helvetica", 14, "bold"),
        bg="#e0f7fa",
    ).pack(pady=10)
    tk.Label(
        about_win,
        text="This tool predicts whether a given protein sequence\nis toxic or non-toxic using a trained machine learning model.",
        bg="#e0f7fa",
        font=("Helvetica", 11),
        justify="center",
    ).pack(pady=5)
    tk.Label(
        about_win, text="Developed by Ayesha Latif", font=("Helvetica", 10), bg="#e0f7fa"
    ).pack(pady=20)

# Main GUI layout
root = tk.Tk()
root.title("🧬 Toxicity Predictor Tool")
root.geometry("600x400")
root.configure(bg="#004d4d")

# Add logo (optional)
if os.path.exists("logo.png"):
    logo = PhotoImage(file="logo.png")
    logo_label = tk.Label(root, image=logo, bg="#004d4d")
    logo_label.pack(pady=5)

tk.Label(
    root,
    text="Enter Protein Sequence:",
    bg="#004d4d",
    fg="white",
    font=("Helvetica", 12),
).pack(pady=5)

entry = tk.Text(root, height=5, width=60, font=("Courier", 11))
entry.pack(padx=10, pady=5)

tk.Button(
    root,
    text="Predict Toxicity",
    command=predict_toxicity,
    bg="#00bfa5",
    fg="white",
    font=("Helvetica", 11, "bold"),
).pack(pady=10)

result_label = tk.Label(root, text="", font=("Helvetica", 13), bg="#004d4d")
result_label.pack(pady=10)

tk.Button(
    root,
    text="About",
    command=show_about,
    bg="#00796b",
    fg="white",
    font=("Helvetica", 10),
).pack(side="bottom", pady=10)

root.mainloop()
