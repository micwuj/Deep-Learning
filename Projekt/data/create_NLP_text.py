import pandas as pd

def row_to_text(row):
    return (
        f"{row['gender']} patient aged {row['age']}, "
        f"{'has hypertension' if row['hypertension'] else 'no hypertension'}, "
        f"{'has heart disease' if row['heart_disease'] else 'no heart disease'}, "
        f"HbA1c level is {row['HbA1c_level']} and blood glucose level is {row['blood_glucose_level']}."
    )

df = pd.read_csv("diabetes_prediction_dataset.csv")

df["text"] = df.apply(row_to_text, axis=1)

df[["text", "diabetes"]].to_csv("diabetes_text_descriptions.csv", index=False)

print("Saved: diabetes_text_descriptions.csv")
