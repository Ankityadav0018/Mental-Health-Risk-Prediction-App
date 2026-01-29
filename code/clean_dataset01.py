import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\Mental Health Dataset.xlsx")

# ---------- 1. Clean column names ----------
df.columns = df.columns.str.strip().str.lower()

# ---------- 2. Convert Yes / No / Maybe ----------
map_yes_no = {"Yes": 1, "No": 0, "Maybe": 0.5}
binary_cols = [
    "self_employed", "family_history", "treatment",
    "changes_habits", "mental_health_history",
    "growing_stress", "coping_struggles",
    "social_weakness", "mental_health_interview"
]

for col in binary_cols:
    df[col] = df[col].map(map_yes_no)

# ---------- 3. Convert Mood Swings ----------
mood_map = {"Low": 0, "Medium": 1, "High": 2}
df["mood_swings"] = df["mood_swings"].map(mood_map)

# ---------- 4. Convert Work Interest ----------
work_map = {"Yes": 1, "No": 0, "Maybe": 0.5}
df["work_interest"] = df["work_interest"].map(work_map)

# ---------- 5. Convert Days Indoors ----------
days_map = {
    "1-14 days": 7,
    "15-30 days": 22,
    "More than 2 months": 75,
    "Go out Every day": 0
}
df["days_indoors"] = df["days_indoors"].map(days_map)

# ---------- 6. Encode Gender, Country, Occupation ----------
encoder = LabelEncoder()
df["gender"] = encoder.fit_transform(df["gender"])
df["country"] = encoder.fit_transform(df["country"])
df["occupation"] = encoder.fit_transform(df["occupation"])

# ---------- 7. Create Mental Health Risk Score ----------
df["risk_score"] = (
    df["family_history"] +
    df["mental_health_history"] +
    df["growing_stress"] +
    df["coping_struggles"] +
    df["mood_swings"] +
    df["social_weakness"]
)

# ---------- 8. Final cleaned columns ----------
cleaned_df = df[
    [
        "gender",
        "country",
        "occupation",
        "self_employed",
        "family_history",
        "treatment",
        "days_indoors",
        "changes_habits",
        "mental_health_history",
        "growing_stress",
        "mood_swings",
        "coping_struggles",
        "work_interest",
        "social_weakness",
        "risk_score"
    ]
]

# ---------- 9. Save cleaned file ----------
cleaned_df.to_csv("cleaned_student_depression1.csv", index=False)

print("✅ cleaned_student_depression1.csv successfully created!")