import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.read_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\cleaned_student_depression00.xlsx")

# Mean for numerical columns
num_cols = df.columns

imputer = SimpleImputer(strategy="mean")
df[num_cols] = imputer.fit_transform(df[num_cols])

# Save fixed dataset
df.to_csv("cleaned_student_depression_final.csv", index=False)

print("✅ Missing values fixed using MEAN imputation!")
