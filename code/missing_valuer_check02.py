import pandas as pd

df = pd.read_excel(r"C:\Users\punee\OneDrive\Desktop\mlproject\cleaned_student_depression00.xlsx")

print("\n✅ DATASET SHAPE:")
print(df.shape)

print("\n✅ FIRST 5 ROWS:")
print(df.head())

print("\n✅ DATA TYPES:")
print(df.dtypes)

print("\n❌ MISSING VALUES PER COLUMN:")
print(df.isnull().sum())
