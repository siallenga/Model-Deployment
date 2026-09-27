import pandas as pd
RAW_PATH = "/content/tourism_project/data/tourism.csv"

# Load the raw tourism dataset
df = pd.read_csv(RAW_PATH)

# Validate that the expected columns are present
expected_columns = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips", "Passport",
    "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting", "Designation",  "MonthlyIncome", "Unnamed: 0",
]
missing = [
    c for c in expected_columns
    if c not in df.columns
]
if missing:
    raise ValueError(
        f"Dataset is missing expected columns: {missing}"
    )
print("Dataset registered successfully.")
print(
    f"Rows: {df.shape[0]}, "
    f"Columns: {df.shape[1]}"
)

print(
    "Columns:",
    list(df.columns)
)
# Target column validation
target_column = "ProdTaken"

if target_column not in df.columns:
    raise ValueError(
        f"Target column '{target_column}' "
        "was not found in the dataset."
    )
print( "\nTarget column preview (ProdTaken):")
print( df[target_column].describe())
print( "\nTarget value counts:")
print( df[target_column].value_counts())
print( "\nTarget value percentages:" )
print(
     df[target_column]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)
# Check missing values
print( "\nMissing values by column:")
missing_values = df.isnull().sum()
print(
    missing_values[
        missing_values > 0
    ]
    if missing_values.sum() > 0
    else "No missing values found."
)
# Check duplicate records
duplicate_count = df.duplicated().sum()
print( f"\nDuplicate rows: {duplicate_count}")
# Basic data types
print( "\nData types:")
print( df.dtypes)
print(  "\nDataset registration and validation completed successfully.")
