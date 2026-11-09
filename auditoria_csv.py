"""Read-only checks of the supplied insurance claims CSV."""

import pandas as pd


df = pd.read_csv("insurance_claims_limpio.csv")
cat = df.select_dtypes(exclude="number")
num = df.select_dtypes(include="number")

print("SHAPE", df.shape)
print("TARGET", df.fraud_reported.value_counts().sort_index().to_dict())
print("DUPLICATE_ROWS", int(df.duplicated().sum()))
print("DUPLICATE_TARGET_FEATURE_ROWS", int(df.drop(columns="fraud_reported").duplicated().sum()))
print("CONSTANT_COLUMNS", df.columns[df.nunique(dropna=False) <= 1].tolist())
print("ACTUAL_NULLS", df.isna().sum()[lambda x: x > 0].to_dict())
print("MISSING_LITERAL", (cat == "MISSING").sum()[lambda x: x > 0].to_dict())
print("OTHER_MISSING_MARKERS", {marker: int((cat == marker).sum().sum()) for marker in ["?", "NA", "N/A", "", "Unknown", "UNKNOWN"]})
print("TEXT_WHITESPACE", {c: int((cat[c].dropna() != cat[c].dropna().str.strip()).sum()) for c in cat if (cat[c].dropna() != cat[c].dropna().str.strip()).any()})
print("NUMERIC_RANGES")
print(num.agg(["min", "median", "max"]).T.to_string())
print("NEGATIVE_COUNTS", (num < 0).sum()[lambda x: x > 0].to_dict())
print("ZERO_COUNTS", (num == 0).sum()[lambda x: x > 0].to_dict())
print("INVALID_RANGES", {
    "age_outside_18_100": int((~df.age.between(18, 100)).sum()),
    "incident_hour_outside_0_23": int((~df.incident_hour_of_the_day.between(0, 23)).sum()),
    "incident_month_outside_1_12": int((~df.incident_month.between(1, 12)).sum()),
    "incident_day_of_week_outside_0_6": int((~df.incident_day_of_week.between(0, 6)).sum()),
    "auto_year_after_2026": int((df.auto_year > 2026).sum()),
    "claim_component_negative": int((df[["injury_claim", "property_claim", "vehicle_claim"]] < 0).any(axis=1).sum()),
    "claims_all_zero": int((df[["injury_claim", "property_claim", "vehicle_claim"]] == 0).all(axis=1).sum()),
    "dias_desde_emision_negative": int((df.dias_desde_emision < 0).sum()),
})
print("MONTHS", df.incident_month.value_counts().sort_index().to_dict())
print("HOBBY_FRAUD")
print(df.groupby("insured_hobbies").fraud_reported.agg(["count", "sum", "mean"]).sort_values("mean", ascending=False).to_string())
print("SEVERITY_FRAUD")
print(df.groupby("incident_severity").fraud_reported.agg(["count", "sum", "mean"]).sort_values("mean", ascending=False).to_string())
print("INCIDENT_COLLISION")
print(pd.crosstab(df.incident_type, df.collision_type, dropna=False).to_string())
print("AUTHORITIES_NULL_BY_INCIDENT")
print(pd.crosstab(df.incident_type, df.authorities_contacted.isna()).to_string())
print("MISSING_BY_TARGET")
for col in ["authorities_contacted", "collision_type", "property_damage", "police_report_available"]:
    marker = df[col].isna() if col == "authorities_contacted" else df[col].eq("MISSING")
    print(col, pd.crosstab(marker, df.fraud_reported).to_dict())
print("AGE_MONTHS_VIOLATIONS", int((df.months_as_customer > (df.age - 18) * 12).sum()))
print("DAYS_MONTHS_CORRELATION", df[["dias_desde_emision", "months_as_customer"]].corr().iloc[0, 1])
