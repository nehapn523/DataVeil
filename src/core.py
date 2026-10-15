import pandas as pd

DEFAULT_QI = [
    "age",
    "education",
    "marital_status",
    "occupation"
]


def analyze_risk(df, quasi_identifiers=None):
    if quasi_identifiers is None:
        quasi_identifiers = DEFAULT_QI

    missing = [
        col for col in quasi_identifiers
        if col not in df.columns
    ]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    result = df.copy()

    group_sizes = (
        result.groupby(
            quasi_identifiers,
            dropna=False,
            observed=True
        )[quasi_identifiers[0]]
        .transform("size")
    )

    result["group_size"] = group_sizes.astype(int)

    result["risk_level"] = result["group_size"].apply(
        lambda size:
            "High" if size == 1
            else "Medium" if size <= 5
            else "Low"
    )

    result["risk_explanation"] = result["group_size"].apply(
        lambda size:
            "Unique combination: appears once"
            if size == 1
            else f"Small group: appears {size} times"
            if size <= 5
            else f"Common combination: appears {size} times"
    )

    return result


def get_risk_summary(risk_df):
    summary = (
        risk_df["risk_level"]
        .value_counts()
        .reindex(["High", "Medium", "Low"], fill_value=0)
    )

    total = len(risk_df)

    result = pd.DataFrame({
        "Risk Category": summary.index,
        "Records": summary.values,
        "Percentage": [
            count / total * 100 if total else 0
            for count in summary.values
        ]
    })

    return result