import re
import pandas as pd
import numpy as np
from faker import Faker


# ============================================================
# PRIVACY-PRESERVING SYNTHETIC DATA GENERATOR
# ============================================================
#
# Goal:
#   Generate useful synthetic data without releasing original
#   individual-level records.
#
# Privacy mechanisms used:
#   1. Detect direct PII columns if a dataset contains them.
#   2. Do not use direct PII columns for generation.
#   3. Apply k-anonymity to generalized quasi-identifiers.
#   4. Generalize age into 5-year ranges for privacy grouping.
#   5. Generate records from safe groups, not from original rows.
#   6. Add controlled noise to numeric attributes.
#   7. Prevent exact copies of original records.
#
# For the UCI Adult dataset there are no direct identifiers such
# as name, email, phone or address. Therefore, the main privacy
# protection is k-anonymity + generalized QIs + synthetic
# generation + exact-record protection.
# ============================================================


fake = Faker()

DEFAULT_QI = [
    "age",
    "education",
    "marital_status",
    "occupation"
]

# Direct identifiers that should never be released when present.
DIRECT_PII_PATTERNS = [
    r"\bname\b",
    r"\bfull[_ -]?name\b",
    r"\bfirst[_ -]?name\b",
    r"\blast[_ -]?name\b",
    r"\bemail\b",
    r"\be[-_ ]?mail\b",
    r"\bphone\b",
    r"\bmobile\b",
    r"\btelephone\b",
    r"\baddress\b",
    r"\bstreet\b",
    r"\bssn\b",
    r"\bsocial[_ -]?security\b",
    r"\bpassport\b",
    r"\baccount[_ -]?number\b",
    r"\bcredit[_ -]?card\b"
]

# Adult education -> education-num relationship.
EDUCATION_NUM_MAP = {
    "Preschool": 1,
    "1st-4th": 2,
    "5th-6th": 3,
    "7th-8th": 4,
    "9th": 5,
    "10th": 6,
    "11th": 7,
    "12th": 8,
    "HS-grad": 9,
    "Some-college": 10,
    "Assoc-voc": 11,
    "Assoc-acdm": 12,
    "Bachelors": 13,
    "Masters": 14,
    "Prof-school": 15,
    "Doctorate": 16
}


# ============================================================
# VALIDATION
# ============================================================

def _validate_input(df, n_samples=None):

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if df.empty:
        raise ValueError("Input dataset is empty.")

    if n_samples is None:
        n_samples = len(df)

    if not isinstance(n_samples, int):
        raise TypeError("n_samples must be an integer.")

    if n_samples <= 0:
        raise ValueError("n_samples must be greater than 0.")

    return n_samples


def _validate_qi(df, quasi_identifiers):

    missing = [
        column
        for column in quasi_identifiers
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Quasi-identifier columns not found: {missing}"
        )


# ============================================================
# DIRECT PII DETECTION
# ============================================================

def identify_direct_pii_columns(df):
    """
    Identify obvious direct-identifier columns by column name.

    The UCI Adult dataset normally returns an empty list because
    it does not contain direct identifiers such as names/emails.
    """

    pii_columns = []

    for column in df.columns:

        normalized = str(column).strip().lower()

        if any(
            re.search(pattern, normalized)
            for pattern in DIRECT_PII_PATTERNS
        ):
            pii_columns.append(column)

    return pii_columns


# ============================================================
# GENERALIZATION
# ============================================================

def _age_bin(value, width=5):
    """
    Convert an exact age into a 5-year generalized interval.

    Example:
        23 -> 20-24
        37 -> 35-39
    """

    value = int(value)

    start = (value // width) * width

    return f"{start}-{start + width - 1}"


def _age_bin_start(value):
    """Return the numerical start of an age interval."""

    return int(
        str(value).split("-")[0]
    )


# ============================================================
# K-ANONYMITY
# ============================================================

def apply_k_anonymity(
    df,
    quasi_identifiers=None,
    k=5,
    age_bin_width=5
):
    """
    Find privacy-safe groups using generalized QIs.

    Age is generalized into 5-year ranges before grouping.

    A group is safe when at least k original records share the
    generalized quasi-identifier combination.

    Returns
    -------
    safe_groups : DataFrame
        Safe group combinations and their sizes.

    group_columns : list
        Columns used for the generalized grouping.

    working_df : DataFrame
        Internal copy containing the age-bin column.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")

    if quasi_identifiers is None:
        quasi_identifiers = DEFAULT_QI.copy()

    _validate_qi(
        df,
        quasi_identifiers
    )

    if "age" not in quasi_identifiers:
        raise ValueError(
            "'age' must be included in the quasi-identifiers."
        )

    working_df = df.copy()

    working_df["_age_bin"] = (
        pd.to_numeric(
            working_df["age"],
            errors="coerce"
        )
        .map(
            lambda x: (
                _age_bin(
                    x,
                    age_bin_width
                )
                if pd.notna(x)
                else "Unknown"
            )
        )
    )

    group_columns = [
        "_age_bin"
        if column == "age"
        else column
        for column in quasi_identifiers
    ]

    group_sizes = (
        working_df
        .groupby(
            group_columns,
            dropna=False
        )
        .size()
        .reset_index(
            name="group_size"
        )
    )

    safe_groups = group_sizes[
        group_sizes["group_size"] >= k
    ].copy()

    return (
        safe_groups.reset_index(drop=True),
        group_columns,
        working_df
    )


def get_k_anonymity_report(
    df,
    quasi_identifiers=None,
    k=5,
    age_bin_width=5
):
    """
    Return a simple k-anonymity privacy report.
    """

    safe_groups, group_columns, working_df = (
        apply_k_anonymity(
            df,
            quasi_identifiers=quasi_identifiers,
            k=k,
            age_bin_width=age_bin_width
        )
    )

    all_sizes = (
        working_df
        .groupby(
            group_columns,
            dropna=False
        )
        .size()
    )

    return {
        "k": k,

        "total_generalized_groups": int(
            len(all_sizes)
        ),

        "safe_groups": int(
            len(safe_groups)
        ),

        "unsafe_groups": int(
            (all_sizes < k).sum()
        ),

        "minimum_group_size": int(
            all_sizes.min()
        ),

        "maximum_group_size": int(
            all_sizes.max()
        ),

        "safe_record_count": int(
            safe_groups["group_size"].sum()
        ),

        "quasi_identifiers": (
            quasi_identifiers
            if quasi_identifiers is not None
            else DEFAULT_QI.copy()
        )
    }


# ============================================================
# SAFE GROUP PREPARATION
# ============================================================

def _prepare_safe_groups(
    df,
    quasi_identifiers,
    k,
    age_bin_width
):

    safe_groups, group_columns, working_df = (
        apply_k_anonymity(
            df,
            quasi_identifiers=quasi_identifiers,
            k=k,
            age_bin_width=age_bin_width
        )
    )

    if safe_groups.empty:
        raise ValueError(
            f"No k-anonymous groups were found for k={k}. "
            "Try a smaller k or broader quasi-identifiers."
        )

    return (
        safe_groups,
        group_columns,
        working_df
    )


def _group_mask(
    working_df,
    group_columns,
    group_key
):

    mask = np.ones(
        len(working_df),
        dtype=bool
    )

    for column, value in zip(
        group_columns,
        group_key
    ):

        mask &= (
            working_df[column].astype(str)
            == str(value)
        ).to_numpy()

    return mask


def _select_safe_groups(
    safe_groups,
    n_samples,
    k,
    rng
):
    """
    Select enough safe groups to generate n_samples records.

    Every selected group receives at least k synthetic records.
    This makes the released synthetic data itself satisfy the
    generalized-group threshold.
    """

    max_groups = max(
        1,
        n_samples // k
    )

    max_groups = min(
        max_groups,
        len(safe_groups)
    )

    weights = np.sqrt(
        safe_groups[
            "group_size"
        ]
        .astype(float)
        .to_numpy()
    )

    weights = (
        weights / weights.sum()
    )

    selected_positions = rng.choice(
        len(safe_groups),
        size=max_groups,
        replace=False,
        p=weights
    )

    selected = safe_groups.iloc[
        selected_positions
    ].copy()

    counts = np.full(
        max_groups,
        k,
        dtype=int
    )

    remaining = (
        n_samples
        - (k * max_groups)
    )

    if remaining > 0:

        extra_weights = np.sqrt(
            selected[
                "group_size"
            ]
            .astype(float)
            .to_numpy()
        )

        extra_weights /= (
            extra_weights.sum()
        )

        extra = rng.multinomial(
            remaining,
            extra_weights
        )

        counts += extra

    selected[
        "_synthetic_count"
    ] = counts

    return selected


# ============================================================
# NUMERIC GENERATION
# ============================================================

def _generate_numeric_values(
    group_series,
    global_series,
    count,
    rng
):
    """
    Generate numeric values by sampling within a safe group and
    adding controlled noise.

    Values are clipped to the global observed range.
    """

    group_values = pd.to_numeric(
        group_series,
        errors="coerce"
    ).dropna()

    global_values = pd.to_numeric(
        global_series,
        errors="coerce"
    ).dropna()

    if group_values.empty:
        return np.zeros(count)

    if global_values.empty:
        return np.zeros(count)

    base = rng.choice(
        group_values.to_numpy(),
        size=count,
        replace=True
    )

    global_std = (
        float(global_values.std())
        if len(global_values) > 1
        else 1.0
    )

    # Small controlled perturbation prevents simple row copying.
    noise_scale = max(
        global_std * 0.05,
        0.01
    )

    values = (
        base
        + rng.normal(
            0,
            noise_scale,
            count
        )
    )

    values = np.clip(
        values,
        float(global_values.min()),
        float(global_values.max())
    )

    return values


# ============================================================
# CATEGORICAL GENERATION
# ============================================================

def _generate_categorical_values(
    group_series,
    count,
    rng,
    smoothing=0.0
):

    values = (
        group_series
        .dropna()
        .astype(str)
    )

    if values.empty:
        return np.array(
            [""] * count
        )

    distribution = (
        values
        .value_counts(
            normalize=True
        )
    )

    categories = (
        distribution.index.to_numpy()
    )

    probabilities = (
        distribution.to_numpy()
    )

    if (
        smoothing > 0
        and len(categories) > 1
    ):

        probabilities = (
            (1 - smoothing)
            * probabilities
            + smoothing / len(categories)
        )

        probabilities /= (
            probabilities.sum()
        )

    return rng.choice(
        categories,
        size=count,
        p=probabilities
    )


# ============================================================
# GENERATE ONE SAFE GROUP
# ============================================================

def _generate_from_group(
    original_df,
    working_df,
    group_columns,
    group_key,
    count,
    quasi_identifiers,
    rng,
    smoothing=0.0,
    age_bin_width=5
):

    mask = _group_mask(
        working_df,
        group_columns,
        group_key
    )

    group = working_df.loc[
        mask
    ]

    result = {}

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------
    # Do NOT simply copy the original age values.
    # Generate a new age inside the generalized age interval.
    # --------------------------------------------------------

    if "age" in original_df.columns:

        age_start = _age_bin_start(
            group_key[
                group_columns.index(
                    "_age_bin"
                )
            ]
        )

        age_min = max(
            int(
                original_df[
                    "age"
                ].min()
            ),
            age_start
        )

        age_max = min(
            int(
                original_df[
                    "age"
                ].max()
            ),
            age_start
            + age_bin_width
            - 1
        )

        result["age"] = rng.integers(
            age_min,
            age_max + 1,
            size=count
        )

    # --------------------------------------------------------
    # QUASI-IDENTIFIERS OTHER THAN AGE
    # --------------------------------------------------------

    for column in quasi_identifiers:

        if column == "age":
            continue

        group_position = (
            group_columns.index(
                column
            )
        )

        fixed_value = (
            group_key[
                group_position
            ]
        )

        result[column] = np.array(
            [fixed_value] * count,
            dtype=object
        )

    # --------------------------------------------------------
    # OTHER ATTRIBUTES
    # --------------------------------------------------------

    for column in original_df.columns:

        if column in result:
            continue

        # Direct PII is intentionally never generated.
        if column in identify_direct_pii_columns(
            original_df
        ):
            continue

        series = original_df[
            column
        ]

        # ----------------------------------------------------
        # EDUCATION-NUM
        # ----------------------------------------------------
        # education-num is derived from education in Adult.
        # ----------------------------------------------------

        if (
            column == "education_num"
            and "education" in result
        ):

            values = []

            fallback = int(
                round(
                    pd.to_numeric(
                        original_df[
                            column
                        ],
                        errors="coerce"
                    )
                    .dropna()
                    .median()
                )
            )

            for education in result[
                "education"
            ]:

                values.append(
                    EDUCATION_NUM_MAP.get(
                        str(education),
                        fallback
                    )
                )

            result[column] = np.array(
                values,
                dtype=int
            )

            continue

        # ----------------------------------------------------
        # NUMERIC ATTRIBUTE
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            values = (
                _generate_numeric_values(
                    group[column],
                    series,
                    count,
                    rng
                )
            )

            if pd.api.types.is_integer_dtype(
                series
            ):

                values = (
                    np.rint(values)
                    .astype(int)
                )

            result[column] = values

        # ----------------------------------------------------
        # CATEGORICAL ATTRIBUTE
        # ----------------------------------------------------

        else:

            result[column] = (
                _generate_categorical_values(
                    group[column],
                    count,
                    rng,
                    smoothing=smoothing
                )
            )

    # ========================================================
    # RELATIONSHIP-SEX CONSISTENCY
    # ========================================================
    #
    # IMPORTANT FIX:
    #
    # The original Adult dataset contains a few anomalous
    # records where:
    #
    #     Female + Husband
    #     Male   + Wife
    #
    # If relationship is generated independently from sex,
    # the synthetic dataset can reproduce these combinations.
    #
    # Therefore relationship is regenerated CONDITIONED on
    # the generated sex.
    #
    # Female:
    #     Husband is excluded.
    #
    # Male:
    #     Wife is excluded.
    #
    # We first use values from the same safe group.
    # If no valid value exists for that sex, we use the full
    # original dataset as a fallback.
    #
    # relationship is NOT a quasi-identifier, so this does
    # not change the k-anonymity grouping.
    # ========================================================

    if (
        "relationship" in result
        and "sex" in result
    ):

        corrected_relationships = []

        for sex in result["sex"]:

            # ------------------------------------------------
            # First preference:
            # Same safe group + same sex
            # ------------------------------------------------

            candidates = group.loc[
                group["sex"] == sex,
                "relationship"
            ].dropna()

            # ------------------------------------------------
            # Remove logically inconsistent combinations.
            # ------------------------------------------------

            if sex == "Female":

                candidates = candidates[
                    candidates != "Husband"
                ]

            elif sex == "Male":

                candidates = candidates[
                    candidates != "Wife"
                ]

            # ------------------------------------------------
            # Fallback:
            # Use the complete original dataset for this sex.
            # ------------------------------------------------

            if candidates.empty:

                candidates = (
                    original_df.loc[
                        original_df["sex"] == sex,
                        "relationship"
                    ]
                    .dropna()
                )

                if sex == "Female":

                    candidates = candidates[
                        candidates != "Husband"
                    ]

                elif sex == "Male":

                    candidates = candidates[
                        candidates != "Wife"
                    ]

            # ------------------------------------------------
            # Final defensive fallback.
            # This should normally never be required for Adult.
            # ------------------------------------------------

            if candidates.empty:

                candidates = (
                    original_df[
                        "relationship"
                    ]
                    .dropna()
                )

            corrected_relationships.append(
                rng.choice(
                    candidates.to_numpy()
                )
            )

        result[
            "relationship"
        ] = np.array(
            corrected_relationships,
            dtype=object
        )

    return pd.DataFrame(
        result
    )


# ============================================================
# EXACT RECORD PROTECTION
# ============================================================

def _row_tuple(
    row,
    columns
):

    return tuple(
        row[column]
        for column in columns
    )


def _build_original_record_set(
    df,
    columns
):

    return {
        _row_tuple(
            row,
            columns
        )
        for _, row
        in df[
            columns
        ].iterrows()
    }


def check_exact_matches(
    original_df,
    synthetic_df
):
    """
    Count exact copies of original records.
    """

    common_columns = [
        column
        for column in original_df.columns
        if column in synthetic_df.columns
    ]

    original_records = (
        _build_original_record_set(
            original_df,
            common_columns
        )
    )

    matches = 0

    for _, row in synthetic_df.iterrows():

        if _row_tuple(
            row,
            common_columns
        ) in original_records:

            matches += 1

    return {
        "exact_matches": int(
            matches
        ),

        "synthetic_records": int(
            len(synthetic_df)
        ),

        "exact_match_rate": (
            float(
                matches
                / len(synthetic_df)
            )
            if len(synthetic_df) > 0
            else 0.0
        )
    }


def _make_record_non_exact(
    original_df,
    synthetic_df,
    quasi_identifiers,
    rng
):
    """
    If an exact original row appears, modify only a non-QI
    attribute.

    QI values are left unchanged so that the k-anonymity
    grouping is not broken.
    """

    if synthetic_df.empty:
        return synthetic_df

    columns = list(
        synthetic_df.columns
    )

    original_records = (
        _build_original_record_set(
            original_df,
            columns
        )
    )

    non_qi_columns = [
        column
        for column in columns
        if column not in quasi_identifiers
    ]

    if not non_qi_columns:
        return synthetic_df

    result = synthetic_df.copy()

    for index in result.index:

        current = _row_tuple(
            result.loc[index],
            columns
        )

        if current not in original_records:
            continue

        # ----------------------------------------------------
        # Prefer numeric non-QI attributes.
        # ----------------------------------------------------

        numeric_candidates = [
            column
            for column in non_qi_columns
            if (
                column in original_df.columns
                and pd.api.types.is_numeric_dtype(
                    original_df[column]
                )
            )
        ]

        if numeric_candidates:

            column = (
                "hours_per_week"
                if "hours_per_week"
                in numeric_candidates
                else numeric_candidates[0]
            )

            minimum = int(
                pd.to_numeric(
                    original_df[column],
                    errors="coerce"
                ).min()
            )

            maximum = int(
                pd.to_numeric(
                    original_df[column],
                    errors="coerce"
                ).max()
            )

            old_value = int(
                result.loc[
                    index,
                    column
                ]
            )

            candidates = [
                value
                for value in (
                    old_value - 1,
                    old_value + 1,
                    int(
                        rng.integers(
                            minimum,
                            maximum + 1
                        )
                    )
                )
                if (
                    minimum
                    <= value
                    <= maximum
                    and value != old_value
                )
            ]

            if candidates:

                result.loc[
                    index,
                    column
                ] = candidates[0]

        else:

            column = non_qi_columns[0]

            values = (
                original_df[
                    column
                ]
                .dropna()
                .unique()
            )

            if len(values) > 1:

                alternatives = [
                    value
                    for value in values
                    if (
                        value
                        != result.loc[
                            index,
                            column
                        ]
                    )
                ]

                if alternatives:

                    result.loc[
                        index,
                        column
                    ] = rng.choice(
                        alternatives
                    )

    return result


# ============================================================
# MAIN PRIVACY-PRESERVING GENERATOR
# ============================================================

def generate_privacy_preserving(
    df,
    n_samples=None,
    quasi_identifiers=None,
    k=5,
    smoothing=0.0,
    random_state=None,
    age_bin_width=5
):
    """
    Generate privacy-preserving synthetic data.

    Privacy workflow:
        1. Detect direct PII columns.
        2. Exclude direct PII from generation.
        3. Generalize age.
        4. Keep only groups with size >= k.
        5. Generate new records from safe groups.
        6. Add controlled numeric perturbation.
        7. Prevent exact original-record copies.

    For the UCI Adult dataset, no direct PII columns are present,
    so the protection focuses on quasi-identifiers and
    individual-record exposure.
    """

    n_samples = _validate_input(
        df,
        n_samples
    )

    if quasi_identifiers is None:
        quasi_identifiers = (
            DEFAULT_QI.copy()
        )

    _validate_qi(
        df,
        quasi_identifiers
    )

    if k < 2:
        raise ValueError(
            "k must be at least 2."
        )

    if not 0 <= smoothing <= 1:
        raise ValueError(
            "smoothing must be between 0 and 1."
        )

    pii_columns = (
        identify_direct_pii_columns(
            df
        )
    )

    safe_groups, group_columns, working_df = (
        _prepare_safe_groups(
            df,
            quasi_identifiers,
            k,
            age_bin_width
        )
    )

    if n_samples < k:
        raise ValueError(
            f"n_samples must be at least k={k}."
        )

    rng = np.random.default_rng(
        random_state
    )

    if random_state is not None:

        np.random.seed(
            random_state
        )

        Faker.seed(
            random_state
        )

    selected_groups = (
        _select_safe_groups(
            safe_groups,
            n_samples,
            k,
            rng
        )
    )

    generated_parts = []

    for _, selected_group in (
        selected_groups.iterrows()
    ):

        group_key = tuple(
            selected_group[column]
            for column in group_columns
        )

        count = int(
            selected_group[
                "_synthetic_count"
            ]
        )

        generated = (
            _generate_from_group(
                original_df=df,
                working_df=working_df,
                group_columns=group_columns,
                group_key=group_key,
                count=count,
                quasi_identifiers=quasi_identifiers,
                rng=rng,
                smoothing=smoothing,
                age_bin_width=age_bin_width
            )
        )

        generated_parts.append(
            generated
        )

    if not generated_parts:
        raise ValueError(
            "Unable to generate synthetic records."
        )

    synthetic = pd.concat(
        generated_parts,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove direct identifiers from the released dataset.
    # --------------------------------------------------------

    if pii_columns:

        synthetic = synthetic.drop(
            columns=pii_columns,
            errors="ignore"
        )

    # --------------------------------------------------------
    # Restore original non-PII column order.
    # --------------------------------------------------------

    output_columns = [
        column
        for column in df.columns
        if (
            column not in pii_columns
            and column in synthetic.columns
        )
    ]

    synthetic = (
        synthetic[
            output_columns
        ]
        .iloc[:n_samples]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Prevent exact original records.
    # --------------------------------------------------------

    synthetic = (
        _make_record_non_exact(
            df.drop(
                columns=pii_columns,
                errors="ignore"
            ),
            synthetic,
            quasi_identifiers,
            rng
        )
    )

    return synthetic


# ============================================================
# APP-COMPATIBLE WRAPPERS
# ============================================================

def generate_baseline(
    df,
    n_samples=None,
    random_state=None
):
    """
    Baseline privacy-preserving generation.

    Kept with the original function signature so the existing
    app.py does not need to be changed.
    """

    return generate_privacy_preserving(
        df=df,
        n_samples=n_samples,
        quasi_identifiers=DEFAULT_QI,
        k=5,
        smoothing=0.0,
        random_state=random_state
    )


def generate_adaptive(
    df,
    n_samples=None,
    smoothing=0.2,
    random_state=None
):
    """
    Adaptive privacy-preserving generation.

    Kept with the original function signature so the existing
    app.py does not need to be changed.
    """

    return generate_privacy_preserving(
        df=df,
        n_samples=n_samples,
        quasi_identifiers=DEFAULT_QI,
        k=5,
        smoothing=smoothing,
        random_state=random_state
    )


def generate_synthetic_data(
    df,
    method="baseline",
    n_samples=None,
    smoothing=0.2,
    random_state=None
):
    """
    General wrapper for synthetic data generation.
    """

    method = method.lower().strip()

    if method == "baseline":

        return generate_baseline(
            df=df,
            n_samples=n_samples,
            random_state=random_state
        )

    if method == "adaptive":

        return generate_adaptive(
            df=df,
            n_samples=n_samples,
            smoothing=smoothing,
            random_state=random_state
        )

    raise ValueError(
        "Invalid method. Choose 'baseline' or 'adaptive'."
    )


# ============================================================
# FAKER-BASED SINGLE SYNTHETIC RECORD
# ============================================================

def generate_fake_record(
    df,
    random_state=None
):
    """
    Generate one artificial record.

    Direct PII columns are excluded from the generated record.
    """

    if not isinstance(
        df,
        pd.DataFrame
    ):
        raise TypeError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Input dataset is empty."
        )

    pii_columns = (
        identify_direct_pii_columns(
            df
        )
    )

    working_df = df.drop(
        columns=pii_columns,
        errors="ignore"
    )

    if random_state is not None:

        Faker.seed(
            random_state
        )

        np.random.seed(
            random_state
        )

    record = {}

    for column in working_df.columns:

        series = working_df[
            column
        ]

        # ----------------------------------------------------
        # Numeric
        # ----------------------------------------------------

        if pd.api.types.is_numeric_dtype(
            series
        ):

            clean = pd.to_numeric(
                series,
                errors="coerce"
            ).dropna()

            if clean.empty:

                record[column] = 0

                continue

            record[column] = (
                fake.random_int(
                    min=int(
                        clean.min()
                    ),
                    max=int(
                        clean.max()
                    )
                )
            )

        # ----------------------------------------------------
        # Categorical
        # ----------------------------------------------------

        else:

            values = (
                series
                .dropna()
            )

            if values.empty:

                record[column] = None

                continue

            record[column] = (
                fake.random_element(
                    elements=values.tolist()
                )
            )

    return pd.DataFrame(
        [record]
    )


# ============================================================
# PRIVACY REPORT
# ============================================================

def get_privacy_report(
    original_df,
    synthetic_df,
    quasi_identifiers=None,
    k=5
):
    """
    Produce a report showing the main privacy checks.
    """

    if quasi_identifiers is None:

        quasi_identifiers = (
            DEFAULT_QI.copy()
        )

    pii_columns = (
        identify_direct_pii_columns(
            original_df
        )
    )

    exact_match_report = (
        check_exact_matches(
            original_df.drop(
                columns=pii_columns,
                errors="ignore"
            ),
            synthetic_df
        )
    )

    try:

        k_report = (
            get_k_anonymity_report(
                original_df,
                quasi_identifiers=(
                    quasi_identifiers
                ),
                k=k
            )
        )

    except Exception as error:

        k_report = {
            "error": str(error)
        }

    released_pii = [
        column
        for column in pii_columns
        if column in synthetic_df.columns
    ]

    return {

        "direct_pii_columns_detected":
            pii_columns,

        "direct_pii_columns_in_output":
            released_pii,

        "pii_exposure_check":
            len(released_pii) == 0,

        "k_anonymity_k":
            k,

        "exact_match_check":
            exact_match_report,

        "k_anonymity_report":
            k_report
    }


# ============================================================
# SYNTHETIC DATA VALIDATION
# ============================================================

def validate_synthetic_data(
    original_df,
    synthetic_df
):
    """
    Perform structural and privacy validation.
    """

    if not isinstance(
        original_df,
        pd.DataFrame
    ):

        raise TypeError(
            "Original data must be a DataFrame."
        )

    if not isinstance(
        synthetic_df,
        pd.DataFrame
    ):

        raise TypeError(
            "Synthetic data must be a DataFrame."
        )

    pii_columns = (
        identify_direct_pii_columns(
            original_df
        )
    )

    expected_columns = [
        column
        for column in original_df.columns
        if column not in pii_columns
    ]

    same_columns = (
        expected_columns
        == list(
            synthetic_df.columns
        )
    )

    missing_values = int(
        synthetic_df.isnull()
        .sum()
        .sum()
    )

    exact_match_report = (
        check_exact_matches(
            original_df.drop(
                columns=pii_columns,
                errors="ignore"
            ),
            synthetic_df
        )
    )

    return {

        "same_columns":
            same_columns,

        "synthetic_rows":
            len(synthetic_df),

        "missing_values":
            missing_values,

        "direct_pii_columns_removed":
            pii_columns,

        "exact_matches":
            exact_match_report[
                "exact_matches"
            ],

        "exact_match_rate":
            exact_match_report[
                "exact_match_rate"
            ]
    }