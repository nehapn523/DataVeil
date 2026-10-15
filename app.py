import streamlit as st

import pandas as pd

import matplotlib.pyplot as plt



from src.data_loader import load_adult_data

from src.core import analyze_risk, get_risk_summary

from src.synthetic_generator import (

    generate_baseline,

    generate_adaptive

)





# --------------------------------------------------

# PAGE CONFIGURATION

# --------------------------------------------------



st.set_page_config(

    page_title="Privacy-Preserving Synthetic Data",

    page_icon="🔐",

    layout="wide"

)





# --------------------------------------------------

# CUSTOM STYLING

# --------------------------------------------------



st.markdown("""

<style>

    .main-title {

        font-size: 32px;

        font-weight: bold;

        color: #355C4B;

    }



    .section-description {

        font-size: 16px;

        color: #666666;

    }



    .risk-high {

        background-color: #FCE8E6;

        padding: 12px;

        border-radius: 8px;

    }



    .risk-medium {

        background-color: #FFF2D5;

        padding: 12px;

        border-radius: 8px;

    }



    .risk-low {

        background-color: #E4F2E8;

        padding: 12px;

        border-radius: 8px;

    }

</style>

""", unsafe_allow_html=True)





# --------------------------------------------------

# TITLE

# --------------------------------------------------



st.markdown(

    '<div class="main-title">'

    '🔐 Synthetic Data Generation for '

    'Privacy-Preserving AI'

    '</div>',

    unsafe_allow_html=True

)



st.markdown(

    '<p class="section-description">'

    'An explainable, experimental dashboard for '

    'rarity-based risk screening and synthetic '

    'data generation.'

    '</p>',

    unsafe_allow_html=True

)



st.divider()





# --------------------------------------------------

# LOAD DATASET

# --------------------------------------------------



@st.cache_data

def get_dataset():

    return load_adult_data("data/adult.data")





try:

    df = get_dataset()



except Exception as e:

    st.error(f"Unable to load dataset: {e}")

    st.info(

        "Check that adult.data is inside the data folder "

        "and that you are running Streamlit from the "

        "project root directory."

    )

    st.stop()





# --------------------------------------------------

# SIDEBAR

# --------------------------------------------------



st.sidebar.title("Navigation")



page = st.sidebar.radio(

    "Choose a section",

    [

        "Dataset Overview",

        "Privacy Risk Analysis",

        "Synthetic Data Generation",

        "Evaluation"

    ]

)





# --------------------------------------------------

# DATASET OVERVIEW

# --------------------------------------------------



if page == "Dataset Overview":



    st.header("📊 Dataset Overview")



    st.markdown("""

    The UCI Adult Income dataset contains demographic

    and employment-related information. It is commonly

    used for classification and data analysis tasks.

    """)



    col1, col2, col3 = st.columns(3)



    col1.metric(

        "Total Records",

        f"{len(df):,}"

    )



    col2.metric(

        "Total Features",

        len(df.columns)

    )



    col3.metric(

        "Missing Values",

        int(df.isnull().sum().sum())

    )



    st.subheader("Dataset Preview")



    st.dataframe(

        df.head(20),

        use_container_width=True

    )



    st.subheader("Feature Information")



    feature_info = pd.DataFrame({

        "Feature": df.columns,

        "Data Type": df.dtypes.astype(str).values,

        "Unique Values": [

            df[col].nunique()

            for col in df.columns

        ]

    })



    st.dataframe(

        feature_info,

        use_container_width=True

    )



    st.subheader("Statistical Summary")



    st.dataframe(

        df.describe(include="all").T,

        use_container_width=True

    )





# --------------------------------------------------

# PRIVACY RISK ANALYSIS

# --------------------------------------------------



elif page == "Privacy Risk Analysis":



    st.header("🔍 Explainable Privacy Risk Analysis")



    st.markdown("""

    ### What is privacy-risk screening?



    A record may be easier to distinguish when its

    combination of attributes is rare.



    This application uses a basic rarity-based approach

    to identify combinations of quasi-identifiers.



    **Quasi-identifiers** are attributes that may not

    identify someone individually but could potentially

    help distinguish a person when combined with

    external information.

    """)



    st.info("""

    **Current method:** Group-size-based rarity screening.



    - High: combination appears once.

    - Medium: combination appears 2–5 times.

    - Low: combination appears more than 5 times.



    These thresholds are experimental. The labels do

    not indicate actual re-identification probability

    or guarantee privacy.

    """)



    st.subheader("1. Select Quasi-identifiers")



    available_qi = [

        "age",

        "education",

        "marital_status",

        "occupation",

        "sex",

        "native_country",

        "race",

        "relationship",

        "workclass"

    ]



    selected_qi = st.multiselect(

        "Choose the attributes used for risk analysis",

        options=available_qi,

        default=[

            "age",

            "education",

            "marital_status",

            "occupation"

        ]

    )



    if not selected_qi:

        st.warning(

            "Select at least one quasi-identifier "

            "to perform risk analysis."

        )

        st.stop()



    risk_df = analyze_risk(df, selected_qi)

    summary = get_risk_summary(risk_df)



    total = len(risk_df)



    st.divider()

    st.subheader("2. Overall Risk Summary")



    cols = st.columns(3)



    descriptions = {

        "High": (

            "Unique combination",

            "This combination appears only once."

        ),

        "Medium": (

            "Small group",

            "This combination appears 2–5 times."

        ),

        "Low": (

            "Larger group",

            "This combination appears more than 5 times."

        )

    }



    counts = (

        risk_df["risk_level"]

        .value_counts()

        .reindex(["High", "Medium", "Low"], fill_value=0)

    )



    for col, level in zip(

        cols,

        ["High", "Medium", "Low"]

    ):

        count = int(counts[level])

        percentage = (

            count / total * 100 if total else 0

        )



        col.metric(

            f"{level} Rarity Risk",

            f"{percentage:.2f}%",

            f"{count:,} records"

        )



        col.caption(descriptions[level][0])



    st.divider()



    # Risk distribution chart

    st.subheader("3. Risk Distribution")

    # Percentage-based horizontal bars are easier to compare
    # than raw counts when the categories are very different.
    risk_levels = ["High", "Medium", "Low"]

    risk_percentages = [
        (int(counts[level]) / total * 100) if total else 0
        for level in risk_levels
    ]

    fig, ax = plt.subplots(figsize=(9, 4.8))

    bars = ax.barh(
        risk_levels,
        risk_percentages,
        height=0.55,
        color=["#C65D55", "#E7AC4A", "#5B9472"]
    )

    ax.set_xlabel("Percentage of Records")
    ax.set_ylabel("Rarity Risk Category")
    ax.set_title("Privacy Risk Distribution")

    max_percentage = max(risk_percentages) if risk_percentages else 0
    ax.set_xlim(0, max(max_percentage * 1.25, 10))

    for bar, percentage, level in zip(
        bars,
        risk_percentages,
        risk_levels
    ):
        count = int(counts[level])

        ax.text(
            percentage + max(max_percentage * 0.01, 0.3),
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.2f}%  ({count:,} records)",
            va="center",
            fontsize=10
        )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25
    )
    ax.set_axisbelow(True)

    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("""
    **How to interpret the chart:**

    - **High:** The selected combination appears only once.
    - **Medium:** The selected combination appears 2–5 times.
    - **Low:** The selected combination appears more than 5 times.

    The percentages make the three risk categories easier
    to compare even when their record counts are very different.

    Changing the selected quasi-identifiers can substantially
    change this distribution.
    """)

        # Risk table



    st.divider()

    st.subheader("4. Explain Individual Records")



    selected_level = st.selectbox(

        "Select a risk category",

        ["High", "Medium", "Low"]

    )



    filtered = risk_df[

        risk_df["risk_level"] == selected_level

    ]



    if filtered.empty:

        st.info(

            f"No records were classified as {selected_level}."

        )



    else:

        selected_index = st.selectbox(

            "Choose a record",

            filtered.index.tolist(),

            format_func=lambda i: f"Record {i + 1}"

        )



        selected_record = risk_df.loc[selected_index]

        group_size = int(selected_record["group_size"])



        st.markdown("#### Attribute values")



        attribute_table = pd.DataFrame({

            "Attribute": selected_qi,

            "Value": [

                selected_record[col]

                for col in selected_qi

            ]

        })



        st.dataframe(

            attribute_table,

            use_container_width=True,

            hide_index=True

        )



        st.markdown("#### Explanation")



        st.write(

            f"The selected combination occurs "

            f"**{group_size} time(s)** in the dataset."

        )



        if group_size == 1:

            st.error("""

            **High rarity risk**



            This combination is unique in the dataset.



            Because no other record shares the selected

            quasi-identifiers, it may be easier to

            distinguish within this dataset.

            """)



        elif group_size <= 5:

            st.warning(f"""

            **Medium rarity risk**



            This combination is shared by only

            {group_size} records.



            The small group size means the combination

            is relatively uncommon and may be more

            distinguishable than common combinations.

            """)



        else:

            st.success(f"""

            **Low rarity risk**



            This combination is shared by

            {group_size} records.



            The combination is more common within

            this dataset and is less rare under

            the current screening method.

            """)



        st.caption("""

        This explanation is based only on the selected

        attributes and the records in this dataset.

        It does not assess external data or measure

        a person's actual chance of being identified.

        """)



    # Automatic insights



    st.divider()

    st.subheader("5. Automatic Insights")



    high_count = int(counts["High"])

    medium_count = int(counts["Medium"])

    low_count = int(counts["Low"])



    high_pct = high_count / total * 100 if total else 0

    medium_pct = medium_count / total * 100 if total else 0

    low_pct = low_count / total * 100 if total else 0



    st.markdown(

        f"""

        **Analysis results**



        - **High rarity:** {high_pct:.2f}% of records

          have unique combinations.

        - **Medium rarity:** {medium_pct:.2f}% of records

          belong to groups containing 2–5 records.

        - **Low rarity:** {low_pct:.2f}% of records

          belong to groups containing more than 5 records.

        """

    )



    if high_pct > 10:

        st.warning("""

        **Recommendation**



        A notable share of records have unique

        combinations under the selected attributes.



        Consider evaluating generalization,

        suppression, or targeted synthetic generation

        to reduce the prevalence of rare combinations.

        """)

    else:

        st.info("""

        **Recommendation**



        The proportion of unique combinations is

        relatively limited under the current screening

        setup. Continue evaluating small groups and

        other privacy risks before drawing conclusions.

        """)



    with st.expander("View full risk analysis table"):

        st.dataframe(

            risk_df[

                selected_qi

                + [

                    "group_size",

                    "risk_level",

                    "risk_explanation"

                ]

            ].head(500),

            use_container_width=True

        )



    csv = risk_df.to_csv(index=False).encode("utf-8")



    st.download_button(

        "Download Risk Analysis CSV",

        data=csv,

        file_name="risk_analysis.csv",

        mime="text/csv"

    )





# --------------------------------------------------

# SYNTHETIC DATA GENERATION

# --------------------------------------------------



elif page == "Synthetic Data Generation":



    st.header("🧪 Synthetic Data Generation")



    st.markdown("""

    Synthetic data consists of artificially generated

    records intended to reproduce useful patterns

    without simply sharing the original records.



    This section compares two experimental methods.

    """)



    st.warning("""

    The methods below are prototypes. Independent

    column sampling and distribution smoothing do

    not preserve all relationships in the original

    dataset and do not provide formal privacy guarantees.

    """)



    n_samples = st.number_input(

        "Number of records to generate",

        min_value=100,

        max_value=100000,

        value=len(df),

        step=100

    )



    method = st.selectbox(

        "Select generation method",

        [

            "Baseline",

            "Smoothed Distribution"

        ]

    )



    smoothing = st.slider(

        "Categorical smoothing strength",

        min_value=0.0,

        max_value=1.0,

        value=0.2,

        step=0.05,

        help=(

            "Higher values make categorical distributions "

            "closer to uniform. This does not guarantee "

            "lower disclosure risk."

        )

    )



    if st.button(

        "Generate Synthetic Data",

        type="primary"

    ):

        with st.spinner("Generating records..."):



            if method == "Baseline":

                synthetic_df = generate_baseline(

                    df,

                    int(n_samples)

                )



            else:

                synthetic_df = generate_adaptive(

                    df,

                    int(n_samples),

                    smoothing=smoothing

                )



            st.session_state["synthetic_df"] = synthetic_df

            st.session_state["generation_method"] = method



    if "synthetic_df" in st.session_state:



        synthetic_df = st.session_state["synthetic_df"]



        st.success("Synthetic data generated.")



        col1, col2 = st.columns(2)



        col1.metric(

            "Original Records",

            f"{len(df):,}"

        )



        col2.metric(

            "Synthetic Records",

            f"{len(synthetic_df):,}"

        )



        st.subheader("Generated Data Preview")



        st.dataframe(

            synthetic_df.head(20),

            use_container_width=True

        )



        csv = synthetic_df.to_csv(

            index=False

        ).encode("utf-8")



        st.download_button(

            "Download Synthetic Dataset",

            data=csv,

            file_name="synthetic_data.csv",

            mime="text/csv"

        )



        st.subheader("Compare Numeric Feature Distributions")



        numeric_cols = df.select_dtypes(

            include="number"

        ).columns.tolist()



        if numeric_cols:

            selected_feature = st.selectbox(

                "Select a numeric feature",

                numeric_cols

            )



            fig, ax = plt.subplots(figsize=(9, 4.8))

            ax.hist(
                df[selected_feature],
                bins=30,
                alpha=0.55,
                density=True,
                label="Original"
            )

            ax.hist(
                synthetic_df[selected_feature],
                bins=30,
                alpha=0.55,
                density=True,
                label="Synthetic"
            )

            ax.set_title(
                f"Original vs Synthetic: {selected_feature}"
            )

            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Density")
            ax.legend()

            ax.grid(
                axis="y",
                linestyle="--",
                alpha=0.25
            )
            ax.set_axisbelow(True)

            fig.tight_layout()

            st.pyplot(fig, use_container_width=True)
            plt.close(fig)





# --------------------------------------------------

# EVALUATION

# --------------------------------------------------



elif page == "Evaluation":



    st.header("📈 Synthetic Data Evaluation")



    st.markdown("""

    This section provides preliminary comparisons

    between the original dataset and generated data.



    Generate synthetic data first in the

    Synthetic Data Generation section.

    """)



    if "synthetic_df" not in st.session_state:

        st.info(

            "No synthetic dataset found. "

            "Generate synthetic data first."

        )

        st.stop()



    synthetic_df = st.session_state["synthetic_df"]



    st.subheader("1. Dataset Size")



    col1, col2 = st.columns(2)



    col1.metric(

        "Original Rows",

        f"{len(df):,}"

    )



    col2.metric(

        "Synthetic Rows",

        f"{len(synthetic_df):,}"

    )



    st.subheader("2. Numeric Distribution Comparison")



    numeric_cols = df.select_dtypes(

        include="number"

    ).columns.tolist()



    if numeric_cols:

        comparison_rows = []



        for col in numeric_cols:

            original_mean = df[col].mean()

            synthetic_mean = synthetic_df[col].mean()



            original_std = df[col].std()

            synthetic_std = synthetic_df[col].std()



            comparison_rows.append({

                "Feature": col,

                "Original Mean": original_mean,

                "Synthetic Mean": synthetic_mean,

                "Original Std": original_std,

                "Synthetic Std": synthetic_std,

                "Absolute Mean Difference": abs(

                    original_mean - synthetic_mean

                )

            })



        comparison_df = pd.DataFrame(comparison_rows)



        st.dataframe(

            comparison_df.round(3),

            use_container_width=True

        )



        st.markdown("""

        **Interpretation**



        Smaller mean differences suggest that the

        synthetic data preserves the average values

        of numeric features more closely.



        Mean and standard deviation alone do not

        capture the full distribution or relationships

        between features.

        """)



    st.subheader("3. Categorical Distribution Comparison")



    categorical_cols = df.select_dtypes(

        exclude="number"

    ).columns.tolist()



    if categorical_cols:

        selected_cat = st.selectbox(

            "Select a categorical feature",

            categorical_cols

        )



        original_dist = (

            df[selected_cat]

            .value_counts(normalize=True)

        )



        synthetic_dist = (

            synthetic_df[selected_cat]

            .value_counts(normalize=True)

        )



        distribution_comparison = pd.concat(

            [

                original_dist.rename("Original"),

                synthetic_dist.rename("Synthetic")

            ],

            axis=1

        ).fillna(0)



        st.bar_chart(distribution_comparison)



        st.markdown("""

        **Interpretation**



        Similar category proportions indicate

        better preservation of the selected

        feature's marginal distribution.



        This does not prove that the synthetic

        data is private or that feature relationships

        have been preserved.

        """)



    st.subheader("4. Privacy Evaluation Reminder")



    st.warning("""

    Distribution similarity is a utility measure,

    not a privacy measure.



    A proper privacy evaluation should additionally

    test record-level similarity, membership inference

    risk, and other relevant disclosure threats.

    """)





# --------------------------------------------------

# FOOTER

# --------------------------------------------------



st.divider()



st.caption(

    "Research prototype | Synthetic Data Generation "

    "for Privacy-Preserving AI"

)
