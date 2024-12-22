
import streamlit as st
import pandas as pd
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report


st.title("COVID-19 Data Analysis and Prediction with Feature Engineering")
st.write("""This app allows you to input parameters for COVID-19 analysis using feature-engineered columns.
Use the sidebar to input values for temporal, geographical, and case-related features.
""")

df = pd.read_csv('WHO COVID-19 cases.csv')
df_before_wrangling = df.copy()

# ------------------------------------------------------Data Wrangling------------------------------------------------------------

# Droping rows which have missing values
df = df.dropna(subset=['Country_code', 'WHO_region'])

# Fill Missing Values
df['New_cases'] = df.New_cases.fillna(value=df['New_cases'].mean())
df['New_deaths'] = df.New_deaths.fillna(value=df['New_deaths'].mean())

# Removing Outliers
from joblib import Parallel, delayed

def remove_outliers(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column[(column >= lower_bound) & (column <= upper_bound)]

columns_to_process = ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']

# Number of iterations to process
iterations = 16  # Change this to your desired number of iterations

# Perform outlier removal iteratively
for i in range(iterations):
    cleaned_columns = Parallel(n_jobs=-1)(
        delayed(remove_outliers)(df[col]) for col in columns_to_process
    )
    df[columns_to_process] = pd.concat(cleaned_columns, axis=1)

# Feature Engineering
df['Date_reported'] = pd.to_datetime(df['Date_reported'], errors='coerce')
df['Year'] = df['Date_reported'].dt.year
df['Month'] = df['Date_reported'].dt.month
df['Day'] = df['Date_reported'].dt.day
df['Weekday'] = df['Date_reported'].dt.weekday  # 0 = Monday, 6 = Sunday
df['Is_weekend'] = df['Weekday'].apply(lambda x: 1 if x >= 5 else 0)


# df = df.dropna(subset=['New_cases', 'New_deaths', 'Cumulative_cases', 'Cumulative_deaths'])

# Again Fill Missing Values
df['New_cases'] = df['New_cases'].fillna(df['New_cases'].median())
df['New_deaths'] = df['New_deaths'].fillna(df['New_deaths'].median())

df['Cumulative_cases'] = df['Cumulative_cases'].interpolate(method='linear')
df['Cumulative_deaths'] = df['Cumulative_deaths'].interpolate(method='linear')

df = df[df['New_cases']>=0]
df = df[df['New_deaths']>=0]
df = df[df['Cumulative_cases']>0]
df = df[df['Cumulative_deaths']>0]




# ------------------------------------------------------Inputs-------------------------------------------------------------------

st.sidebar.header("Choose Visualization")
visualization = st.sidebar.selectbox(
    "Choose Visualization",
    ["Distribution Plot", "Correlation Heatmap", "Boxplots", "Geographical Plot"]
)

# Sidebar Inputs
st.sidebar.header("**Input Features for Prediction**")
year = st.sidebar.number_input("Year", min_value=df["Year"].min(), max_value=df["Year"].max(), value=df["Year"].max())
month = st.sidebar.selectbox("Month", list(range(1, 13)))
country = st.sidebar.text_input("Country", "United States")
continent = st.sidebar.selectbox("Continent", ["Asia", "Europe", "Africa", "North America", "South America", "Oceania"])
task = st.sidebar.radio("Task", ["Regression", "Classification"])

if task == "Regression":
    output_to_predict = st.sidebar.selectbox(
        "Choose what to predict:",
        ["New_deaths using New_cases", "New_cases using New_deaths"]
    )
    if output_to_predict == "New_deaths using New_cases":
        new_cases_input = st.sidebar.number_input(
            "New_cases",
            min_value=0,
            max_value=int(df["New_cases"].max()),
            value=100,
            step=10
        )
        input_feature = "New_cases"
        target_feature = "New_deaths"
        input_value = new_cases_input
    else:
        new_deaths_input = st.sidebar.number_input(
            "New_deaths",
            min_value=0,
            max_value=int(df["New_deaths"].max()),
            value=10,
            step=1
        )
        input_feature = "New_deaths"
        target_feature = "New_cases"
        input_value = new_deaths_input

else:
    output_to_predict = st.sidebar.selectbox(
        "Choose what to classify:",
        ["Severity_New_deaths using New_cases", "Severity_New_cases using New_deaths"]
    )
    severity_threshold = st.sidebar.slider("Set Severity Threshold", min_value=int(df["New_deaths"].min()), max_value=int(df["New_deaths"].max()), value=100, step=50)

    df['Severity_New_cases'] = (df['New_cases'] > severity_threshold).astype(int)
    df['Severity_New_deaths'] = (df['New_deaths'] > severity_threshold).astype(int)

    if output_to_predict == "Severity_New_deaths using New_cases":
        input_feature = "New_cases"
        target_feature = "Severity_New_deaths"
    else:
        input_feature = "New_deaths"
        target_feature = "Severity_New_cases"
    

model_type = st.sidebar.selectbox(
    "Select Model Type",
    [
        "Linear Regression/Logistic Regression", "Decision Tree", "Random Forest",
        "Gradient Boosting", "SVR/SVC", "K Neighbors"
    ]
)

# Display the user inputs
st.subheader("User Inputs")
if output_to_predict == "New_deaths":
    user_inputs = {
        "Year": year,
        "Month": month,
        "Country": country,
        "Continent": continent,
        "Output to Predict": output_to_predict,
        "Task" : task,
        "Model Type" : model_type,
    }
    user_df = pd.DataFrame([user_inputs])
    st.write(user_df)

else:
    user_inputs = {
    "Year": year,
    "Month": month,
    "Country": country,
    "Continent": continent,
    "Output to Predict": output_to_predict,
    "Task" : task,
    "Model Type" : model_type,
    }
    user_df = pd.DataFrame([user_inputs])
    st.write(user_df)

st.write('---')

# -----------------------------------------------Pandas Profiling-EDA---------------------------------------------------------------

# Profile Report using ydata_profiling
st.markdown(
    """
    <div style="text-align: center;">
        <h2>Exploratory Data Analysis</h2>
    </div>
    """,
    unsafe_allow_html=True
)
@st.cache_data
def load_data():
    return df_before_wrangling
pr = ProfileReport(df_before_wrangling, explorative=True)
st.header('**Profile Report:**')
st_profile_report(pr)

st.write('---')
# -------------------------------------------------Visualization--------------------------------------------------------


import plotly.express as px
import plotly.graph_objects as go


st.title("Before and After Data Wrangling:")
st.write("""
    This app demonstrates the transformation of data through EDA and Data Wrangling.
    Check out the dataset before and after cleaning using the visualizations below.
""")


# Limit the values in the raw dataset based on the cleaned dataset's range
columns_to_limit = ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']

for col in columns_to_limit:
    min_val = df[col].min()
    max_val = df[col].max()
    df_before_wrangling[col] = df_before_wrangling[col].clip(lower=min_val, upper=max_val)


if visualization == "Distribution Plot":
    st.subheader("Distribution of Cases: Before and After Cleaning")
    columns = ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']
    selected_column = st.selectbox("Choose a Column", columns)
    # Plotly histograms
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df_before_wrangling[selected_column], name="Before Cleaning", marker=dict(color='red')))
    fig.add_trace(go.Histogram(x=df[selected_column], name="After Cleaning", marker=dict(color='green')))
    fig.update_layout(
        title="Distribution of New Cases: Before vs After Cleaning",
        xaxis=dict(title="New Cases", range=[df[selected_column].min(), df[selected_column].max()]),
        barmode='overlay'
    )
    st.plotly_chart(fig)
    
elif visualization == "Correlation Heatmap":
    st.subheader("Correlation Heatmap: Before and After Cleaning")
    # Plotly heatmaps
    numeric_columns = ['New_cases', 'Cumulative_cases', 'New_deaths', 'Cumulative_deaths']
    before_corr = df_before_wrangling[numeric_columns].corr()
    after_corr = df[numeric_columns].corr()
    fig_before = px.imshow(before_corr, text_auto=True, color_continuous_scale="reds", title="Before Cleaning")
    fig_after = px.imshow(after_corr, text_auto=True, color_continuous_scale="greens", title="After Cleaning")
    st.plotly_chart(fig_before)
    st.plotly_chart(fig_after)
elif visualization == "Boxplots":
    st.subheader("Boxplots: Before and After Cleaning")
    columns = ['New_cases', 'New_deaths']
    selected_column = st.selectbox("Choose a Column", columns)
    # Plotly boxplots
    fig = go.Figure()
    fig.add_trace(go.Box(y=df_before_wrangling[selected_column], name="Before Cleaning", marker_color='red'))
    fig.add_trace(go.Box(y=df[selected_column], name="After Cleaning", marker_color='green'))
    fig.update_layout(
        title="Boxplot of New Cases: Before vs After Cleaning",
        yaxis=dict(title="Cases", range=[df[selected_column].min(), df[selected_column].max()])
    )
    st.plotly_chart(fig)

elif visualization == "Geographical Plot":
    st.subheader("Geographical Distribution of Cases")
    
    # Plotly choropleths
    fig_before = px.choropleth(df_before_wrangling, locations="Country", locationmode="country names", color="New_cases",
                               title="Before Cleaning - New Cases", color_continuous_scale="Reds")
    fig_after = px.choropleth(df, locations="Country", locationmode="country names", color="New_cases",
                              title="After Cleaning - New Cases", color_continuous_scale="Greens")
    st.plotly_chart(fig_before)
    st.plotly_chart(fig_after)



st.write('---')
# --------------------------------------------------Machine Learning-------------------------------------------------------------

st.markdown(
    """
    <div style="text-align: center;">
        <h1>Machine Learning Section</h1>
    </div>
    """,
    unsafe_allow_html=True
)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Prepare Data for Model
X = df[["Year", "Month", "Country", "Continent", "WHO_region", "Cumulative_cases", "Cumulative_deaths" ,input_feature]]
y = df[target_feature]

# One-Hot Encoding for Categorical Variables
categorical_features = ["Country", "Continent"]
numerical_features = ["Year", "Month", input_feature]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Model Selection
if task == "Regression":
    if model_type == "Linear Regression/Logistic Regression":
        model = LinearRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_type == "SVR/SVC":
        model = SVR()
    elif model_type == "K Neighbors":
        model = KNeighborsRegressor()

else:  # Classification
    if model_type == "Linear Regression/Logistic Regression":
        model = LogisticRegression()
    elif model_type == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_type == "SVR/SVC":
        model = SVC(probability=True)
    elif model_type == "K Neighbors":
        model = KNeighborsClassifier()

# Build and Train the Pipeline
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Predict for User Input
if task == "Regression":
    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Country": [country],
        "Continent": [continent],
        input_feature: [input_value]
    })
else:
    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Country": [country],
        "Continent": [continent],
        input_feature: [st.sidebar.number_input(input_feature, min_value=0, max_value=int(df[input_feature].max()), value=100)]
    })

prediction = pipeline.predict(input_data)[0]

# Evaluate the Model
y_pred = pipeline.predict(X_test)
if task == "Regression":
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance Scores")
    st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R2 Score (r2): {r2:.2f}")
else:
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader("Model Performance Scores")
    st.write(f"Accuracy: {accuracy:.2%}")
    st.write(f"Precision Score: {precision:.2%}")
    st.write(f"Recall Score: {recall:.2%}")
    st.write(f"F1 Score: {f1:.2%}")

st.subheader(f"Prediction for {target_feature}")
st.write(f"Predicted {target_feature}: {prediction}")
