**COVID-19 Prediction and Analysis App**
----------------------------------------

This repository contains a **Streamlit app** developed to analyze and predict COVID-19 data using machine learning techniques. The application leverages the **WHO COVID-19 dataset** and provides insights through **Exploratory Data Analysis (EDA)**, data cleaning, and machine learning models for both **regression** and **classification** tasks.

### **Features of the App**

1.  **Data Exploration and Visualization**
    
    *   Perform **EDA** to understand the data's structure and distribution.
        
    *   Visualize key metrics and trends in the dataset.
        
2.  **Data Cleaning and Wrangling**
    
    *   Showcase the **difference between raw (uncleaned) data and cleaned data**.
        
    *   Handle missing values, invalid entries, and other inconsistencies effectively.
        
3.  **Machine Learning Models**
    
    *   Support for **6 algorithms** for regression and classification tasks.
        
        *   Algorithms include popular techniques like Linear Regression, Random Forest, Decision Tree, etc.
            
    *   Automatically preprocesses data with pipelines for numerical and categorical features.
        
4.  **Customizable Model Inputs**
    
    *   Allow users to **customize input features** via the app's **interactive sidebar**.
        
    *   Predict target outcomes (e.g., new cases, new deaths, or severity levels).
        
5.  **Model Evaluation**
    
    *   Display metrics like **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)** for regression.
        
    *   Show **accuracy**, **precision**, and other metrics for classification tasks.
  
### **How to Run the App**

1.  Clone this repository:

    ```python
    [https://github.com/MTalhaSaleem22/DataScience_on_COVID19_Dataset.git](https://github.com/MTalhaSaleem22/DataScience_on_COVID19_Dataset.git)
    ```

2. Install the required packages:
   ```python
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:
   ```python
   streamlit run app.py
   ```
### **Dataset**

*   The app uses the **WHO COVID-19 dataset**, which includes information about new cases, deaths, countries, and continents.
    
*   The data is cleaned and prepared for analysis and machine learning.
    

### **Technologies Used**

*   **Python**
    
*   **Streamlit** for the interactive web application
    
*   **Pandas**, **NumPy** for data wrangling and cleaning
    
*   **Scikit-learn** for machine learning models
    

### **Screenshots**

*   Include images of your app interface showing:
    
    *   Data visualizations
        
    *   Sidebar inputs
        
    *   Before-and-after cleaning visualizations
        
    *   Model predictions
        

### **Future Enhancements**

*   Add more advanced visualizations using libraries like **Plotly**.
    
*   Support additional machine learning models.
    
*   Implement feature importance analysis for better insights.
    

Feel free to contribute or suggest improvements to make the app even better!
