# **Crime Data Analysis and Prediction Dashboard**

### **1. Project Goals**
The primary goals of this project are:
- **Data Analysis**: Process and analyze crime data to uncover trends and patterns based on time and location.
- **Synthetic Data Simulation**: Provide users with the ability to simulate synthetic crime data based on various adjustable parameters like base crime rate, nighttime factor, and socioeconomic factors.
- **Predictive Modeling**: Build a simple predictive model to forecast future crime counts based on historical or simulated data.
- **Interactive Dashboard**: Deliver a user-friendly dashboard for visualizing crime trends, performing simulations, and analyzing results interactively.

---

### **2. Significance and Novelty of the Project**

#### **Background Information**
Crime data is essential for law enforcement agencies and policy-makers to make informed decisions regarding resource allocation, crime prevention strategies, and public safety. Predicting crime trends and understanding the factors that influence criminal activity can help reduce crime rates and increase community safety.

#### **Significance**
- The project provides a seamless integration of real and synthetic data analysis.
- It empowers users with a hands-on tool to explore the effects of various factors like unemployment and weather on crime rates.
- The predictive capabilities allow decision-makers to anticipate and prepare for future trends.

#### **Novelty**
- Incorporates both real and synthetic datasets for analysis, allowing flexible exploration of hypothetical scenarios.
- Provides an intuitive interface with adjustable parameters for real-time simulation and analysis.
- Combines interactive visualizations (like crime trends and hourly patterns) with a machine learning-based predictive model.

---

### **3. Installation and Usage Instructions**

#### **Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VadimG0/CrimeTrends
   cd CrimeTrends
   ```
2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Application**:
   Start the Streamlit server:
   ```bash
   streamlit run main.py
   ```

#### **Usage**
1. Open the provided URL in your browser (typically `http://localhost:8501`).
2. Select the data mode:
   - **Real Mode**: Loads and analyzes historical crime data from a CSV file.
   - **Synthetic Mode**: Generates and analyzes simulated crime data based on user-defined parameters.
3. Adjust parameters (e.g., number of days, nighttime factor) in Synthetic Mode.
4. Visualize the results using crime trend plots and hourly distribution graphs.
5. View predictive results and download the processed data as a CSV file.

---

### **4. Code Structure**

#### **Flow-Chart of Code Structure**
![Code Structure Diagram](images/Code%20Structure%20Diagram.png)

#### **File Descriptions**
1. **`main.py`**: The core application file containing all functionalities: data loading, preprocessing, modeling, and visualization.
2. **`data/`**: Contains the `incidents_part1_part2.csv` file with real crime data.
3. **`requirements.txt`**: Lists the dependencies required to run the application.

---

### **5. List of Functionalities and Verification Results**

#### **Functionalities**
1. **Data Loading**: 
   - Load and preprocess real crime data (`dispatch_date` → `day`, `hour` → numeric, calculate crimes per hour).
   - Generate synthetic crime data with customizable parameters.
2. **Visualization**:
   - **Crime Trend Over Time**: Line plot showing daily crime counts.
   - **Hourly Crime Patterns**: Bar graph illustrating average crimes per hour.
3. **Predictive Modeling**:
   - Train a simple linear regression model to forecast future crime counts.
   - Automatically predict for 25% of the total days.
4. **Downloadable Results**:
   - Processed data can be downloaded as a CSV file.

#### **Verification Results**
- **Synthetic Data**: Tested for different base crime rates, nighttime factors, and unemployment rates. Results consistently showed accurate trend lines and hourly patterns.
- **Real Data**: Successfully processed and visualized sample data from `incidents_part1_part2.csv`.
- **Predictive Model**: Demonstrated smooth predictions for future days with low mean squared error on both synthetic and real datasets.

---

### **6. Showcasing the Achievement of Project Goals**

#### **Execution Results**
- **Sample Real Input**:
    - ![Sample Real Input](images/Real%20Input.png)

- **Sample Real Output**:
    - ![Real Output 1](images/Real%20Output%20(1).png)
    - ![Real Output 2](images/Real%20Output%20(2).png)
    - ![Real Output 3](images/Real%20Output%20(3).png)
    - ![Real Output 4](images/Real%20Output%20(4).png)

- **Sample Synthetic Input**:
    - ![Sample Synthetic Input](images/Synthetic%20Input.png)

- **Sample Synthetic Output**:
    - ![Synthetic Output 1](images/Synthetic%20Output%20(1).png)
    - ![Synthetic Output 2](images/Synthetic%20Output%20(2).png)
    - ![Synthetic Output 3](images/Synthetic%20Output%20(3).png)
    - ![Synthetic Output 4](images/Synthetic%20Output%20(4).png)

- **Visualizations**:
  - **Crime Trend Over Time**: A clear line graph showing daily crime patterns.
  - **Hourly Distribution**: A bar chart demonstrating the effect of nighttime factor.
  - **Future Predictions**: Predicted trend lines extending beyond the historical data.

#### **Discussion**
The project achieves its primary goal of providing an interactive platform to explore and analyze crime data. The combination of real and synthetic data allows users to simulate scenarios and test hypotheses effectively. The predictive model, though simple, provides actionable insights for short-term forecasting.

---

### **7. Discussion and Conclusions**

#### **Project Issues**
1. **Data Quality**: Missing or inconsistent data in the real dataset may affect accuracy. Future work could include handling missing values more robustly.
2. **Model Simplicity**: The linear regression model may not capture complex crime patterns influenced by multiple factors. A more sophisticated model (e.g., time-series analysis) could improve accuracy.
3. **Geospatial Analysis**: The current implementation does not utilize geographic features (e.g., latitude, longitude).

#### **Limitations**
- The project relies on the quality and coverage of the input data.
- Predictions are linear and do not account for seasonality or other non-linear trends.

#### **Course Learning Applied**
- **Algorithms**: Implemented linear regression for prediction.
- **Data Preprocessing**: Extracted and aggregated features from raw data.
- **Visualization**: Used Matplotlib and Streamlit for interactive visualizations.
- **Software Development**: Designed a user-friendly application with modular, reusable code.

#### **Conclusion**
This project successfully integrates data analysis, synthetic simulation, and predictive modeling into a single interactive dashboard. It offers valuable insights for exploring crime trends and serves as a strong foundation for further enhancements, such as incorporating geospatial data and advanced machine learning models.
