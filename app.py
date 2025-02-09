import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans

class PhoneUsageAnalyzer:
    def __init__(self, data_path):
        self.load_and_prepare_data(data_path)
        self.setup_models()
        
    def load_and_prepare_data(self, data_path):
        """Load and prepare the dataset with feature engineering"""
        try:
            self.df = pd.read_csv(data_path)
            self.create_derived_features()
        except FileNotFoundError:
            st.error(f"Error: Could not find the file {data_path}")
            st.stop()
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.stop()
        
    def create_derived_features(self):
        """Create derived features from raw data"""
        try:
            self.df["Entertainment Ratio"] = (self.df["Streaming Time (hrs/day)"] + 
                                            self.df["Gaming Time (hrs/day)"]) / self.df["Screen Time (hrs/day)"]
            self.df["Calls per Hour"] = self.df["Calls Duration (mins/day)"] / (self.df["Screen Time (hrs/day)"] * 60)
            self.df["App Density"] = self.df["Number of Apps Installed"] / self.df["Screen Time (hrs/day)"]
            self.df["Data per App"] = self.df["Data Usage (GB/month)"] / self.df["Number of Apps Installed"]
            
            # Handle potential divide by zero or infinite values
            self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
            self.df.fillna(0, inplace=True)
            
            self.df["Recharge Category"] = pd.qcut(self.df["Monthly Recharge Cost (INR)"], 
                                                  q=3, labels=["Low", "Medium", "High"])
            self.df["Spender Category"] = pd.qcut(self.df["E-commerce Spend (INR/month)"], 
                                                 q=3, labels=["Low", "Medium", "High"])
        except Exception as e:
            st.error(f"Error creating derived features: {str(e)}")
            st.stop()
        
    def setup_models(self):
        """Set up clustering and classification models"""
        try:
            self.setup_clustering()
            self.setup_classification()
        except Exception as e:
            st.error(f"Error setting up models: {str(e)}")
            st.stop()
        
    def setup_clustering(self):
        """Initialize and fit KMeans clustering"""
        clustering_features = self.df[[
            'Screen Time (hrs/day)', 'Data Usage (GB/month)', 
            'Social Media Time (hrs/day)', 'Streaming Time (hrs/day)',
            'Gaming Time (hrs/day)', 'Calls Duration (mins/day)', 
            'Monthly Recharge Cost (INR)'
        ]]
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(clustering_features)
        
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.df["Cluster"] = self.kmeans.fit_predict(self.scaled_features)
        
    def setup_classification(self):
        """Initialize and fit Random Forest classifier"""
        self.df["High Spender"] = (self.df["E-commerce Spend (INR/month)"] > 
                                  self.df["E-commerce Spend (INR/month)"].median()).astype(int)
        
        X_class = self.df[[
            'Screen Time (hrs/day)', 'Data Usage (GB/month)', 
            'Social Media Time (hrs/day)', 'Gaming Time (hrs/day)', 
            'Streaming Time (hrs/day)', 'Monthly Recharge Cost (INR)',
            'Calls Duration (mins/day)'
        ]]
        y_class = self.df['High Spender']
        
        X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, 
                                                           test_size=0.2, random_state=42)
        
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(X_train, y_train)
        self.model_accuracy = accuracy_score(y_test, self.clf.predict(X_test))
        self.classification_report = classification_report(y_test, self.clf.predict(X_test))
def main():
    st.set_page_config(page_title="Phone Usage Analysis", layout="wide")
    
    # Initialize the analyzer
    analyzer = PhoneUsageAnalyzer("cleanedPhoneUsage.csv")
    
    # Page Header
    st.title("📱 Phone Usage Analysis Dashboard")
    st.write("Analyze and predict user behavior based on phone usage patterns")
    
    # Sidebar for navigation
    page = st.sidebar.selectbox("Select Page", 
                               ["Overview", "User Segmentation", "Spending Prediction", 
                                "User Behavior Analysis", "AI Insights"])
    
    if page == "Overview":
        show_overview(analyzer)
    elif page == "User Segmentation":
        show_segmentation(analyzer)
    elif page == "Spending Prediction":
        show_prediction(analyzer)
    elif page == "User Behavior Analysis":
        show_behavior_analysis(analyzer)
    else:
        show_ai_insights(analyzer)

def show_overview(analyzer):
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Sample Data")
        st.dataframe(analyzer.df.head(10))
    
    with col2:
        st.write("### Key Statistics")
        st.write(f"Total Users: {len(analyzer.df)}")
        st.write(f"Average Screen Time: {analyzer.df['Screen Time (hrs/day)'].mean():.2f} hrs/day")
        st.write(f"Average Data Usage: {analyzer.df['Data Usage (GB/month)'].mean():.2f} GB/month")

def show_segmentation(analyzer):
    st.header("🎯 User Segmentation Analysis")
    
    # Clustering visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=analyzer.df, 
                    x="Screen Time (hrs/day)", 
                    y="Data Usage (GB/month)", 
                    hue="Cluster", 
                    palette="coolwarm")
    plt.title("User Segments Based on Usage Patterns")
    st.pyplot(fig)
    
    # Cluster characteristics
    st.write("### Cluster Characteristics")
    for cluster in range(3):
        cluster_data = analyzer.df[analyzer.df["Cluster"] == cluster]
        st.write(f"**Cluster {cluster}**")
        st.write(f"- Average Screen Time: {cluster_data['Screen Time (hrs/day)'].mean():.2f} hrs/day")
        st.write(f"- Average Data Usage: {cluster_data['Data Usage (GB/month)'].mean():.2f} GB/month")

def show_prediction(analyzer):
    st.header("💰 E-Commerce Spending Prediction")
    
    st.write("### Model Performance")
    st.write(f"Prediction Accuracy: {analyzer.model_accuracy:.2f}")
    st.write("Detailed Report:")
    st.text(analyzer.classification_report)
    
    st.write("### Make a Prediction")
    col1, col2 = st.columns(2)
    
    with col1:
        screen_time = st.number_input('Screen Time (hrs/day)', min_value=0.0, step=0.1)
        data_usage = st.number_input('Data Usage (GB/month)', min_value=0.0, step=0.1)
        social_media_time = st.number_input('Social Media Time (hrs/day)', min_value=0.0, step=0.1)
        gaming_time = st.number_input('Gaming Time (hrs/day)', min_value=0.0, step=0.1)
    
    with col2:
        streaming_time = st.number_input('Streaming Time (hrs/day)', min_value=0.0, step=0.1)
        calls_duration = st.number_input('Calls Duration (mins/day)', min_value=0, step=1)
        monthly_recharge_cost = st.number_input('Monthly Recharge Cost (INR)', min_value=0.0, step=10.0)
    
    if st.button("Predict"):
        user_input = np.array([[
            screen_time, data_usage, social_media_time, gaming_time,
            streaming_time, monthly_recharge_cost, calls_duration
        ]])
        
        scaled_input = analyzer.scaler.transform(user_input)
        cluster = analyzer.kmeans.predict(scaled_input)[0]
        spend_pred = analyzer.clf.predict(scaled_input)[0]
        
        st.write("### Results")
        st.write(f"User Segment: Cluster {cluster}")
        st.write(f"Predicted to be a {'High' if spend_pred else 'Low'} E-Commerce Spender")

def show_behavior_analysis(analyzer):
    st.header("🔍 User Behavior Analysis")
    
    # Time Distribution Analysis
    st.subheader("Time Distribution")
    activities = ['Screen Time', 'Social Media', 'Gaming', 'Streaming']
    avg_times = [
        analyzer.df['Screen Time (hrs/day)'].mean(),
        analyzer.df['Social Media Time (hrs/day)'].mean(),
        analyzer.df['Gaming Time (hrs/day)'].mean(),
        analyzer.df['Streaming Time (hrs/day)'].mean()
    ]
    
    fig = px.pie(values=avg_times, names=activities, hole=0.3)
    st.plotly_chart(fig)
    
    # Usage Patterns by Age
    st.subheader("Usage Patterns by Age")
    age_groups = sorted(analyzer.df['Age'].unique())
    selected_age = st.selectbox("Select Age Group", age_groups)
    
    age_data = analyzer.df[analyzer.df['Age'] == selected_age]
    
    fig = go.Figure(data=go.Scatterpolar(
        r=[
            age_data['Screen Time (hrs/day)'].mean(),
            age_data['Social Media Time (hrs/day)'].mean(),
            age_data['Gaming Time (hrs/day)'].mean(),
            age_data['Streaming Time (hrs/day)'].mean(),
            age_data['Data Usage (GB/month)'].mean()/30
        ],
        theta=['Screen Time', 'Social Media', 'Gaming', 'Streaming', 'Data Usage'],
        fill='toself'
    ))
    st.plotly_chart(fig)

def show_ai_insights(analyzer):
    st.header("🤖 AI-Powered Insights")
    
    # Persona Generation
    st.subheader("User Persona Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.selectbox("Age Group", sorted(analyzer.df['Age'].unique()))
        screen_time = st.slider("Screen Time (hrs/day)", 0, 24, 4)
        data_usage = st.slider("Data Usage (GB/month)", 0, 100, 20)
    
    with col2:
        similar_users = analyzer.df[
            (analyzer.df['Age'] == age) &
            (analyzer.df['Screen Time (hrs/day)'].between(screen_time-2, screen_time+2)) &
            (analyzer.df['Data Usage (GB/month)'].between(data_usage-10, data_usage+10))
        ]
        
        if len(similar_users) > 0:
            avg_social = similar_users['Social Media Time (hrs/day)'].mean()
            avg_gaming = similar_users['Gaming Time (hrs/day)'].mean()
            avg_streaming = similar_users['Streaming Time (hrs/day)'].mean()
            avg_spend = similar_users['E-commerce Spend (INR/month)'].mean()
            
            st.write("### Predicted Behavior")
            fig = go.Figure(data=[
                go.Bar(
                    x=['Social Media', 'Gaming', 'Streaming', 'E-commerce'],
                    y=[avg_social, avg_gaming, avg_streaming, avg_spend],
                    marker_color=['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
                )
            ])
            st.plotly_chart(fig)
            
            # User Type Determination
            activities = {
                'Social Media': avg_social,
                'Gaming': avg_gaming,
                'Streaming': avg_streaming
            }
            main_activity = max(activities.items(), key=lambda x: x[1])[0]
            
            st.write(f"Primary Activity: {main_activity}")
            st.write(f"Predicted Monthly Spend: ₹{avg_spend:.2f}")
        else:
            st.write("Not enough similar users found. Please adjust your inputs.")

if __name__ == "__main__":
    main()