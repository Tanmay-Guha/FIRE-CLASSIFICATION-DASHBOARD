# -*- coding: utf-8 -*-
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from streamlit_folium import st_folium
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
from PIL import Image
import io
import scipy.stats as stats
import statsmodels.api as sm
import base64

# Custom CSS for styling
def set_custom_style():
    st.markdown("""
    <style>
        /* Main content styling */
        .stApp {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-image: linear-gradient(to bottom, #ff7e5f, #feb47b);
        }

        /* Button styling */
        .stButton>button {
            background-color: #ff7e5f;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s;
        }

        .stButton>button:hover {
            background-color: #feb47b;
            color: white;
            transform: scale(1.05);
        }

        /* Slider styling */
        .stSlider {
            color: #ff7e5f;
        }

        /* Selectbox styling */
        .stSelectbox>div>div>select {
            border: 2px solid #ff7e5f;
            border-radius: 5px;
        }

        /* Tab styling */
        .stTabs>div>div>div {
            background-color: #f8f9fa;
            border-radius: 5px;
        }

        .stTabs>div>div>div>button {
            color: #495057;
            font-weight: bold;
        }

        .stTabs>div>div>div>button[aria-selected="true"] {
            color: #ff7e5f;
            border-bottom: 2px solid #ff7e5f;
        }

        /* Dataframe styling */
        .dataframe {
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        /* Title styling */
        h1 {
            color: #d9534f;
            border-bottom: 2px solid #f0f0f0;
            padding-bottom: 10px;
        }

        h2 {
            color: #5bc0de;
        }

        h3 {
            color: #5cb85c;
        }
        
        /* Footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #0e1117;
            color: white;
            text-align: center;
            padding: 10px;
            z-index: 1000;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

# Background image function
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set page config
st.set_page_config(
    page_title="Fire Classification Dashboard",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom styles
set_custom_style()

# Set background image for home and about pages
def set_page_background(page_name):
    if page_name in ["Home", "About"]:
        set_background("img.jpg")
        
        # Add overlay for better text readability
        st.markdown("""
        <style>
        .main .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            border-radius: 10px;
            padding: 2rem;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)

# Footer function
def footer():
    st.markdown("""
    <div class="footer">
        <p>🔥 Fire Classification Dashboard | Developed With ❤️ by TANMAY GUHA | © 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df1 = pd.read_csv('modis_2021_India.csv')
    df2 = pd.read_csv('modis_2022_India.csv')
    df3 = pd.read_csv('modis_2023_India.csv')
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Data preprocessing
    df['acq_date'] = pd.to_datetime(df['acq_date'])
    df['year'] = df['acq_date'].dt.year
    df['month'] = df['acq_date'].dt.month
    df['day_of_week'] = df['acq_date'].dt.dayofweek
    df['day_of_year'] = df['acq_date'].dt.dayofyear
    df['hour'] = df['acq_time'].astype(str).str[:2].astype(int)
    
    # Remove outliers
    numerical_cols = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)].copy()
    
    # Encoding
    categorical_cols_to_encode = ['daynight', 'satellite', 'instrument']
    df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=False)
    
    # Scaling
    scaler = StandardScaler()
    numerical_cols_to_scale = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
    df_encoded[numerical_cols_to_scale] = scaler.fit_transform(df_encoded[numerical_cols_to_scale])
    
    return df, df_encoded, scaler

df, df_encoded, scaler = load_data()

# Load or train model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_fire_detection_model.pkl')
        return model
    except:
        # Train model if not saved
        features = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
        target = 'type'
        X = df_encoded[features]
        y = df_encoded[target]
        
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled
        )
        
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        
        # Save the model for future use
        joblib.dump(model, 'best_fire_detection_model.pkl')
        
        return model

model = load_model()

# Sidebar
st.sidebar.title("🔥 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Evaluation", "About"])

# Set page background
set_page_background(page)

# In your app.py, update the Home page section with this code:

if page == "Home":
    # Set a white background container for better visibility
    st.markdown("""
    <div style="background-color:white;padding:30px;border-radius:10px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h1 style="color:#d9534f;text-align:center;">🔥 Fire Classification Dashboard</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:white;padding:25px;border-radius:10px;margin-top:20px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color:#5bc0de;">Welcome to the Fire Classification Dashboard!</h2>
        <p style="font-size:16px;line-height:1.6;">This application helps analyze and classify fire events in India from 2021 to 2023 using MODIS satellite data.</p>
    </div>
    """, unsafe_allow_html=True)

    # Create feature cards with proper spacing
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color:white;padding:20px;border-radius:10px;margin:10px 0;box-shadow: 0 4px 8px rgba(0,0,0,0.1);height:150px;">
            <h3 style="color:#1890ff;">📊 Data Exploration</h3>
            <p style="font-size:14px;">View raw data and statistics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:white;padding:20px;border-radius:10px;margin:10px 0;box-shadow: 0 4px 8px rgba(0,0,0,0.1);height:150px;">
            <h3 style="color:#faad14;">📈 Visualizations</h3>
            <p style="font-size:14px;">Interactive plots and maps</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color:white;padding:20px;border-radius:10px;margin:10px 0;box-shadow: 0 4px 8px rgba(0,0,0,0.1);height:150px;">
            <h3 style="color:#52c41a;">🔮 Model Prediction</h3>
            <p style="font-size:14px;">Classify fire types using ML</p>
        </div>
        """, unsafe_allow_html=True)

    # Add the image with updated parameter
    st.image("img.jpg", caption="Satellite Imagery of Fire Events in India", use_container_width=True)
    
    # Add quick stats with better spacing
    st.markdown("""
    <div style="background-color:white;padding:20px;border-radius:10px;margin-top:20px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h3 style="color:#2f4f4f;border-bottom:1px solid #eee;padding-bottom:10px;">Quick Stats</h3>
        <div style="display:flex;justify-content:space-between;text-align:center;">
            <div style="width:30%;">
                <h2 style="color:#d9534f;">{:,}</h2>
                <p>Total Fire Events</p>
            </div>
            <div style="width:30%;">
                <h2 style="color:#5bc0de;">{}</h2>
                <p>Years Covered</p>
            </div>
            <div style="width:30%;">
                <h2 style="color:#5cb85c;">{}</h2>
                <p>Fire Types</p>
            </div>
        </div>
    </div>
    """.format(len(df), len(df['year'].unique()), len(df['type'].unique())), unsafe_allow_html=True)
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    tab1, tab2, tab3 = st.tabs(["📋 Raw Data", "📈 Statistics", "ℹ️ Data Info"])
    
    with tab1:
        st.subheader("Raw Data Preview")
        rows_to_show = st.slider("Number of rows to display", 5, 100, 10, key="rows_slider")
        st.dataframe(df.head(rows_to_show))
        
    with tab2:
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe().T)
        
    with tab3:
        st.subheader("Data Information")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Values"))
        
        st.subheader("Duplicate Rows")
        st.markdown(f"""
        <div style="background-color:#f8d7da;color:#721c24;padding:10px;border-radius:5px;">
            Number of duplicate rows: <strong>{df.duplicated().sum()}</strong>
        </div>
        """, unsafe_allow_html=True)

elif page == "Visualizations":
    st.title("📈 Data Visualizations")
    
    viz_option = st.selectbox("Select Visualization", [
        "Fire Type Distribution",
        "Confidence Distribution",
        "Confidence by Fire Type",
        "Fire Locations Map",
        "Day/Night Observations",
        "Satellite Distribution",
        "Version Distribution",
        "Numerical Features Correlation",
        "Temporal Analysis"
    ], key="viz_select")
    
    st.markdown(f"""
    <div style="background-color:#e9ecef;padding:10px;border-radius:5px;margin-bottom:20px;">
        <h3 style="color:#495057;">{viz_option}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if viz_option == "Fire Type Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='type', data=df, palette='viridis', ax=ax)
        ax.set_title('Distribution of Fire Types', fontsize=14, pad=20)
        ax.set_xlabel('Fire Type', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_option == "Confidence Distribution":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df['confidence'], bins=20, kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Confidence', fontsize=14, pad=20)
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        st.pyplot(fig)
        
    elif viz_option == "Confidence by Fire Type":
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='type', y='confidence', data=df, palette='Set2', ax=ax)
        ax.set_title('Confidence by Fire Type', fontsize=14, pad=20)
        ax.set_xlabel('Fire Type', fontsize=12)
        ax.set_ylabel('Confidence', fontsize=12)
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    elif viz_option == "Fire Locations Map":
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:10px;border-radius:5px;margin-bottom:20px;">
            <p>Interactive map showing fire locations (sampled for performance)</p>
        </div>
        """, unsafe_allow_html=True)
        
        sample_size = st.slider("Sample size", 100, 5000, 1000, step=100, key="map_slider")
        sample_df = df.sample(n=min(sample_size, len(df)), random_state=42)
        
        # Create map with OpenStreetMap tiles (no attribution needed)
        m = folium.Map(location=[22.351115, 78.667743], zoom_start=5)
        
        # Add a heatmap
        from folium.plugins import HeatMap
        heat_data = [[row['latitude'], row['longitude']] for _, row in sample_df.iterrows()]
        HeatMap(heat_data, radius=15).add_to(m)
        
        # Add circle markers for individual points
        for _, row in sample_df.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color='red',
                fill=True,
                fill_opacity=0.6,
                popup=f"Type: {row['type']}, FRP: {row['frp']:.2f}, Date: {row['acq_date'].strftime('%Y-%m-%d')}"
            ).add_to(m)
        
        st_folium(m, width=1200, height=600)
        
    elif viz_option == "Day/Night Observations":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='daynight', data=df, palette='pastel', ax=ax)
        ax.set_title('Distribution of Day/Night Observations', fontsize=14, pad=20)
        ax.set_xlabel('Day/Night', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        
    elif viz_option == "Satellite Distribution":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='satellite', data=df, palette='muted', ax=ax)
        ax.set_title('Distribution of Satellite Observations', fontsize=14, pad=20)
        ax.set_xlabel('Satellite', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        
    elif viz_option == "Version Distribution":
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x='version', data=df, palette='deep', ax=ax)
        ax.set_title('Distribution of Version', fontsize=14, pad=20)
        ax.set_xlabel('Version', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        st.pyplot(fig)
        
    elif viz_option == "Numerical Features Correlation":
        fig, ax = plt.subplots(figsize=(12, 8))
        correlation_matrix = df[['latitude', 'longitude', 'brightness', 'confidence', 'frp']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, center=0)
        ax.set_title('Correlation Heatmap of Numerical Features', fontsize=14, pad=20)
        st.pyplot(fig)
        
    elif viz_option == "Temporal Analysis":
        temp_option = st.radio("Select Temporal Analysis", ["By Month", "By Day of Week"], key="temp_radio")
        
        if temp_option == "By Month":
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=df, x='month', palette='viridis', ax=ax)
            ax.set_title('Fire Detections by Month', fontsize=14, pad=20)
            ax.set_xlabel('Month', fontsize=12)
            ax.set_ylabel('Number of Detections', fontsize=12)
            ax.set_xticks(range(12))
            ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
            st.pyplot(fig)
            
        else:
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.countplot(data=df, x='day_of_week', palette='viridis', ax=ax)
            ax.set_title('Fire Detections by Day of Week', fontsize=14, pad=20)
            ax.set_xlabel('Day of Week', fontsize=12)
            ax.set_ylabel('Number of Detections', fontsize=12)
            ax.set_xticks(range(7))
            ax.set_xticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
            st.pyplot(fig)

elif page == "Model Prediction":
    st.title("🔮 Fire Type Prediction")
    
    st.markdown("""
    <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin-bottom:20px;">
        <p>Use the interactive controls below to input fire detection parameters and predict the fire type.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get the actual min/max/mean values from the encoded data
    features_data = df_encoded[['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;">
            <h4 style="color:#343a40;">Primary Features</h4>
        </div>
        """, unsafe_allow_html=True)
        brightness = st.slider("Brightness", 
                             float(features_data['brightness'].min()), 
                             float(features_data['brightness'].max()), 
                             float(features_data['brightness'].mean()),
                             key="brightness")
        scan = st.slider("Scan", 
                       float(features_data['scan'].min()), 
                       float(features_data['scan'].max()), 
                       float(features_data['scan'].mean()),
                       key="scan")
        track = st.slider("Track", 
                        float(features_data['track'].min()), 
                        float(features_data['track'].max()), 
                        float(features_data['track'].mean()),
                        key="track")
        
    with col2:
        st.markdown("""
        <div style="background-color:#f8f9fa;padding:15px;border-radius:10px;">
            <h4 style="color:#343a40;">Secondary Features</h4>
        </div>
        """, unsafe_allow_html=True)
        confidence = st.slider("Confidence", 
                             int(features_data['confidence'].min()), 
                             int(features_data['confidence'].max()), 
                             int(features_data['confidence'].mean()),
                             key="confidence")
        bright_t31 = st.slider("Bright T31", 
                             float(features_data['bright_t31'].min()), 
                             float(features_data['bright_t31'].max()), 
                             float(features_data['bright_t31'].mean()),
                             key="bright_t31")
        frp = st.slider("Fire Radiative Power (FRP)", 
                       float(features_data['frp'].min()), 
                       float(features_data['frp'].max()), 
                       float(features_data['frp'].mean()),
                       key="frp")
    
    if st.button("Predict Fire Type", key="predict_btn"):
        # Prepare input data in the exact same format as training
        input_data = pd.DataFrame({
            'brightness': [brightness],
            'scan': [scan],
            'track': [track],
            'confidence': [confidence],
            'bright_t31': [bright_t31],
            'frp': [frp]
        })
        
        # Scale the input data
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        try:
            prediction = model.predict(input_scaled)
            proba = model.predict_proba(input_scaled)
            
            # Get the actual class names
            class_names = model.classes_
            
            st.markdown(f"""
            <div style="background-color:#d4edda;color:#155724;padding:15px;border-radius:5px;margin:20px 0;">
                <h3 style="color:#155724;">Predicted Fire Type: <strong>{class_names[prediction[0]]}</strong></h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probabilities
            st.subheader("Prediction Probabilities")
            proba_df = pd.DataFrame({
                'Fire Type': class_names,
                'Probability': proba[0]
            }).sort_values('Probability', ascending=False)
            
            st.dataframe(proba_df.style.format({'Probability': '{:.2%}'}))
            
            # Visualize probabilities
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.barplot(x='Probability', y='Fire Type', data=proba_df, palette='viridis', ax=ax)
            ax.set_title('Prediction Probabilities', fontsize=14, pad=20)
            ax.set_xlabel('Probability', fontsize=12)
            ax.set_ylabel('Fire Type', fontsize=12)
            st.pyplot(fig)
            
        except Exception as e:
            st.markdown(f"""
            <div style="background-color:#f8d7da;color:#721c24;padding:15px;border-radius:5px;">
                <h4>Prediction failed: {str(e)}</h4>
            </div>
            """, unsafe_allow_html=True)

elif page == "Model Evaluation":
    st.title("📊 Model Evaluation")
    
    # Get features and target
    features = ['brightness', 'scan', 'track', 'confidence', 'bright_t31', 'frp']
    target = 'type'
    X = df_encoded[features]
    y = df_encoded[target]
    
    # Apply SMOTE and split data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.25, random_state=42, stratify=y_resampled
    )
    
    # Train model (or use cached one)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Show metrics
    st.subheader("Classification Report")
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(12, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=model.classes_, 
               yticklabels=model.classes_, ax=ax)
    ax.set_title('Confusion Matrix', fontsize=14, pad=20)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("Feature Importance")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=importance, palette='viridis', ax=ax)
    ax.set_title('Feature Importance', fontsize=14, pad=20)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    st.pyplot(fig)
    
    # Model metrics cards
    accuracy = accuracy_score(y_test, y_pred)
    st.markdown("""
    <div style="display:flex;justify-content:space-between;margin:20px 0;">
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;width:30%;text-align:center;">
            <h3 style="color:#1890ff;">Accuracy</h3>
            <h2 style="color:#1890ff;">{:.2%}</h2>
        </div>
        <div style="background-color:#fff7e6;padding:15px;border-radius:10px;width:30%;text-align:center;">
            <h3 style="color:#faad14;">Precision (Avg)</h3>
            <h2 style="color:#faad14;">{:.2%}</h2>
        </div>
        <div style="background-color:#f6ffed;padding:15px;border-radius:10px;width:30%;text-align:center;">
            <h3 style="color:#52c41a;">Recall (Avg)</h3>
            <h2 style="color:#52c41a;">{:.2%}</h2>
        </div>
    </div>
    """.format(accuracy, report['weighted avg']['precision'], report['weighted avg']['recall']), unsafe_allow_html=True)

elif page == "About":
    # Create a white container with shadow for the title
    st.markdown("""
    <div style="background-color:white;padding:20px;border-radius:10px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);margin-bottom:20px;">
        <h1 style="color:#d9534f;text-align:center;">ℹ️ About</h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#f8f9fa;padding:20px;border-radius:10px;box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
        <h2 style="color:#d9534f;">Fire Classification Dashboard</h2>
        <p>India witnesses various types of fire incidents annually, including forest fires, agricultural burning, volcanic activity, and other thermal anomalies. Accurate identification of fire sources is crucial for timely disaster response, environmental monitoring, and resource management. The MODIS sensors aboard NASA’s Terra and Aqua satellites provide reliable, near real-time thermal anomaly data globally, including for India.

While the MODIS dataset includes rich geospatial and thermal parameters, the challenge lies in correctly classifying the type of fire event — whether it stems from vegetation, volcanoes, static land sources, or offshore sources — using satellite-captured features.

I developed a machine learning classification model that can accurately predict the type of fire using MODIS fire detection data for India from 2021 to 2023 </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color:#e6f7ff;padding:15px;border-radius:10px;margin:10px 0;">
            <h3 style="color:#1890ff;">Features</h3>
            <ul>
                <li>Data exploration and visualization</li>
                <li>Temporal analysis of fire events</li>
                <li>Machine learning model for fire type classification</li>
                <li>Model performance evaluation</li>
            </ul>
        </div>
        
        <div style="background-color:#f6ffed;padding:15px;border-radius:10px;margin:10px 0;">
            <h3 style="color:#52c41a;">Dataset</h3>
            <ul>
                <li>MODIS fire detection data for India (2021-2023)</li>
                <li>Contains information about fire location, brightness, confidence, and other parameters</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color:#fff7e6;padding:15px;border-radius:10px;margin:10px 0;">
            <h3 style="color:#faad14;">Models Used</h3>
            <ul>
                <li>Random Forest Classifier</li>
                <li>Uses SMOTE for handling class imbalance</li>
            </ul>
        </div>
        
        <div style="background-color:#f8f0ff;padding:15px;border-radius:10px;margin:10px 0;">
            <h3 style="color:#722ed1;">Technical Details</h3>
            <ul>
                <li>Built with Python and Streamlit</li>
                <li>Uses scikit-learn for machine learning</li>
                <li>Visualizations with Matplotlib and Seaborn</li>
                <li>Interactive maps with Folium</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background-color:#f0f2f5;padding:15px;border-radius:10px;margin-top:20px;text-align:center;">
        <h3 style="color:#2f4f4f;">Developed With ❤️ by TANMAY GUHA</h3>
        <p>© 2025 Fire Classification Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

# Add footer to all pages
footer()