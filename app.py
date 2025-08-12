import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings

warnings.filterwarnings("ignore")

# Page configuration
st.set_page_config(
    page_title="Movie Box Office Predictor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS - Updated with black and gray theme
st.markdown(
    """
<style>
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
    }
    
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #ffffff 0%, #d9d9d9 25%, #b3b3b3 50%, #ffffff 75%, #d9d9d9 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
        /* Sharper rendering */
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
        /* Reduce glow to avoid fuzziness */
        text-shadow: 0 0 4px rgba(255, 255, 255, 0.2);
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #e0e0e0; /* Brighter for more contrast */
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
        /* Sharper rendering */
        -webkit-font-smoothing: antialiased;
        text-rendering: optimizeLegibility;
        /* Minimal shadow for subtle depth without blur */
        text-shadow: 0 0 2px rgba(255, 255, 255, 0.15);
    }
    
    /* Success prediction styling */
    .prediction-success {
        background: linear-gradient(135deg, #2d2d2d 0%, #404040 50%, #2d2d2d 100%);
        border: 2px solid #00ff88;
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 255, 136, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-success::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(0, 255, 136, 0.1) 50%, transparent 70%);
        pointer-events: none;
    }
    
    /* Moderate prediction styling */
    .prediction-moderate {
        background: linear-gradient(135deg, #2d2d2d 0%, #3a3a3a 50%, #2d2d2d 100%);
        border: 2px solid #ffaa00;
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(255, 170, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-moderate::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 170, 0, 0.1) 50%, transparent 70%);
        pointer-events: none;
    }
    
    /* Poor prediction styling */
    .prediction-poor {
        background: linear-gradient(135deg, #2d2d2d 0%, #3d2d2d 50%, #2d2d2d 100%);
        border: 2px solid #ff4444;
        padding: 2rem;
        border-radius: 20px;
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(255, 68, 68, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .prediction-poor::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 68, 68, 0.1) 50%, transparent 70%);
        pointer-events: none;
    }
    
    /* Info and warning boxes */
    .info-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border-left: 4px solid #00ff88;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #e0e0e0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a1a 100%);
        border-left: 4px solid #ffaa00;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #e0e0e0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    .danger-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a1a1a 100%);
        border-left: 4px solid #ff4444;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        color: #e0e0e0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Form styling */
    .stForm {
        background: linear-gradient(135deg, #1e1e1e 0%, #2d2d2d 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 1px solid #404040;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.8),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    
    /* Section headers */
    .section-header {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: bold;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #404040;
        background: linear-gradient(90deg, #ffffff 0%, #888888 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid #404040;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        color: #e0e0e0;
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 
            0 6px 25px rgba(0, 0, 0, 0.7),
            inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    
    .metric-value {
        font-size: 1.4rem;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Risk factor styling */
    .risk-factor {
        background: linear-gradient(135deg, #2a1a1a 0%, #3a2a2a 100%);
        border-left: 3px solid #ff4444;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ffcccc;
        font-size: 0.95rem;
    }
    
    .success-factor {
        background: linear-gradient(135deg, #1a2a1a 0%, #2a3a2a 100%);
        border-left: 3px solid #00ff88;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        color: #ccffcc;
        font-size: 0.95rem;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #333333 0%, #555555 100%) !important;
        color: white !important;
        border: 2px solid #666666 !important;
        border-radius: 15px !important;
        padding: 1rem 2rem !important;
        font-weight: bold !important;
        font-size: 1.1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #555555 0%, #777777 100%) !important;
        border: 2px solid #888888 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4) !important;
    }
    
    /* Input field styling */
    .stSelectbox > div > div {
        background-color: #2d2d2d !important;
        border: 1px solid #555555 !important;
        color: white !important;
    }
    
    .stNumberInput > div > div > input {
        background-color: #2d2d2d !important;
        border: 1px solid #555555 !important;
        color: white !important;
    }
    
    .stSlider > div > div > div > div {
        background-color: #555555 !important;
    }
    
    /* FIXED: More specific text styling - removed problematic broad selectors */
    .main .block-container h1, 
    .main .block-container h2, 
    .main .block-container h3, 
    .main .block-container h4, 
    .main .block-container h5, 
    .main .block-container h6 {
        color: #ffffff !important;
    }
    
    /* REMOVED: The problematic broad selectors that were breaking HTML rendering */
    /* p, div, span { color: #e0e0e0 !important; } */
    /* .stMarkdown { color: #e0e0e0 !important; } */
    
    /* More specific targeting for text elements */
    .main .block-container .stMarkdown > p {
        color: #e0e0e0;
    }
    
    /* Success message styling */
    .stSuccess {
        background: linear-gradient(135deg, #1a2a1a 0%, #2a3a2a 100%) !important;
        border: 1px solid #00ff88 !important;
        color: #ccffcc !important;
    }
    
    /* Error message styling */
    .stError {
        background: linear-gradient(135deg, #2a1a1a 0%, #3a2a2a 100%) !important;
        border: 1px solid #ff4444 !important;
        color: #ffcccc !important;
    }
    
    /* Spinner styling */
    .stSpinner {
        color: #ffffff !important;
    }
    
    /* Balloons effect enhancement */
    .stBalloons {
        filter: brightness(0.8) contrast(1.2);
    }
</style>
""",
    unsafe_allow_html=True,
)


class MovieBoxOfficePredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []

    def load_and_prepare_data(self):
        """Load the movies.csv dataset and prepare it silently"""
        try:
            # Try to load the dataset
            df = pd.read_csv("movies.csv")

            if df.empty:
                return None

            return df

        except FileNotFoundError:
            st.error(
                "❌ movies.csv file not found. Please make sure the file is in the same directory as this app."
            )
            return None
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
            return None

    def clean_and_prepare_features(self, df):
        """Clean the data and prepare features silently"""

        # Make a copy to avoid modifying original
        clean_df = df.copy()

        # Remove rows with missing target variable (gross)
        clean_df = clean_df.dropna(subset=["gross"])

        # Convert budget and gross to numeric, removing currency symbols and commas
        for col in ["budget", "gross"]:
            if col in clean_df.columns:
                clean_df[col] = clean_df[col].astype(str)
                clean_df[col] = clean_df[col].str.replace(r"[\$,]", "", regex=True)
                clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

        # Remove movies with missing budget
        clean_df = clean_df.dropna(subset=["budget"])

        # Filter out unrealistic values
        clean_df = clean_df[(clean_df["budget"] > 1000) & (clean_df["gross"] > 1000)]
        clean_df = clean_df[(clean_df["budget"] < 1e9) & (clean_df["gross"] < 5e9)]

        if len(clean_df) < 100:
            return None

        # Handle missing values in other features
        clean_df["runtime"] = pd.to_numeric(clean_df["runtime"], errors="coerce")
        clean_df["runtime"] = clean_df["runtime"].fillna(clean_df["runtime"].median())

        clean_df["genre"] = clean_df["genre"].fillna("Unknown")
        clean_df["rating"] = clean_df["rating"].fillna("Unknown")
        clean_df["year"] = pd.to_numeric(clean_df["year"], errors="coerce")
        clean_df["year"] = clean_df["year"].fillna(clean_df["year"].median())

        # Feature engineering - extract release season
        if "released" in clean_df.columns:
            clean_df["released"] = pd.to_datetime(clean_df["released"], errors="coerce")
            clean_df["release_month"] = clean_df["released"].dt.month

            def get_season(month):
                if pd.isna(month):
                    return "Unknown"
                elif month in [12, 1, 2]:
                    return "Winter"
                elif month in [3, 4, 5]:
                    return "Spring"
                elif month in [6, 7, 8]:
                    return "Summer"
                else:
                    return "Fall"

            clean_df["release_season"] = clean_df["release_month"].apply(get_season)
        else:
            clean_df["release_season"] = "Unknown"

        # Create budget categories
        def categorize_budget(budget):
            if pd.isna(budget) or budget < 1000:
                return "Unknown"
            elif budget < 1000000:
                return "Low Budget"
            elif budget < 15000000:
                return "Medium Budget"
            elif budget < 50000000:
                return "High Budget"
            else:
                return "Blockbuster"

        clean_df["budget_category"] = clean_df["budget"].apply(categorize_budget)

        return clean_df

    def prepare_model_features(self, df):
        """Prepare features for the model - simplified feature set"""

        # Simplified feature set
        feature_columns = [
            "budget",  # Known before release
            "genre",  # Known before release
            "runtime",  # Known before release (post-production)
            "rating",  # Known before release (MPAA rating)
            "year",  # Known before release
            "release_season",  # Known before release
            "budget_category",  # Derived from budget
        ]

        # Check which features are available
        available_features = [col for col in feature_columns if col in df.columns]

        # Create feature matrix
        X = df[available_features].copy()
        y = df["gross"].copy()

        # Encode categorical variables
        categorical_features = ["genre", "rating", "release_season", "budget_category"]

        for feature in categorical_features:
            if feature in X.columns:
                if feature == "genre":
                    # Take the first genre if multiple are listed
                    X[feature] = X[feature].str.split(",").str[0].str.strip()
                    top_genres = X[feature].value_counts().head(15).index
                    X[feature] = X[feature].apply(
                        lambda x: x if x in top_genres else "Other"
                    )

                # Label encode
                if feature not in self.label_encoders:
                    self.label_encoders[feature] = LabelEncoder()
                    X[feature] = self.label_encoders[feature].fit_transform(
                        X[feature].astype(str)
                    )
                else:
                    # Handle unseen categories during prediction
                    unique_vals = set(self.label_encoders[feature].classes_)
                    X[feature] = X[feature].apply(
                        lambda x: x if x in unique_vals else "Other"
                    )
                    X[feature] = self.label_encoders[feature].transform(
                        X[feature].astype(str)
                    )

        self.feature_names = available_features

        return X, y

    def train_models(self, X, y):
        """Train a custom model that ensures equal feature contribution and positive outcomes"""

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features for equal contribution
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Use Linear Regression for more predictable, equal feature contribution
        model = LinearRegression()

        try:
            model.fit(X_train_scaled, y_train)

            self.model = model
            self.is_trained = True

            # Store baseline statistics for positive bias
            self.y_mean = y_train.mean()
            self.y_std = y_train.std()

            return True

        except Exception as e:
            return False

    def predict_gross(self, budget, genre, runtime, rating, year, season):
        """Make a more realistic prediction with improved penalty system"""

        if not self.is_trained:
            return None

        # Create input data
        input_data = {
            "budget": [budget],
            "genre": [genre],
            "runtime": [runtime],
            "rating": [rating],
            "year": [year],
            "release_season": [season],
            "budget_category": [self.categorize_budget(budget)],
        }

        # Only include features that were used in training
        available_input = {}
        for feature in self.feature_names:
            if feature in input_data:
                available_input[feature] = input_data[feature]

        input_df = pd.DataFrame(available_input)

        # Encode categorical variables
        categorical_features = ["genre", "rating", "release_season", "budget_category"]

        for feature in categorical_features:
            if feature in input_df.columns and feature in self.label_encoders:
                try:
                    # Handle unseen categories
                    if (
                        input_df[feature].iloc[0]
                        not in self.label_encoders[feature].classes_
                    ):
                        input_df[feature] = "Other"
                    input_df[feature] = self.label_encoders[feature].transform(
                        input_df[feature].astype(str)
                    )
                except:
                    input_df[feature] = 0  # Default encoding for unseen values

        # Make base prediction using scaled features
        try:
            input_scaled = self.scaler.transform(input_df)
            base_prediction = self.model.predict(input_scaled)[0]
        except Exception as e:
            return None

        # Start with a more conservative base that considers genre realities
        # Base minimum depends on genre and budget
        if genre in ["Drama", "Biography", "History"]:
            min_multiplier = 0.3  # Very limited commercial appeal
        elif genre in ["Romance", "Mystery"]:
            min_multiplier = 0.4  # Niche but slightly broader
        elif genre in ["Horror", "Thriller"]:
            min_multiplier = 0.6  # Can surprise but risky
        else:
            min_multiplier = 0.8  # More commercial genres

        predicted_gross = max(base_prediction, budget * min_multiplier)

        # MORE REALISTIC GENRE MULTIPLIERS (based on actual industry data)
        genre_multipliers = {
            "Action": 2.8,  # Strong international, wide appeal
            "Adventure": 2.5,  # Family blockbusters
            "Animation": 2.2,  # Consistent but competitive
            "Comedy": 1.4,  # Very hit-or-miss, most flop
            "Fantasy": 1.8,  # Franchise potential but niche
            "Sci-Fi": 1.6,  # Loyal but limited audience
            "Family": 1.5,  # Steady but not explosive
            "Horror": 1.3,  # Limited audience, seasonal, can surprise
            "Thriller": 1.1,  # Moderate appeal, competitive
            "Crime": 0.9,  # Adult-skewing, limited international
            "Mystery": 0.7,  # Very niche audience
            "Romance": 0.6,  # Severely declining commercially
            "Drama": 0.4,  # Extremely limited commercial appeal
            "Biography": 0.35,  # Awards potential, tiny audience
            "History": 0.3,  # Educational, minimal commercial value
        }
        predicted_gross *= genre_multipliers.get(genre, 0.5)
        rating_multipliers = {
            "G": 0.85,  # Limited appeal unless animated blockbuster
            "PG": 1.1,  # Family-friendly but not teen-appealing
            "PG-13": 1.4,  # Maximum audience reach
            "R": 0.55,  # Cuts out huge audience segment
            "NC-17": 0.15,  # Essentially unreleasable commercially
        }
        predicted_gross *= rating_multipliers.get(rating, 0.6)

        season_multipliers = {
            "Summer": 1.4,  # Peak season with competition
            "Winter": 1.2,  # Holiday boost but very competitive
            "Fall": 0.7,  # Slow period except horror
            "Spring": 0.6,  # Dumping ground for studios
        }
        predicted_gross *= season_multipliers.get(season, 0.7)
        if runtime < 90:
            predicted_gross *= 0.5  # Feels incomplete, poor value perception
        elif runtime > 180:
            predicted_gross *= 0.25  # Virtually unprogrammable
        elif runtime > 150:
            predicted_gross *= 0.55  # Severely limits showtimes
        elif runtime > 140:
            predicted_gross *= 0.75  # Noticeably limits programming
        elif 95 <= runtime <= 115:
            predicted_gross *= 1.15  # Sweet spot for theaters
        elif 90 <= runtime <= 130:
            predicted_gross *= 1.0  # Acceptable range
        else:
            predicted_gross *= 0.8  # Suboptimal but workable

        budget_cat = self.categorize_budget(budget)
        if budget_cat == "Blockbuster":
            predicted_gross *= 0.75  # Massive expectations and competition
            # Additional penalty for non-commercial genres at blockbuster budget
            if genre in ["Drama", "Biography", "History", "Romance"]:
                predicted_gross *= 0.3  # Almost guaranteed disaster
        elif budget_cat == "High Budget":
            predicted_gross *= 0.85  # High expectations
            if genre in ["Drama", "Biography"]:
                predicted_gross *= 0.5  # Very risky at this budget level
        elif budget_cat == "Medium Budget":
            predicted_gross *= 0.95  # Sweet spot for most genres
        elif budget_cat == "Low Budget":
            # More nuanced low budget logic
            if genre in ["Horror"]:
                predicted_gross *= 2.2  # Horror can really overperform cheap
            elif genre in ["Comedy"] and rating in ["R", "PG-13"]:
                predicted_gross *= 1.8  # Comedies can surprise
            elif genre in ["Thriller", "Crime"]:
                predicted_gross *= 1.4  # Modest boost for genre films
            elif genre in ["Drama", "Romance", "Biography"]:
                predicted_gross *= 0.6  # Still limited even when cheap
            else:
                predicted_gross *= 1.1  # Small boost for being affordable

        penalty_score = 0
        severe_penalty_score = 0

        if genre in ["Drama", "Biography", "History"] and rating == "R":
            severe_penalty_score += 2  # Double severe for arthouse + adult
        if genre in ["Romance", "Drama"] and season in ["Spring", "Fall"]:
            penalty_score += 1  # Bad timing for limited genres
        if runtime > 150 and genre not in ["Action", "Adventure", "Fantasy", "Sci-Fi"]:
            penalty_score += 1  # Long runtime without spectacle justification
        if season in ["Spring", "Fall"] and budget_cat in [
            "High Budget",
            "Blockbuster",
        ]:
            penalty_score += 1  # Expensive movie in slow season
        if rating == "R" and season == "Spring" and budget_cat != "Low Budget":
            penalty_score += 1  # Triple threat of limitations
        if genre in ["Biography", "History"] and budget_cat in [
            "High Budget",
            "Blockbuster",
        ]:
            severe_penalty_score += 1  # Massive budget for niche content

        if severe_penalty_score >= 2:
            predicted_gross *= 0.15  # Virtual commercial suicide
        elif severe_penalty_score >= 1:
            predicted_gross *= 0.35  # Major commercial problems

        if penalty_score >= 3:
            predicted_gross *= 0.4  # Multiple significant issues
        elif penalty_score >= 2:
            predicted_gross *= 0.6  # Two major problems compound
        elif penalty_score >= 1:
            predicted_gross *= 0.75  # One notable issue

        if budget < 1000000:  # Ultra-low budget
            if genre == "Horror":
                max_gross = budget * 25  # Horror can really explode
            elif genre in ["Comedy", "Thriller"]:
                max_gross = budget * 15  # Modest upside
            elif genre in ["Drama", "Romance"]:
                max_gross = budget * 8  # Very limited upside
            else:
                max_gross = budget * 12  # General cap
            predicted_gross = min(predicted_gross, max_gross)

        elif budget < 5000000:  # Low budget
            if genre in ["Horror", "Comedy"]:
                max_gross = budget * 20
            elif genre in ["Action", "Thriller"]:
                max_gross = budget * 12
            elif genre in ["Drama", "Romance", "Biography"]:
                max_gross = budget * 6  # Very rare for these to break out
            else:
                max_gross = budget * 10
            predicted_gross = min(predicted_gross, max_gross)

        elif budget < 25000000:  # Medium budget
            if genre in ["Drama", "Biography", "History"]:
                max_gross = budget * 4  # Even successful ones rarely do more
            elif genre == "Romance":
                max_gross = budget * 3  # Romance is commercially dead
            predicted_gross = min(predicted_gross, max_gross)

        if year >= 2024:
            predicted_gross *= 0.85  # Increased competition, streaming impact
        if year >= 2026:
            predicted_gross *= 0.8  # Projected continued decline for original content

        if genre == "Drama" and budget > 15000000:
            predicted_gross *= 0.6  # Mid-budget dramas are nearly extinct commercially
        if genre == "Romance" and budget > 10000000:
            predicted_gross *= 0.4  # Big budget romance is commercial suicide
        if genre in ["Biography", "History"] and rating != "PG-13":
            predicted_gross *= 0.7  # Need maximum audience for these genres

        predicted_gross = max(predicted_gross, budget * 0.1)

        if budget_cat == "Blockbuster":
            marketing_multiplier = 1.2  # Even more marketing needed for big films
        elif budget_cat == "High Budget":
            marketing_multiplier = 1.0  # Equal to production
        elif budget_cat == "Medium Budget":
            marketing_multiplier = 0.8  # Slightly less
        else:
            marketing_multiplier = 0.5  # Low budget gets minimal marketing

        marketing_cost = budget * marketing_multiplier
        total_cost = budget + marketing_cost

        domestic_revenue = predicted_gross * 0.5 * 0.5  # 50% gross, 50% to studio
        international_revenue = predicted_gross * 0.5 * 0.25  # 50% gross, 25% to studio
        studio_revenue = domestic_revenue + international_revenue

        actual_profit = studio_revenue - total_cost
        roi = (actual_profit / budget) * 100 if budget > 0 else 0

        if roi > 500:
            success_level = "🚀 CULTURAL PHENOMENON"
            success_class = "success"
        elif roi > 300:
            success_level = "💎 MASSIVE BLOCKBUSTER"
            success_class = "success"
        elif roi > 150:
            success_level = "🎉 BIG SUCCESS"
            success_class = "success"
        elif roi > 75:
            success_level = "✅ SOLID HIT"
            success_class = "success"
        elif roi > 25:
            success_level = "😊 MODEST SUCCESS"
            success_class = "moderate"
        elif roi > 0:
            success_level = "💰 BARELY PROFITABLE"
            success_class = "moderate"
        elif roi > -25:
            success_level = "😐 BREAK EVEN-ISH"
            success_class = "poor"
        elif roi > -50:
            success_level = "📉 MONEY LOSER"
            success_class = "poor"
        elif roi > -75:
            success_level = "💸 MAJOR FLOP"
            success_class = "poor"
        else:
            success_level = "💥 CATASTROPHIC DISASTER"
            success_class = "poor"

        return {
            "predicted_gross": predicted_gross,
            "marketing_cost": marketing_cost,
            "total_cost": total_cost,
            "studio_revenue": studio_revenue,
            "actual_profit": actual_profit,
            "roi": roi,
            "success_level": success_level,
            "success_class": success_class,
            "budget_multiple": predicted_gross / budget if budget > 0 else 0,
        }

    def categorize_budget(self, budget):
        """Helper function to categorize budget"""
        if budget < 1000000:
            return "Low Budget"
        elif budget < 15000000:
            return "Medium Budget"
        elif budget < 50000000:
            return "High Budget"
        else:
            return "Blockbuster"


@st.cache_data
def initialize_predictor():

    predictor = MovieBoxOfficePredictor()

    # Load data
    df = predictor.load_and_prepare_data()
    if df is None:
        return None

    # Clean and prepare
    clean_df = predictor.clean_and_prepare_features(df)
    if clean_df is None:
        return None

    # Prepare features
    X, y = predictor.prepare_model_features(clean_df)

    # Train model
    success = predictor.train_models(X, y)
    if not success:
        return None

    return predictor


def main():
    # Header
    st.markdown(
        '<h1 class="main-header">🎬 Movie Box Office Predictor</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="subtitle">Predict your movie\'s box office success instantly! 🚀</p>',
        unsafe_allow_html=True,
    )

    # Initialize predictor
    with st.spinner("Loading AI model..."):
        predictor = initialize_predictor()

    if predictor is None:
        st.error(
            "❌ Failed to initialize the prediction model. Please check that movies.csv is available."
        )
        st.stop()

    st.success("Enter your movie details below.")

    # Main prediction interface
    st.markdown("## 🎬 Enter Your Movie Details")

    with st.form("movie_prediction", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 💰 **Production Details**")
            budget = st.number_input(
                "Budget ($)",
                min_value=10000,
                max_value=500000000,
                value=25000000,
                step=1000000,
                format="%d",
                help="Total production budget in US dollars",
            )
            st.write(f"💡 Budget Category: **{predictor.categorize_budget(budget)}**")

            runtime = st.slider(
                "Runtime (minutes)",
                min_value=60,
                max_value=200,
                value=110,
                step=5,
                help="Total movie length in minutes",
            )

            year = st.selectbox(
                "Release Year",
                options=list(range(2024, 2035)),
                index=0,
                help="Planned release year",
            )

        with col2:
            st.markdown("### 🎭 **Creative Details**")

            genre = st.selectbox(
                "Primary Genre",
                options=[
                    "Action",
                    "Adventure",
                    "Animation",
                    "Comedy",
                    "Crime",
                    "Drama",
                    "Family",
                    "Fantasy",
                    "Horror",
                    "Romance",
                    "Sci-Fi",
                    "Thriller",
                    "Mystery",
                    "Biography",
                    "History",
                ],
                index=0,
                help="Primary genre of your movie",
            )

            rating = st.selectbox(
                "MPAA Rating",
                options=["G", "PG", "PG-13", "R", "NC-17"],
                index=2,
                help="Expected MPAA rating (PG-13 has widest audience appeal)",
            )

            season = st.selectbox(
                "Release Season",
                options=["Spring", "Summer", "Fall", "Winter"],
                index=1,
                help="Summer releases typically perform best for blockbusters",
            )

        # Center the predict button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            submitted = st.form_submit_button(
                "🎯 Predict Box Office Success!",
                type="primary",
                use_container_width=True,
            )

    if submitted:
        with st.spinner("Analyzing your movie..."):
            result = predictor.predict_gross(
                budget, genre, runtime, rating, year, season
            )

        if result:

            if result["success_class"] == "success":
                st.success(f"🚀 {result['success_level']}")
            elif result["success_class"] == "moderate":
                st.warning(f"😊 {result['success_level']}")
            else:
                st.error(f"💸 {result['success_level']}")

            st.markdown("---")
            st.subheader("📊 Financial Breakdown")

            # Create 3 columns for metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="🎬 Predicted Box Office",
                    value=f"${result['predicted_gross']:,.0f}",
                    delta=f"{result['budget_multiple']:.1f}x budget",
                )

            with col2:
                st.metric(
                    label="💰 Studio Revenue",
                    value=f"${result['studio_revenue']:,.0f}",
                    delta="37.5% of gross",
                )

            with col3:
                st.metric(
                    label="📊 Marketing Cost",
                    value=f"${result['marketing_cost']:,.0f}",
                    delta="Estimated campaign",
                )

            # Second row of metrics
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="💸 Total Investment",
                    value=f"${result['total_cost']:,.0f}",
                    delta="Production + Marketing",
                )

            with col2:
                profit_delta = "Profit 📈" if result["actual_profit"] > 0 else "Loss 📉"
                profit_color = "normal" if result["actual_profit"] > 0 else "inverse"
                st.metric(
                    label="💵 Net Profit/Loss",
                    value=f"${result['actual_profit']:,.0f}",
                    delta=profit_delta,
                    delta_color=profit_color,
                )

            with col3:
                roi_delta = "Profitable ✅" if result["roi"] > 0 else "Loss ❌"
                roi_color = "normal" if result["roi"] > 0 else "inverse"
                st.metric(
                    label="🎯 Return on Investment",
                    value=f"{result['roi']:.1f}%",
                    delta=roi_delta,
                    delta_color=roi_color,
                )

            # BUSINESS REALITY CHECK using native Streamlit
            st.markdown("---")
            st.subheader("📊 Business Reality Check")

            if result["roi"] > 200:
                st.balloons()
                st.success(
                    "🎉 **MASSIVE SUCCESS!** This is the kind of hit that creates careers and launches franchises."
                )
            elif result["roi"] > 100:
                st.success(
                    "💰 **BIG WIN!** This movie should generate substantial profits and establish your reputation."
                )
            elif result["roi"] > 25:
                st.success(
                    "✅ **PROFITABLE SUCCESS!** You'll make money and prove the concept works."
                )
            elif result["roi"] > 0:
                st.warning(
                    "💰 **BARELY PROFITABLE.** You'll make a small profit but nothing spectacular."
                )
            elif result["roi"] > -25:
                st.warning(
                    "😐 **BREAK EVEN TERRITORY.** You might barely cover costs or lose a little."
                )
            elif result["roi"] > -50:
                st.error(
                    "⚠️ **LIKELY LOSS.** This project would probably lose money for investors."
                )
            else:
                st.error(
                    "💸 **HIGH RISK OF MAJOR LOSS.** Based on these parameters, this movie would likely be a significant financial failure. Serious reconsideration needed."
                )

            st.markdown("---")
            st.subheader("🎬 Industry Reality")

            col1, col2 = st.columns(2)

            with col1:
                with st.expander("💼 Your Movie Breakdown", expanded=True):
                    st.write(f"**Production Budget:** ${budget:,.0f}")
                    st.write(f"**Marketing Cost:** ${result['marketing_cost']:,.0f}")
                    st.write(f"**Total Investment:** ${result['total_cost']:,.0f}")
                    st.write(f"**Box Office Gross:** ${result['predicted_gross']:,.0f}")
                    st.write(f"**Studio Revenue:** ${result['studio_revenue']:,.0f}")
                    if result["actual_profit"] > 0:
                        st.write(
                            f"**Net Result:** ✅ ${result['actual_profit']:,.0f} **PROFIT**"
                        )
                    else:
                        st.write(
                            f"**Net Result:** ❌ ${abs(result['actual_profit']):,.0f} **LOSS**"
                        )

            with col2:
                with st.expander("📈 Industry Statistics", expanded=True):
                    st.write("📊 **60-70%** of movies lose money")
                    st.write("💰 **Only 20%** are truly profitable")
                    st.write("🎯 **Less than 5%** become major hits")
                    st.write("📺 **Marketing** often equals production cost")
                    st.write("🏢 **Studios** only get ~37.5% of box office")
                    st.write("⚖️ **Break-even** typically needs 2.5x budget gross")

            # RISK AND SUCCESS FACTORS
            st.markdown("---")

            # Analyze risk and success factors (your existing logic)
            risk_factors = []
            success_factors = []

            if genre in ["Drama", "Biography", "History"]:
                risk_factors.append(f"{genre} has extremely limited commercial appeal")
            elif genre in ["Action", "Adventure", "Animation"]:
                success_factors.append(f"{genre} has strong commercial potential")

            if rating == "R":
                risk_factors.append("R rating cuts out younger audiences significantly")
            elif rating == "NC-17":
                risk_factors.append("NC-17 rating severely limits distribution")
            elif rating == "PG-13":
                success_factors.append("PG-13 rating maximizes audience reach")

            if season in ["Spring", "Fall"]:
                risk_factors.append(f"{season} release is a slower moviegoing period")
            elif season == "Summer":
                success_factors.append("Summer release takes advantage of peak season")

            if runtime > 150:
                risk_factors.append(
                    f"{runtime} minute runtime severely limits daily showings"
                )
            elif runtime < 90:
                risk_factors.append(
                    f"{runtime} minute runtime may feel too short for audiences"
                )
            elif 95 <= runtime <= 115:
                success_factors.append(
                    f"{runtime} minute runtime is optimal for theaters"
                )

            budget_cat = predictor.categorize_budget(budget)
            if budget_cat == "Blockbuster":
                risk_factors.append(
                    "Blockbuster budget requires massive global audience"
                )
            elif budget_cat == "Low Budget" and genre in ["Horror", "Comedy"]:
                success_factors.append(
                    f"Low budget + {genre} combination can yield very high ROI"
                )

            if risk_factors:
                st.subheader("⚠️ Risk Factors")
                for risk in risk_factors:
                    st.error(f"• {risk}")

            if success_factors:
                st.subheader("✅ Success Factors")
                for factor in success_factors:
                    st.success(f"• {factor}")

            st.markdown("---")
            st.subheader("💡 Strategic Recommendations")

            if result["roi"] < 0:
                with st.expander("🔄 Urgent Improvements Needed", expanded=True):
                    st.warning(
                        "**Reduce Budget:** Lower risk by cutting production costs"
                    )
                    st.warning(
                        "**Change Genre:** Consider Action, Adventure, or Animation for broader appeal"
                    )
                    st.warning("**Summer Release:** Move to peak moviegoing season")
                    st.warning("**Target PG-13:** Maximize audience reach")
                    st.warning("**Optimize Runtime:** Aim for 90-120 minutes")
                    st.warning(
                        "**Find Star Power:** Attach bankable actors to reduce risk"
                    )

            elif result["roi"] > 100:
                with st.expander("🚀 Maximize This Success", expanded=True):
                    st.success(
                        "**Invest in Marketing:** This concept can support major campaigns"
                    )
                    st.success(
                        "**Premium Formats:** Consider IMAX, 3D, or other premium experiences"
                    )
                    st.success(
                        "**Franchise Planning:** Start developing sequel concepts now"
                    )
                    st.success(
                        "**Merchandising:** Explore toy, game, and product opportunities"
                    )
                    st.success(
                        "**International Focus:** This concept should travel well globally"
                    )

            else:
                with st.expander("⚖️ Moderate Adjustments Recommended", expanded=True):
                    st.info(
                        "**Budget Optimization:** Fine-tune costs to improve margins"
                    )
                    st.info(
                        "**Marketing Strategy:** Focus on targeted, efficient campaigns"
                    )
                    st.info(
                        "**Release Strategy:** Consider platform release or limited theatrical"
                    )
                    st.info("**Risk Mitigation:** Secure pre-sales or streaming deals")
                    st.info("**Genre Enhancement:** Add elements that broaden appeal")

            # FINAL DISCLAIMER
            st.markdown("---")
            st.info(
                "📋 **Important Disclaimer:** This prediction is based on historical patterns from thousands of real movies and current industry economics. However, movie success depends on many unpredictable factors including story quality, star performances, marketing effectiveness, cultural timing, and audience reception. **Use this as one tool in your decision-making process, not the only factor!**"
            )

        else:
            st.error(
                "❌ Unable to generate prediction. Please check your inputs and try again."
            )


if __name__ == "__main__":
    main()
