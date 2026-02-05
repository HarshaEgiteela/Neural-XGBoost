import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
import joblib
import os

# ---------------------------------------------------------
# 1. ENHANCED DATA LOADING & FEATURE ENGINEERING
# ---------------------------------------------------------
def load_and_preprocess_data(filepath):
    if not os.path.exists(filepath):
        print(f"ERROR: File not found at: {filepath}")
        exit()

    print(f"Reading file: {filepath}...")
    try:
        xl = pd.ExcelFile(filepath, engine='openpyxl')
        sheet_name = 'EM-DAT Data' if 'EM-DAT Data' in xl.sheet_names else xl.sheet_names[0]
        df = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as e:
        print(f"Failed to read file: {e}")
        exit()
    
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    
    # Selection of Disaster Types for N-XGB Research
    target_disasters = ['Flood', 'Wildfire', 'Earthquake']
    df = df[df['Disaster Type'].isin(target_disasters)]
    
    # 1.1 ADVANCED FEATURE ENGINEERING
    # Log transformation for skewed numerical data
    df['Affected_Log'] = np.log1p(df['Total Affected'].fillna(0))
    df['Deaths_Log'] = np.log1p(df['Total Deaths'].fillna(0))
    
    # Magnitude Interaction
    df['Magnitude_Clean'] = df['Magnitude'].fillna(df['Magnitude'].median())
    df['Mag_Impact_Ratio'] = df['Magnitude_Clean'] * (df['Affected_Log'] + 1)

    # Seasonality Encoding (Cyclic)
    df['Month_Sin'] = np.sin(2 * np.pi * df['Start Month'].fillna(1)/12)
    df['Month_Cos'] = np.cos(2 * np.pi * df['Start Month'].fillna(1)/12)

    features = [
        'Start Year', 'Month_Sin', 'Month_Cos', 'Start Day', 
        'Country', 'Region', 'Subregion',
        'Magnitude_Clean', 'Latitude', 'Longitude',
        'Deaths_Log', 'No. Injured', 'Affected_Log', 'Mag_Impact_Ratio'
    ]
    target = 'Disaster Type'
    
    df = df.dropna(subset=[target])
    existing_features = [f for f in features if f in df.columns]
    X = df[existing_features].copy()
    y = df[target].copy()
    
    # 1.2 IMPUTATION & ENCODING
    num_cols = X.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        X[col] = X[col].fillna(X[col].median())
        
    cat_cols = X.select_dtypes(include=['object']).columns
    le_dict = {}
    for col in cat_cols:
        X[col] = X[col].fillna("Unknown")
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le
        
    target_le = LabelEncoder()
    y_encoded = target_le.fit_transform(y)
    
    return X, y_encoded, le_dict, target_le

# ---------------------------------------------------------
# 2. DEEP RESIDUAL FEATURE EXTRACTOR
# ---------------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.activation(x + self.block(x))

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim):
        super(FeatureExtractor, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            ResidualBlock(256),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            ResidualBlock(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32) # Final Embedding Space
        )
        
    def forward(self, x):
        return self.encoder(x)

# ---------------------------------------------------------
# 3. HYBRID TRAINING PIPELINE
# ---------------------------------------------------------
def train():
    # User default path
    default_path = r"C:\Users\harsh\OneDrive\Desktop\Xgb\2000.xlsx"
    X, y, encoders, target_encoder = load_and_preprocess_data(default_path)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Balancing classes with SMOTE...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.15, random_state=42
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train.shape[1]
    extractor = FeatureExtractor(input_dim).to(device)
    
    # 3.1 PHASE 1: NEURAL ENCODING
    print("Phase 1: Deep Feature Extraction Training...")
    # Add temporary classifier head for supervised embedding learning
    clf_head = nn.Linear(32, len(target_encoder.classes_)).to(device)
    
    optimizer = optim.AdamW(list(extractor.parameters()) + list(clf_head.parameters()), lr=0.001, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    
    extractor.train()
    for epoch in range(200): # Increased for residual convergence
        optimizer.zero_grad()
        embeddings = extractor(X_train_t)
        loss = criterion(clf_head(embeddings), y_train_t)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")
        
    # 3.2 PHASE 2: XGBOOST BOOSTING
    print("Phase 2: Extracting Embeddings and Training XGBoost...")
    extractor.eval()
    with torch.no_grad():
        X_train_deep = extractor(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        X_test_deep = extractor(torch.FloatTensor(X_test).to(device)).cpu().numpy()
    
    # Weighted boosting to prioritize difficult classes
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    xgb_model = xgb.XGBClassifier(
        n_estimators=1200,
        max_depth=12,            # High depth to capture deep feature interactions
        learning_rate=0.02,      # Very slow learning for maximum precision
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,               # Regularization to prevent overfitting at high depth
        objective='multi:softprob',
        tree_method='hist',
        random_state=42
    )
    
    xgb_model.fit(X_train_deep, y_train, sample_weight=sample_weights)
    
    # Final Evaluation
    y_pred = xgb_model.predict(X_test_deep)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {acc:.4%}")
    print("\nDetailed Performance Matrix:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))
    
    # Save optimized artifacts
    if not os.path.exists('models'): os.makedirs('models')
    torch.save(extractor.state_dict(), 'models/neural_extractor.pth')
    joblib.dump(xgb_model, 'models/xgb_classifier.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(target_encoder, 'models/target_encoder.pkl')
    joblib.dump(encoders, 'models/categorical_encoders.pkl')
    print("All High-Precision Artifacts Saved Successfully.")

if __name__ == "__main__":
    train()