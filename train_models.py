"""
InsureAI Model Training Pipeline
Trains and saves models for:
- Module 1: ML Regression (Premium Prediction) & Classification (Fraud Detection)
- Module 2: Deep Learning (ANN) on Tabular Data
- Module 3: CNN (Vehicle Damage Detection with Transfer Learning)
- Module 4: RNN/LSTM (Customer Review Sentiment Analysis)
"""

import os
import sys
import json
import pickle
import joblib

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

os.makedirs('models', exist_ok=True)
os.makedirs('assets', exist_ok=True)

metrics_summary = {}

# ==========================================
# 1. MODULE 1 (PART A) & MODULE 2: PREMIUM REGRESSION
# ==========================================
print("\n" + "="*50)
print("TRAINING MODULE 1 & 2: PREMIUM REGRESSION (ML & ANN)")
print("="*50)

df_prem = pd.read_csv('data/insurance_premium.csv')
print(f"Loaded premium data: {df_prem.shape}")

# Core feature set for interactive app & baseline
core_features = ['age', 'gender', 'bmi', 'children', 'smoker', 'region']
df_p = df_prem[core_features + ['annual_premium_inr']].copy()
df_p['bmi'] = df_p['bmi'].fillna(df_p['bmi'].median())

# Train/Test Split (80/20) before any encoding/scaling
X_p = df_p[core_features]
y_p = df_p['annual_premium_inr'].values

X_p_train, X_p_test, y_p_train, y_p_test = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42
)

# Preprocessing: Map binary categoricals, One-Hot multi-category
def encode_premium_df(df, fit=False, encoders=None):
    df_enc = df.copy()
    df_enc['gender'] = df_enc['gender'].str.lower().map({'male': 1, 'female': 0}).fillna(0)
    df_enc['smoker'] = df_enc['smoker'].str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    
    # One-hot encode region
    if fit:
        encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
        region_encoded = encoder.fit_transform(df_enc[['region']])
        feature_names = [f"region_{c}" for c in encoder.categories_[0][1:]]
        return df_enc, encoder, feature_names
    else:
        encoder = encoders['region_encoder']
        region_encoded = encoder.transform(df_enc[['region']])
        return df_enc, region_encoded

X_p_train_enc, region_encoder, region_cols = encode_premium_df(X_p_train, fit=True)
X_p_train_reg = region_encoder.transform(X_p_train[['region']])
X_p_test_enc, X_p_test_reg = encode_premium_df(X_p_test, fit=False, encoders={'region_encoder': region_encoder})

numeric_cols = ['age', 'bmi', 'children']
scaler_p = StandardScaler()
X_p_train_num = scaler_p.fit_transform(X_p_train_enc[numeric_cols])
X_p_test_num = scaler_p.transform(X_p_test_enc[numeric_cols])

# Assembled feature matrix: [gender, smoker, age_scaled, bmi_scaled, children_scaled, region_dummies]
X_p_train_final = np.hstack([
    X_p_train_enc[['gender', 'smoker']].values,
    X_p_train_num,
    X_p_train_reg
])
X_p_test_final = np.hstack([
    X_p_test_enc[['gender', 'smoker']].values,
    X_p_test_num,
    X_p_test_reg
])

feature_order_premium = ['gender', 'smoker'] + numeric_cols + [f"region_{c}" for c in region_encoder.categories_[0][1:]]

# Train Classical ML Models
models_reg = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
}

reg_results = {}
best_reg_name = None
best_r2 = -float('inf')
best_reg_model = None

for name, model in models_reg.items():
    model.fit(X_p_train_final, y_p_train)
    y_pred = model.predict(X_p_test_final)
    r2 = r2_score(y_p_test, y_pred)
    mae = mean_absolute_error(y_p_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_p_test, y_pred))
    reg_results[name] = {'R2': round(float(r2), 4), 'MAE': round(float(mae), 2), 'RMSE': round(float(rmse), 2)}
    print(f"  {name:18} | R2: {r2:.4f} | MAE: INR {mae:,.2f} | RMSE: INR {rmse:,.2f}")
    if r2 > best_r2:
        best_r2 = r2
        best_reg_name = name
        best_reg_model = model

# Train Module 2: Deep Learning (ANN) Regressor
print("\nTraining ANN Regressor...")
ann_reg = Sequential([
    Input(shape=(X_p_train_final.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])

ann_reg.compile(optimizer='adam', loss='mse', metrics=['mae'])
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history_ann_reg = ann_reg.fit(
    X_p_train_final, y_p_train,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    callbacks=[early_stop],
    verbose=0
)

y_pred_ann = ann_reg.predict(X_p_test_final, verbose=0).flatten()
ann_r2 = r2_score(y_p_test, y_pred_ann)
ann_mae = mean_absolute_error(y_p_test, y_pred_ann)
ann_rmse = np.sqrt(mean_squared_error(y_p_test, y_pred_ann))
reg_results['ANN (Keras)'] = {'R2': round(float(ann_r2), 4), 'MAE': round(float(ann_mae), 2), 'RMSE': round(float(ann_rmse), 2)}
print(f"  {'ANN (Keras)':18} | R2: {ann_r2:.4f} | MAE: INR {ann_mae:,.2f} | RMSE: INR {ann_rmse:,.2f}")

# Save regression artifacts
joblib.dump(best_reg_model, 'models/premium_model.pkl')
joblib.dump(scaler_p, 'models/premium_scaler.pkl')
joblib.dump(region_encoder, 'models/premium_region_encoder.pkl')
with open('models/premium_features.json', 'w') as f:
    json.dump({
        'numeric_cols': numeric_cols,
        'feature_order': feature_order_premium,
        'best_model': best_reg_name,
        'metrics': reg_results
    }, f, indent=2)

ann_reg.save('models/ann_premium_model.keras')
print("Saved models/premium_model.pkl and models/ann_premium_model.keras")
metrics_summary['regression'] = reg_results

# ==========================================
# 2. MODULE 1 (PART B) & MODULE 2: FRAUD DETECTION
# ==========================================
print("\n" + "="*50)
print("TRAINING MODULE 1 & 2: FRAUD CLASSIFICATION (ML & ANN)")
print("="*50)

df_fraud = pd.read_csv('data/insurance_claims.csv')
print(f"Loaded fraud claims data: {df_fraud.shape}")

# Preprocessing: drop leakage/ID column
df_f = df_fraud.drop(columns=['claim_id']).copy()
df_f['days_to_report'] = df_f['days_to_report'].fillna(df_f['days_to_report'].median())

# Map target Y/N to 1/0
df_f['fraud_reported'] = (df_f['fraud_reported'] == 'Y').astype(int)

# Separate features & target
X_f = df_f.drop(columns=['fraud_reported'])
y_f = df_f['fraud_reported'].values

# Split Train/Test (80/20) with stratification
X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(
    X_f, y_f, test_size=0.2, random_state=42, stratify=y_f
)

f_cat_cols = ['gender', 'policy_type', 'incident_type', 'incident_severity', 
              'police_report_filed', 'documents_complete', 'claim_channel', 
              'income_bracket', 'claim_location']
f_num_cols = ['customer_age', 'policy_tenure_years', 'annual_premium_inr', 
              'claim_amount_inr', 'days_to_report', 'witnesses', 'num_past_claims_3yrs']

encoder_f = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
X_f_train_cat = encoder_f.fit_transform(X_f_train[f_cat_cols])
X_f_test_cat = encoder_f.transform(X_f_test[f_cat_cols])

scaler_f = StandardScaler()
X_f_train_num = scaler_f.fit_transform(X_f_train[f_num_cols])
X_f_test_num = scaler_f.transform(X_f_test[f_num_cols])

X_f_train_pre = np.hstack([X_f_train_num, X_f_train_cat])
X_f_test_final = np.hstack([X_f_test_num, X_f_test_cat])

# Apply SMOTE to training data ONLY to handle class imbalance
print(f"Before SMOTE - Class distribution: {np.bincount(y_f_train)}")
smote = SMOTE(random_state=42)
X_f_train_smote, y_f_train_smote = smote.fit_resample(X_f_train_pre, y_f_train)
print(f"After SMOTE  - Class distribution: {np.bincount(y_f_train_smote)}")

# Train Classification Models
models_clf = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=120, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1)
}

clf_results = {}
best_clf_name = None
best_f1 = -float('inf')
best_clf_model = None

for name, model in models_clf.items():
    model.fit(X_f_train_smote, y_f_train_smote)
    y_pred = model.predict(X_f_test_final)
    y_prob = model.predict_proba(X_f_test_final)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    acc = accuracy_score(y_f_test, y_pred)
    prec = precision_score(y_f_test, y_pred, zero_division=0)
    rec = recall_score(y_f_test, y_pred, zero_division=0)
    f1 = f1_score(y_f_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_f_test, y_prob)
    cm = confusion_matrix(y_f_test, y_pred).tolist()
    
    clf_results[name] = {
        'Accuracy': round(float(acc), 4),
        'Precision': round(float(prec), 4),
        'Recall': round(float(rec), 4),
        'F1-Score': round(float(f1), 4),
        'ROC-AUC': round(float(auc), 4),
        'Confusion Matrix': cm
    }
    print(f"  {name:20} | Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    if f1 > best_f1:
        best_f1 = f1
        best_clf_name = name
        best_clf_model = model

# Train Module 2: Deep Learning (ANN) Classifier
print("\nTraining ANN Classifier...")
ann_clf = Sequential([
    Input(shape=(X_f_train_smote.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop_clf = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)

history_ann_clf = ann_clf.fit(
    X_f_train_smote, y_f_train_smote,
    validation_split=0.2,
    epochs=30,
    batch_size=64,
    callbacks=[early_stop_clf],
    verbose=0
)

y_prob_ann = ann_clf.predict(X_f_test_final, verbose=0).flatten()
y_pred_ann = (y_prob_ann >= 0.5).astype(int)

ann_acc = accuracy_score(y_f_test, y_pred_ann)
ann_prec = precision_score(y_f_test, y_pred_ann, zero_division=0)
ann_rec = recall_score(y_f_test, y_pred_ann, zero_division=0)
ann_f1 = f1_score(y_f_test, y_pred_ann, zero_division=0)
ann_auc = roc_auc_score(y_f_test, y_prob_ann)

clf_results['ANN (Keras)'] = {
    'Accuracy': round(float(ann_acc), 4),
    'Precision': round(float(ann_prec), 4),
    'Recall': round(float(ann_rec), 4),
    'F1-Score': round(float(ann_f1), 4),
    'ROC-AUC': round(float(ann_auc), 4),
    'Confusion Matrix': confusion_matrix(y_f_test, y_pred_ann).tolist()
}
print(f"  {'ANN (Keras)':20} | Acc: {ann_acc:.4f} | Prec: {ann_prec:.4f} | Rec: {ann_rec:.4f} | F1: {ann_f1:.4f} | AUC: {ann_auc:.4f}")

# Save fraud classification artifacts
joblib.dump(best_clf_model, 'models/fraud_model.pkl')
joblib.dump(scaler_f, 'models/fraud_scaler.pkl')
joblib.dump(encoder_f, 'models/fraud_encoder.pkl')
with open('models/fraud_features.json', 'w') as f:
    json.dump({
        'numeric_cols': f_num_cols,
        'categorical_cols': f_cat_cols,
        'best_model': best_clf_name,
        'metrics': clf_results
    }, f, indent=2)

ann_clf.save('models/ann_fraud_model.keras')
print("Saved models/fraud_model.pkl and models/ann_fraud_model.keras")
metrics_summary['classification'] = clf_results

# ==========================================
# 3. MODULE 4: RNN / LSTM (CUSTOMER REVIEW SENTIMENT)
# ==========================================
print("\n" + "="*50)
print("TRAINING MODULE 4: RNN / LSTM SENTIMENT CLASSIFIER")
print("="*50)

df_rev = pd.read_csv('data/insurance_reviews.csv')
print(f"Loaded reviews dataset: {df_rev.shape}")

X_rev_train, X_rev_test, y_rev_train, y_rev_test = train_test_split(
    df_rev['review_text'].values, df_rev['sentiment'].values,
    test_size=0.2, random_state=42, stratify=df_rev['sentiment'].values
)

vocab_size = 5000
max_len = 100
tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(X_rev_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_rev_train), maxlen=max_len, padding='post', truncating='post')
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_rev_test), maxlen=max_len, padding='post', truncating='post')

# LSTM Model Architecture
lstm_model = Sequential([
    Input(shape=(max_len,)),
    Embedding(vocab_size, 64),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop_lstm = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

history_lstm = lstm_model.fit(
    X_train_seq, y_rev_train,
    validation_split=0.2,
    epochs=15,
    batch_size=32,
    callbacks=[early_stop_lstm],
    verbose=0
)

y_prob_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()
y_pred_lstm = (y_prob_lstm >= 0.5).astype(int)

lstm_acc = accuracy_score(y_rev_test, y_pred_lstm)
lstm_f1 = f1_score(y_rev_test, y_pred_lstm)
print(f"LSTM Test Accuracy: {lstm_acc*100:.2f}% | F1-Score: {lstm_f1:.4f}")

# Sanity check sample sentences
test_sentences = [
    "My claim was settled very fast and the support team was genuinely helpful throughout.",
    "Waited 3 months with no response. Worst customer service ever encountered.",
    "Hospital cashless desk completed approvals within 30 minutes with zero hassle."
]
test_seqs = pad_sequences(tokenizer.texts_to_sequences(test_sentences), maxlen=max_len, padding='post')
preds = lstm_model.predict(test_seqs, verbose=0).flatten()
for sent, p in zip(test_sentences, preds):
    label = "Positive" if p >= 0.5 else "Negative"
    print(f"  Sample: '{sent[:50]}...' -> {label} ({p*100:.1f}%)")

lstm_model.save('models/lstm_sentiment_model.keras')
with open('models/sentiment_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

metrics_summary['lstm_sentiment'] = {
    'Accuracy': round(float(lstm_acc), 4),
    'F1-Score': round(float(lstm_f1), 4)
}
print("Saved models/lstm_sentiment_model.keras and models/sentiment_tokenizer.pkl")

# ==========================================
# 4. MODULE 3: CNN (VEHICLE DAMAGE DETECTION - TRANSFER LEARNING)
# ==========================================
print("\n" + "="*50)
print("TRAINING MODULE 3: CNN VEHICLE DAMAGE CLASSIFIER")
print("="*50)

train_dir = 'data/car_damage/training'
val_dir = 'data/car_damage/validation'

# Use tf.keras.utils.image_dataset_from_directory for fast, modern data pipeline
img_size = (224, 224)
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

print(f"Class names: {train_ds.class_names}")
class_names = train_ds.class_names

# Data augmentation on training data only
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Transfer learning backbone: MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

# Build complete model with Rescaling & Augmentation
inputs = Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = tf.keras.layers.Rescaling(1./127.5, offset=-1)(x)  # MobileNetV2 expects [-1, 1]
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
x = Dense(64, activation='relu')(x)
outputs = Dense(1, activation='sigmoid')(x)

cnn_model = Model(inputs, outputs)
cnn_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

early_stop_cnn = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)

print("Training MobileNetV2 Transfer Learning Model...")
history_cnn = cnn_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop_cnn],
    verbose=1
)

val_loss, val_acc = cnn_model.evaluate(val_ds, verbose=0)
print(f"CNN Validation Accuracy: {val_acc*100:.2f}% (Target: >= 85%)")

cnn_model.save('models/cnn_damage_model.keras')
with open('models/cnn_classes.json', 'w') as f:
    json.dump({'class_names': class_names, 'accuracy': round(float(val_acc), 4)}, f, indent=2)

metrics_summary['cnn_damage'] = {
    'Validation Accuracy': round(float(val_acc), 4),
    'Target Met': bool(val_acc >= 0.85)
}
print("Saved models/cnn_damage_model.keras")

# Save complete metrics summary
with open('models/all_metrics_summary.json', 'w') as f:
    json.dump(metrics_summary, f, indent=2)

print("\n" + "="*50)
print("ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("="*50)
