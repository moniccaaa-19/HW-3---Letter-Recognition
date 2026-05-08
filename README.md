from ucimlrepo import fetch_ucirepo 
letter_recognition = fetch_ucirepo(id=59) 
  
#data (as pandas dataframes) 
X = letter_recognition.data.features 
y = letter_recognition.data.targets 
  
#metadata 
print(letter_recognition.metadata) 
  
#variable information 
print(letter_recognition.variables) 

# --- Part 1 ---

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Fixed seed for reproducibility
randseed = 17342
np.random.seed(randseed)

#Dataset 
letter_recognition = fetch_ucirepo(id=59) 
X = letter_recognition.data.features 
y = letter_recognition.data.targets.values.ravel()

#1-2. Preprocessing
X = X.dropna() 

#Encode the target
le = LabelEncoder()
y = le.fit_transform(y)

#Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
feature_names = X.columns.tolist()

print(f"Shape: {X_scaled.shape}, Classes: {len(np.unique(y))}")

# --- Part 2 ---

from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.feature_selection import SequentialFeatureSelector

models = {
    'Linear Classifier': Perceptron(max_iter=1000, random_state=RANDOM_STATE),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Gaussian NB': GaussianNB(),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=RANDOM_STATE)
}

rkf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=RANDOM_STATE)

part2_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, cv=rkf, scoring='accuracy', n_jobs=-1)
    part2_results[name] = (scores.mean(), scores.std())
    print(f"{name:20s} Mean Accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}")

# --- Part 3---

part3_results = {}

#Forward Selection (Heuristic Search)
for name, model in models.items():
    sfs = SequentialFeatureSelector(
        model, 
        n_features_to_select='auto', 
        direction='forward', 
        scoring='accuracy', 
        cv=5, 
        n_jobs=-1
    )
    sfs.fit(X_scaled, y)
    
  #Identify selected features
  selected_indices = sfs.get_support()
  X_subset = X_scaled[:, selected_indices]
  selected_features = np.array(feature_names)[selected_indices]
    
  scores = cross_val_score(model, X_subset, y, cv=rkf, scoring='accuracy', n_jobs=-1)
    
  part3_results[name] = {
        'subset': list(selected_features),
        'mean': scores.mean(),
        'std': scores.std()
    }
    
  print(f"Algorithm: {name}")
  print(f"Best Subset: {list(selected_features)}")
  print(f"Mean Accuracy: {scores.mean():.4f}, Std: {scores.std():.4f}\n")
