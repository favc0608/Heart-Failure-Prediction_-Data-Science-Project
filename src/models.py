from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression # Cambiado por clasificación
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
def baseline(df, target_column):
    X= df.drop(columns=[target_column])
    y= df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    num_cols= X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols= X.select_dtypes(include=['object', 'bool']).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('LogisticRegression', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_probs)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    print("\n--- Informe de Clasificación (Test Set) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {auc:.4f}")
    
def randomforest_basemodel(df, target):
    # 1. Dividir los datos en características (X) y objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Preprocesamiento: Escalado y codificación
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # 4. Crear el pipeline con RandomForestClassifier
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # 5. Entrenar el modelo
    pipe.fit(X_train, y_train)
    
    # 6. Hacer predicciones
    y_pred = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    
    # 7. Evaluar el modelo
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    print("\n--- Informe de Clasificación (Test Set) ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

def svc_basemodel(df, target):
    # 1. Dividir los datos en características (X) y objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Preprocesamiento: Escalado y codificación
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # 4. Crear el pipeline con SVC
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', SVC(probability=True, random_state=42))
    ])
    
    # 5. Entrenar el modelo
    pipe.fit(X_train, y_train)
    
    # 6. Hacer predicciones
    y_pred = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    
    # 7. Evaluar el modelo
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    print("\n--- Informe de Clasificación (Test Set) ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

def decisiontree_basemodel(df, target):
    # 1. Dividir los datos en características (X) y objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Preprocesamiento: Escalado y codificación
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # 4. Crear el pipeline con DecisionTreeClassifier
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])
    
    # 5. Entrenar el modelo
    pipe.fit(X_train, y_train)
    
    # 6. Hacer predicciones
    y_pred = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    
    # 7. Evaluar el modelo
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    print("\n--- Informe de Clasificación (Test Set) ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC Score: {auc:.4f}")

def xgboost_basemodel(df, target):
    import xgboost as xgb
    # 1. Dividir los datos en características (X) y objetivo (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    # 2. Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Preprocesamiento: Escalado y codificación
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', 'passthrough', numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    # 4. Crear el pipeline con XGBClassifier
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb.XGBClassifier( eval_metric='logloss', random_state=42))
    ])
    
    pipe.fit(X_train, y_train)
        # 6. Hacer predicciones
    y_pred = pipe.predict(X_test)
    y_probs = pipe.predict_proba(X_test)[:, 1]
    
    # 7. Evaluar el modelo
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    print("\n--- Informe de Clasificación (Test Set) ---")
    print(classification_report(y_test, y_pred))
    
    auc = roc_auc_score(y_test, y_probs)
    print(f"ROC-AUC Score: {auc:.4f}")