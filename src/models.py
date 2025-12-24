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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import joblib
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import optuna
from sklearn.pipeline import Pipeline
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import optuna
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import StackingClassifier

from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import pandas as pd



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


def logistic_regression_optuna(df,target):
    X= df.drop(columns=[target])
    y= df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    col_num=X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col_cat=X.select_dtypes(include=['object', 'category']).columns.tolist()
    preprocessor=  ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), col_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), col_cat)
        ])
    
    def objective(trial):
        C= trial.suggest_float('C', 1e-5, 1e2, log=True)
        solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga'])
        if solver == 'liblinear':
            penalty = trial.suggest_categorical('penalty_lib', ['l1', 'l2'])
        elif solver == 'saga':
            penalty = trial.suggest_categorical('penalty_saga', ['l1', 'l2', 'elasticnet', None])
        else: # lbfgs
            penalty = trial.suggest_categorical('penalty_lbfgs', ['l2', None])
        l1_ratio = None
        if penalty == 'elasticnet':
            l1_ratio = trial.suggest_float('l1_ratio', 0, 1)    
        
        model= LogisticRegression(
            C=C,
            solver=solver,
            penalty=penalty,
            l1_ratio=l1_ratio,
            max_iter=1000,
            random_state=42
        )
        pipe=Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return score

    study= optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial), n_trials=100)
        

    best_model= LogisticRegression(
        C=study.best_params['C'],
        solver=study.best_params['solver'],
        penalty=study.best_params.get('penalty_lib') or study.best_params.get('penalty_saga') or study.best_params.get('penalty_lbfgs'),
        l1_ratio=study.best_params.get('l1_ratio', None),
        max_iter=1000,
        random_state=42 
    )
    best_pipe= Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
    best_pipe.fit(X_train, y_train)

    y_pred= best_pipe.predict(X_test)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    y_proba= best_pipe.predict_proba(X_test)[:, 1]
    roc_auc= roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    print("score in training", best_pipe.score(X_train, y_train))
    print("score in testing", best_pipe.score(X_test, y_test))
    
    return study.best_params

def random_forest_optuna(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    col_num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), col_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), col_cat)
        ])
    
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 5, 500)
        max_depth = trial.suggest_int('max_depth', 3, 30)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
        max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            criterion=criterion,
            random_state=42,
            n_jobs=-1
        )
        
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial), n_trials=20)

    best_model = RandomForestClassifier(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        max_features=study.best_params['max_features'],
        criterion=study.best_params['criterion'],
        random_state=42,
        n_jobs=-1
    )
    
    best_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    print("score in training", best_pipe.score(X_train, y_train))
    print("score in testing", best_pipe.score(X_test, y_test))
    
    return study.best_params   



def svc_optuna(df, target):
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    col_num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), col_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), col_cat)
        ])
    
    def objective(trial):
        C = trial.suggest_float('C', 1e-3, 1e3, log=True)
        kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
        
        model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=True,
            random_state=42
        )
        
        pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
        score = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial), n_trials=50)

    best_model = SVC(
        C=study.best_params['C'],
        kernel=study.best_params['kernel'],
        gamma=study.best_params['gamma'],
        probability=True,
        random_state=42
    )
    
    best_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', best_model)])
    best_pipe.fit(X_train, y_train)

    y_pred = best_pipe.predict(X_test)
    y_proba = best_pipe.predict_proba(X_test)[:, 1]
    
    print("--- SVC OPTIMIZADO ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("score in training", best_pipe.score(X_train, y_train))
    print("score in testing", best_pipe.score(X_test, y_test))
    
    return study.best_params


def create_triple_stacking(df, target, params_lr, params_rf, params_svc):
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
    
    col_num = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    col_cat = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), col_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), col_cat)
        ])

    penalty_lr = params_lr.get('penalty_lib') or params_lr.get('penalty_saga') or params_lr.get('penalty_lbfgs')
    
    model_lr = LogisticRegression(
        C=params_lr['C'],
        solver=params_lr['solver'],
        penalty=penalty_lr,
        l1_ratio=params_lr.get('l1_ratio'),
        max_iter=2000,
        random_state=42
    )
    
    model_rf = RandomForestClassifier(
        n_estimators=params_rf['n_estimators'],
        max_depth=params_rf['max_depth'],
        min_samples_split=params_rf['min_samples_split'],
        min_samples_leaf=params_rf['min_samples_leaf'],
        max_features=params_rf['max_features'],
        criterion=params_rf['criterion'],
        random_state=42,
        n_jobs=-1
    )

    model_svc = SVC(
        C=params_svc['C'],
        kernel=params_svc['kernel'],
        gamma=params_svc['gamma'],
        probability=True,
        random_state=42
    )

    base_models = [
        ('lr', Pipeline(steps=[('preprocessor', preprocessor), ('model', model_lr)])),
        ('rf', Pipeline(steps=[('preprocessor', preprocessor), ('model', model_rf)])),
        ('svc', Pipeline(steps=[('preprocessor', preprocessor), ('model', model_svc)]))
    ]

    stacking_model = StackingClassifier(
        estimators=base_models,
        final_estimator=LogisticRegression(),
        cv=5,
        stack_method='predict_proba'
    )

    stacking_model.fit(X_train, y_train)

    y_pred = stacking_model.predict(X_test)
    y_proba = stacking_model.predict_proba(X_test)[:, 1]

    print("\n--- RESULTADOS DEL ENSAMBLE TRIPLE ---")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"Score in testing: {stacking_model.score(X_test, y_test):.4f}")
    print(f"Score in training: {stacking_model.score(X_train, y_train):.4f}")

    return stacking_model


def plot_ensemble_importance(model, X_test, y_test):
    # Calculamos la importancia por permutación
    results = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42, scoring='roc_auc')
    
    # Organizamos los datos
    feature_importance = pd.Series(results.importances_mean, index=X_test.columns)
    feature_importance = feature_importance.sort_values(ascending=True)

    # Graficamos
    plt.figure(figsize=(10, 6))
    feature_importance.plot(kind='barh', color='skyblue')
    plt.title('Importancia de Variables: Ensamble Triple (0.95 AUC)')
    plt.xlabel('Caída en ROC-AUC al desordenar la variable')
    plt.show()
