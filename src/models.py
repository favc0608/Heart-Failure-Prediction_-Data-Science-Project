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
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred.round()))
    print("Score in Training set:", pipe.score(X_train, y_train))
    print("Score in Test set:", pipe.score(X_test, y_test))
    