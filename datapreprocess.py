import pandas as pd

def preprocess(df_in):
    df = df_in.copy()
    drop_this = ["institute","education","race","gender","native-country"]
    df = df.drop(drop_this, axis = 1)
    df.loc[df["relationship"].isin(["Husband", "Wife"]), "relationship"] = "Married"
    df = pd.get_dummies(df, columns=["workclass", "marital-status", "occupation", "relationship"])
    return df



#daniel
# def split_data(df):
#     # Drop "race" and "gender" columns
#     df_processed = df.drop(['race', 'gender'], axis=1)
    
#     # Encode categorical variables
#     label_encoders = {}
#     for column in df_processed.select_dtypes(include=['object']).columns:
#         label_encoders[column] = LabelEncoder()
#         df_processed[column] = label_encoders[column].fit_transform(df_processed[column])
    
#     # Split data into features and target variable
#     X = df_processed.drop('income', axis=1)
#     y = df_processed['income']
    
#     # Split data into training and testing sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Scale features using MinMaxScaler
#     scaler = MinMaxScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     return X_train_scaled, X_test_scaled, y_train, y_test