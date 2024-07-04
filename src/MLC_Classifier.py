from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
import logging
import joblib


logger = logging.getLogger()
logger.setLevel(logging.INFO)

class Training_Classifier:

    def __init__(self,X, y,metric = 'acc'):
        self.best_model = None
        self.best_score = 0
        self.model_scores = {}
        self.X = X
        self.y = y
        self.metric = metric
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=200),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
            'AdaBoost': AdaBoostClassifier(n_estimators=100),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Decision Tree': DecisionTreeClassifier(),
            'Gaussian Naive Bayes': GaussianNB(),
            'Ridge Classifier': RidgeClassifier(),
            'SGD Classifier': SGDClassifier(max_iter=1000, tol=1e-3),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100),
            'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
            'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
            'MLP Classifier': MLPClassifier(max_iter=300),
            'XGBoost': XGBClassifier(eval_metric='mlogloss'),
            'LightGBM': LGBMClassifier(),
            'Voting Classifier (Hard)': VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=50)),
                ('dt', DecisionTreeClassifier()),
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ], voting='hard')
        }

    def spitter(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def fitter(self,X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)

            if self.metric == 'f1':
                f1_val = f1_score(y_test, y_pred)
                self.model_scores[name] = f1_val
                if f1_val > self.best_score:
                    self.best_score = f1_val
                    self.best_model = model
                    self.down_model = model


            else:
                accuracy = accuracy_score(y_test, y_pred)
                self.model_scores[name] = accuracy
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.down_model = model

    def final_model(self):
        print(f"\nBest Model: {self.best_model}")
        print(f"Best Score: {self.best_score:.4f}")
        joblib.dump(self.down_model, f'Model/{self.best_model}.joblib')
        return self.best_model
        
    def all_models(self):
        print("\nAll Model Scores:")
        for name, score in self.model_scores.items():
            print(f"{name}: {score:.4f}")


    def Run(self):
        X_train, X_test, y_train, y_test = self.spitter()
        self.fitter(X_train, X_test, y_train, y_test)
        self.final_model()


logger.info("Catching CLass")