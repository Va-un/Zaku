import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, PassiveAggressiveRegressor, BayesianRidge, TweedieRegressor, OrthogonalMatchingPursuit, TheilSenRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib


# Creating an object
logger = logging.getLogger()

logger.setLevel(logging.INFO)
class Training_Regressor:
    def __init__(self, X, y, metric='acc'):
        self.best_model = None
        self.best_score = 0
        self.model_scores = {}
        self.X = X
        self.y = y
        self.metric = metric
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'Lasso Regression': Lasso(),
            'Decision Tree Regressor': DecisionTreeRegressor(),
            'Random Forest Regressor': RandomForestRegressor(n_estimators=100),
            'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=100),
            'XGBoost Regressor': XGBRegressor(eval_metric='rmse'),
            'LightGBM Regressor': LGBMRegressor(),
            'ElasticNet Regression': ElasticNet(),
            'Support Vector Regressor': SVR(),
            'K-Nearest Neighbors Regressor': KNeighborsRegressor(),
            'Huber Regressor': HuberRegressor(),
            'AdaBoost Regressor': AdaBoostRegressor(),
            'Bagging Regressor': BaggingRegressor(),
            'Extra Trees Regressor': ExtraTreesRegressor(),
            'HistGradientBoosting Regressor': HistGradientBoostingRegressor(),
            'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
            'Passive Aggressive Regressor': PassiveAggressiveRegressor(),
            'Theil-Sen Regressor': TheilSenRegressor(),
            'Tweedie Regressor': TweedieRegressor(),
            'Bayesian Ridge Regressor': BayesianRidge(),
        }

    def spitter(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    def fitter(self, X_train, X_test, y_train, y_test):
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            # Train the model
            model.fit(X_train, y_train)

            # Predict on test set
            y_pred = model.predict(X_test)

            if self.metric == 'mse':
                f1_val = mean_squared_error(y_test, y_pred)
                self.model_scores[name] = f1_val
                if f1_val > self.best_score:
                    self.best_score = f1_val
                    self.best_model = model
                    self.down_model = model


            else:
                accuracy = r2_score(y_test, y_pred)
                self.model_scores[name] = accuracy
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.best_model = model
                    self.down_model = model

    def final_model(self):
        print(f"\nBest Model: {self.best_model}")
        print(f"Best Score: {self.best_score:.4f}")
        joblib.dump( self.down_model, f'Model/{self.best_model}.joblib')
        return self.best_model

    def all_models(self):
        print("\nAll Model Scores:")
        for name, score in self.model_scores.items():
            print(f"{name}: {score:.4f}")

    def Run(self):
        X_train, X_test, y_train, y_test = self.spitter()
        self.fitter(X_train, X_test, y_train, y_test)
        self.final_model()






