import logging
from sklearn.preprocessing import StandardScaler,LabelEncoder
import pandas as pd


logging.basicConfig(filename="logs/Preporcessing.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')

# Creating an object
logger = logging.getLogger()

logger.setLevel(logging.INFO)

class DataCleaner:
    def __init__(self, X):
        self.Dropping = []
        self.Filling = []
        self.X = X

    def null_identifier(self):
        """
        Classifying the null values as  Dropping and Filling
        """
        for i in self.X.columns:
            bool_series = pd.isnull(self.X[i])
            missing_values_percent = (bool_series.sum() / self.X.shape[0]) * 100
            logger.info(f"Count of missing values in the {i} column:")
            logger.info(missing_values_percent)
            if missing_values_percent > 40:
                self.Dropping.append(i)
                logger.info('Adding to Dropping')

            else:
                self.Filling.append(i)
                logger.info('Adding to Filling')

    def null_dropper(self):
        """ 
        Dropping the null values
        """
        for col in self.Dropping:
            self.X.loc[:, col] = self.X[col].dropna()

    def null_filler(self):
        """
        Filling the null values
        """
        for col in self.Filling:
            self.X.loc[:, col] = self.X[col].ffill()

    def validator(self):
        """
        Confirming all the null values are filled are not
        """
        for i in self.X.columns:
            bool_series = pd.isnull(self.X[i])
            missing_values_percent = (bool_series.sum() / self.X.shape[0]) * 100
            logger.info(f"Count of missing values in the {i} column:")
            logger.info( missing_values_percent)
            if missing_values_percent > 0:
                logger.exception(f"VALUE IN {i} IS NOT NULL")
                break

    def Run(self) -> pd.DataFrame:
        try:
            self.null_identifier()
            self.null_dropper()
            self.null_filler()
            self.validator()
            return self.X
        except Exception as e:
            logger.exception("Error in DataCleaner", e)


class DataPreprocessor:
    def __init__(self, X):
        self.X = X
        self.cols = self.X.columns
        self.scaled_columns = []
        self.label_columns = []

    def hot_encoder(self, i):
        """
        converting categorical variables into a binary matrix representation
        """
        self.X = pd.get_dummies(self.X, columns=self.label_columns)

    def num_scaler(self, col):
        """
        Scaling the numerical variables
        """
        scaler = StandardScaler()
        self.X.loc[:, col] = scaler.fit_transform(self.X[col])

    def Seperator(self) -> pd.DataFrame:
        """"
        separating and fitting the data
        """
        # Process numerical columns
        for i in self.X.select_dtypes(include=['int64', 'float64']):
            self.X[i] = self.X[i].astype('float64')
            if self.X[i].nunique() < 5:
                self.label_columns.append(i)

            else:
                self.scaled_columns.append(i)

        # Process categorical columns
        for i in self.X.select_dtypes(include=['object']):
            self.label_columns.append(i)

        logger.info("Columns identified for label encoding:")
        logger.info(self.label_columns)

        logger.info("Columns identified for scaling:")
        logger.info(self.scaled_columns)
        try:
            self.hot_encoder(self.label_columns)
            self.num_scaler(self.scaled_columns)
            return self.X
        except Exception as e:
            logger.exception("error in DataPreprocessor:", e)



def label_encodeing(labels):
    Labeler = LabelEncoder()
    encoded_labels = Labeler.fit_transform(labels)
    return encoded_labels , Labeler

def label_decoding(encoded_labels,le):
    original_labels = le.inverse_transform(encoded_labels)
    return original_labels


def Preprocessing(df, label) -> [pd.DataFrame, pd.DataFrame]:
    df = df.dropna(subset=[label])
    y = df.pop(label)
    y, Labeler = label_encodeing(y)
    X = df
    cleaner = DataCleaner(X=X)
    clean_X = cleaner.Run()
    prep = DataPreprocessor(X=clean_X)
    prep_X = prep.Seperator()
    return prep_X, y,Labeler



