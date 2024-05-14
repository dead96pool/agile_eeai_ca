import numpy as np
import pandas as pd
from model.base import BaseModel
from numpy import *
import random
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

num_folds = 0
seed =0
# Data
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class GNB(BaseModel):
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        super(GNB, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = BinaryRelevance(GaussianNB())
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        self.mdl = self.mdl.fit(data.X_train, data.y_train)

    def predict(self, X_test: pd.Series):
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions

    def print_results(self, data):
        y_predictions = self.predictions.toarray()
        y_test = data.y_test.to_numpy()
        accuracy_type2 = []
        accuracy_type3 = []
        accuracy_type4 = []
        for i in range(len(y_test)):
            if(y_test[i, 0] == y_predictions[i, 0]):
                accuracy_type2_partial = 100
                accuracy_type2.append(accuracy_type2_partial)
                if(y_test[i, 1] == y_predictions[i, 1]):
                    accuracy_type3_partial = 100
                    accuracy_type3.append(accuracy_type3_partial)
                    if(y_test[i, 2] == y_predictions[i, 2]):
                        accuracy_type4_partial = 100
                        accuracy_type4.append(accuracy_type4_partial)
                    else:
                        accuracy_type4_partial = 0
                        accuracy_type4.append(accuracy_type4_partial)
                else:
                    accuracy_type3_partial = 0
                    accuracy_type3.append(accuracy_type3_partial)
                    accuracy_type4_partial = 0
                    accuracy_type4.append(accuracy_type4_partial)
            else:
                accuracy_type2_partial = 0
                accuracy_type2.append(accuracy_type2_partial)
                accuracy_type3_partial = 0
                accuracy_type3.append(accuracy_type3_partial)
                accuracy_type4_partial = 0
                accuracy_type4.append(accuracy_type4_partial)
        accuracy_2 = np.mean(accuracy_type2)
        accuracy_3 = np.mean(accuracy_type3)
        accuracy_4 = np.mean(accuracy_type4)
        print(f'Accuracy on Type2: {accuracy_2:.2f}%')
        print(f'Accuracy on Type2 + Type3: {accuracy_3:.2f}%')
        print(f'Accuracy on Type2 + Type3 + Type4: {accuracy_4:.2f}%')



    def data_transform(self) -> None:
        ...

