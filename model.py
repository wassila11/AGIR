import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor



class ModelCreator:
    """Class for Model creation and application.

    The order of execution follows the code logic :
    - create model data to adapt the data to the model
    - train model to create the model and use training data on it
    - use model to produce predictions based on the model.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe

    def create_model_data(
            self,
            column_to_predict: str,
            quantitative_variables: list[str],
            categorical_variables: list[str],
            test_size: float = 0.1,
            random_state: int = 42,  # Add random_state for reproducibility
        ):
        """Private method to transform an input with the scalers.

        Only to use after a first create_model_data execution.
        scores (the two last values returned) are calculated based on the R²
        method. Feel free to document on this formula, but in summary:
        - score = 1 : perfect model (impossible but we want to get as close)
        - score = 0 : useless model (using y mean is better to predict than our model)
        - score < 0 : retry because using the y mean is actually better than the model

        Args:
            column_to_predict (str): The variable you want to predict (electricity consumption)
            quantitative_variables (list[str]): Column names of your quantitative variables (ex: temperature)
            categorical_variables (list[str]): Column names of your categorical variables (ex: region)
            test_size (float): value between 0 and 1 of the proportion of your dataset
                you want to use to test your model on data it didn't train on.
                I recommend staying around 0.1 (0.05 or 0.15 are nice values for example)
            random_state (int): Seed for the random number generator, ensures reproducibility.
        """
        self.quantitative_variables = quantitative_variables
        self.categorical_variables = categorical_variables
        self.test_size = test_size

        # Normalize quantitative data
        quantitative_df = self.dataframe[quantitative_variables]
        self.quantitative_scaler = StandardScaler()
        norm_quantitative_df = pd.DataFrame(
            self.quantitative_scaler.fit_transform(quantitative_df),
            columns=quantitative_df.columns,
        )

        # One-hot encode categorical data
        categorical_df = self.dataframe[categorical_variables]
        self.categorical_scaler = OneHotEncoder(sparse_output=False)
        categorical_intermediate = self.categorical_scaler.fit_transform(categorical_df)
        norm_categorical_df = pd.DataFrame(
            categorical_intermediate,
            columns=[
                str(x) for sublist in self.categorical_scaler.categories_ for x in sublist
            ],
        )

        # Prepare the target variable
        self.y = self.dataframe[column_to_predict].to_numpy()

        # Combine quantitative and categorical data into one dataset
        self.X = pd.concat([norm_quantitative_df, norm_categorical_df], axis=1)

        # Split the data into train and test sets using the random_state for reproducibility
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state  # Pass random_state here
        )

    def _transform_input(self, input: dict):
        """Private method to transform an input with the scalers.

        Only to use after a first create_model_data execution.
        scores (the two last values returned) are calculated based on the R²
        method. Feel free to document on this formula, but in summary:
        - score = 1 : perfect model (impossible but we want to get as close)
        - score = 0 : useless model (using y mean is better to predict than our model)
        - score < 0 : retry because using the y mean is actually better than the model

        Args:
            input (dict): A dict that proposes one value for each variable
                submitted to the initial create_model_data.

        Returns:
            list: the values transformed with the scalers.
        """
        quantitative_input = []
        for var in self.quantitative_variables + self.categorical_variables:
            tmp = input.get(var)
            if tmp is None:
                raise ValueError(f"Missing {var}")
            if not isinstance(tmp, list):
                input[var] = [tmp]

        if len(set([len(v) for v in input.values()])) > 1:
            raise ValueError("Missing values in list")

        quantitative_input = pd.DataFrame(
            {
                key: value
                for key, value in input.items()
                if key in self.quantitative_variables
            }
        )
        quantitative_norm_input = self.quantitative_scaler.transform(
            quantitative_input
        )

        categorical_input = pd.DataFrame(
            {
                key: value
                for key, value in input.items()
                if key in self.categorical_variables
            }
        )
        categorical_norm_input = self.categorical_scaler.transform(
            categorical_input
        )

        return pd.DataFrame(
            np.concatenate(
                (quantitative_norm_input, categorical_norm_input), axis=1
            ),
            columns=self.X.columns,
        )

    def train_model(
        self, network_layers: list[int] = [24, 12, 6], max_iterations: int = 500 ,random_state: int = 1,
    ):
        """Trains the model with passed training parameters.

        Only to use after a first create_model_data execution.
        scores (the two last values returned) are calculated based on the R²
        method. Feel free to document on this formula, but in summary:
        - score = 1 : perfect model (impossible but we want to get as close)
        - score = 0 : useless model (using y mean is better to predict than our model)
        - score < 0 : retry because using the y mean is actually better than the model

        Args:
            network_layers (list[int]): describes the complexity of the model
                bigger the numbers and bigger the list imply harder to train but
                has more potential to yield results.
            max_iterations (int): sets a limit to end training. 500 iterations
                means the dataset will be used 500 times to train the model.

        Returns:
            list[float]: describes how far the model is to "being good" at each
                iteration. In a training, generally converges to a value we hope
                to be as low as possible
            float: the score on the training data
            float: the score on the set data
        """
        self.base_model = MLPRegressor(
            hidden_layer_sizes=network_layers,
            random_state=random_state,
            max_iter=max_iterations,
        )
        self.base_model.fit(self.X_train, self.y_train)
        return (
            self.base_model.loss_curve_,
            self.base_model.score(self.X_train, self.y_train),
            self.base_model.score(self.X_test, self.y_test),
        )

    def use_model(self, input_dict: dict):
        """Method to apply the model.

        Only to use after a first train_model

        Args:
            input_dict (dict): A dict that proposes at least one value for each variable
                submitted to the initial create_model_data.

        Returns
            np.array: The prediction contained in a np.array
        """
        x = self._transform_input(input_dict)
        prediction = self.base_model.predict(x)
        return prediction
