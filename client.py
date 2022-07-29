import warnings
import flwr as fl
import numpy as np
import sys

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import utils

if __name__ == "__main__":
    # Load Rainfall data
    (X_train, y_train), (X_test, y_test) = utils.load_data()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(1)
    (X_train, y_train) = utils.partition(X_train, y_train, 1)[partition_id]

    # Create LinearRegression Model
    model = LinearRegression()
  
    # Setting initial parameters, akin to model.compile for keras models
    utils.set_initial_params(model)

    # Define Flower client
    class MnistClient(fl.client.NumPyClient):
        def get_parameters(self):  # type: ignore
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            # Ignore convergence failure due to low local epochs
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            print(f"Training finished for round {config['rnd']}")
            return utils.get_model_parameters(model), len(X_train), {}

        def evaluate(self, parameters, config):  # type: ignore
            utils.set_model_params(model, parameters)
            preds = model.predict(X_test)
            loss = mean_squared_error(y_test, preds)
            accuracy = r2_score(preds, y_test)
            return loss, len(X_test), {"accuracy": accuracy}

    # Start Flower client
    fl.client.start_numpy_client(
        #server_address = "localhost:"+ str(sys.argv[1]), 
        server_address = "localhost:5040",
        client=MnistClient())