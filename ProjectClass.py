# Import libraries and modules
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping
import joblib
from tensorflow.keras.optimizers import Adam


class FinalProject():
    """Project class that predicts the genres of movies.

    Recieves a dataframe with the training data and a dataframe with the
    testing data. The dataframe must have the following columns:
    - year: the year of the movie
    - title: the title of the movie
    - plot: the plot of the movie
    - genres: the genres of the movie
    - rating: the rating of the movie

    The class has the following methods:
    - get_plots: returns the plots of the training and testing dataframes
    - vectorize_x: vectorizes the training and testing dataframes' plots
    - vectorize_y: vectorizes the training dataframe's genres column
    - split_data: splits the training dataframe into training and validation
    - model: creates and train the model
    - predict: predicts the genres of the testing dataframe
    - save_predictions: saves the predictions in a csv file

    Attributes:
    - training_df: the training dataframe
    - testing_df: the testing dataframe
    """

    def __init__(
        self,
        training_df: pd.core.frame.DataFrame,
        testing_df: pd.core.frame.DataFrame
    ) -> None:

        self.training_df = training_df
        self.testing_df = testing_df

    def get_plots(
        self,
        df: pd.core.frame.DataFrame
    ) -> np.ndarray:
        """Returns the plots of the dataframe.

        Args:
            df: the dataframe

        Returns:
            The plots of the dataframe as a numpy array
        """

        return df["plot"].values

    def vectorize_x(self) -> None:
        """Vectorizes the training and testing dataframes' plots.

        Uses the CountVectorizer from sklearn to vectorize the plots.

        Args:
            None

        Returns:
            None
        """
        # Get both training and testing plots
        self.train_plots = self.get_plots(self.training_df)
        self.test_plots = self.get_plots(self.testing_df)

        # Fit the vectorizer
        self.vectorizer = CountVectorizer(max_features=5000)
        self.vectorizer.fit(self.train_plots)

        # Get the vectorized matrix
        self.x_train_all = self.vectorizer.transform(self.train_plots)
        self.x_test = self.vectorizer.transform(self.test_plots)

    def vectorize_y(self) -> None:
        """Vectorizes the training dataframe's genres column.

        Uses the MultiLabelBinarizer from sklearn to vectorize the genres.

        Args:
            None

        Returns:
            None
        """
        self.binarizer = MultiLabelBinarizer()
        self.y = self.binarizer.fit_transform(
            self.training_df["genres"].map(lambda x: eval(x))
        )

    def split_data(self) -> None:
        """Splits the training dataframe into training and validation.

        Uses the train_test_split from sklearn to split the training
        dataframe into training and validation.

        Args:
            None

        Returns:
            None
        """
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train_all,
            self.y,
            test_size=0.2,
            random_state=100
        )

    def model(
        self,
        epochs: int,
        patience: int,
        learning_rate: float
    ) -> None:
        """Creates and trains the model.

        Creates a sequential model with the following layers:
        - Input layer with 250 neurons and the shape of the vectorized
          plots and tanh activation
        - Dense layer with 250 neurons and tanh activation
        - Dense layer with 250 neurons and tanh activation
        - Dense layer as output layer with the shape of the vectorized
          genres and sigmoid activation

        Args:
            epochs: the number of epochs
            patience: the number of epochs without improvement before
                the model stops training
            learning_rate: the learning rate of the model

        Returns:
            None
        """
        stop = EarlyStopping(monitor="val_loss", patience=patience)
        adam = Adam(learning_rate=learning_rate)
        input_dim = self.x_train.shape[1]
        out_dim = self.y_train.shape[1]

        self.model = Sequential()
        self.model.add(layers.Dense(
            250,
            input_dim=input_dim,
            activation="tanh")
        )
        self.model.add(layers.Dense(250, activation="tanh"))
        self.model.add(layers.Dense(250, activation="tanh"))
        self.model.add(layers.Dense(out_dim, activation="sigmoid"))
        self.model.compile(
            loss="binary_crossentropy",
            optimizer=adam,
            metrics=["accuracy"]
        )

        self.history = self.model.fit(
                            self.x_train,
                            self.y_train,
                            epochs=epochs,
                            verbose=True,
                            validation_data=(
                                self.x_val,
                                self.y_val
                            ),
                            batch_size=10,
                            callbacks=[stop]
        )

    def predict(self) -> None:
        """ Predicts the genres of the testing dataframe.

        Uses the model to predict the genres of the testing dataframe.

        Args:
            None

        Returns:
            None
        """
        self.predictions = self.model.predict(self.x_test)

    def save_predictions(
        self,
        path: str
    ) -> None:
        """Saves the predictions in a csv file.

        Saves the predictions in a csv file with the following columns:
        - p_Action: the probability of the movie being Action
        - p_Adventure: the probability of the movie being Adventure
        - p_Animation: the probability of the movie being Animation
        - p_Biography: the probability of the movie being Biography
        - p_Comedy: the probability of the movie being Comedy
        - p_Crime: the probability of the movie being Crime
        - p_Documentary: the probability of the movie being Documentary
        - p_Drama: the probability of the movie being Drama
        - p_Family: the probability of the movie being Family
        - p_Fantasy: the probability of the movie being Fantasy
        - p_FilmNoir: the probability of the movie being FilmNoir
        - p_History: the probability of the movie being History
        - p_Horror: the probability of the movie being Horror
        - p_Music: the probability of the movie being Music
        - p_Musical: the probability of the movie being Musical
        - p_Mystery: the probability of the movie being Mystery
        - p_News: the probability of the movie being News
        - p_Romance: the probability of the movie being Romance
        - p_SciFi: the probability of the movie being SciFi
        - p_Sport: the probability of the movie being Sport
        - p_Sport: the probability of the movie being Thriller
        - p_Thriller: the probability of the movie being Thriller
        - p_War: the probability of the movie being War
        - p_Western: the probability of the movie being Western

        Args:
            path: the path to the csv file to save the predictions

        Returns:
            None
        """
        cols = [
            "p_Action", "p_Adventure", "p_Animation",
            "p_Biography", "p_Comedy", "p_Crime",
            "p_Documentary", "p_Drama", "p_Family",
            "p_Fantasy", "p_Film-Noir", "p_History",
            "p_Horror", "p_Music", "p_Musical",
            "p_Mystery", "p_News", "p_Romance",
            "p_Sci-Fi", "p_Short", "p_Sport",
            "p_Thriller", "p_War", "p_Western"]

        res = pd.DataFrame(
            self.predictions,
            index=self.testing_df.index,
            columns=cols
        )
        res.to_csv(path, index_label='ID')

    def save_model(self, path: str) -> None:
        """Saves the model.

        Saves the trained model in a h5 file.

        Args:
            path: the path to the h5 file to save the model

        Returns:
            None
        """
        self.model.save(path)

    def save_vectorizer(self, path: str) -> None:
        """Saves the vectorizer.

        Saves the vectorizer in a pickle file.

        Args:
            path: the path to the pickle file to save the vectorizer

        Returns:
            None
        """
        with open(path, "wb") as f:
            joblib.dump(self.vectorizer, f)

    def run(
        self,
        epochs: int,
        patience: int,
        learning_rate: float,
        predictions_path: str,
        model_path: str,
        vectorizer_path: str
    ) -> None:
        """Runs the entire pipeline.

        Runs each of the functions in the pipeline.
        The functions are:
        - vectorize_x: vectorizes the training dataframe's plots column
        - vectorize_y: vectorizes the training dataframe's genres column
        - split_data: splits the dataframe into training and validation
        - model: creates and trains the model
        - predict: predicts the genres of the testing dataframe
        - save_predictions: saves the predictions in a csv file

        Args:
            epochs: the number of epochs

            patience: the number of epochs without improvement before
                the model stops training

            learning_rate: the learning rate of the model

            predictions_path: the path to the csv file to save the predictions

            model_path: the path to the h5 file to save the model

            vectorizer_path: the path to the pickle file to save the vectorizer

        Returns:
            None
        """
        self.vectorize_x()
        self.vectorize_y()
        self.split_data()
        self.model(epochs, patience, learning_rate)
        self.predict()
        self.save_predictions(predictions_path)
        self.save_model(model_path)
        self.save_vectorizer(vectorizer_path)


if __name__ == "__main__":
    # Get the data from the csv file and store it in a dataframe
    path = "https://github.com/albahnsen/MIAD_ML_and_NLP/raw/main/datasets/"
    training = path + "dataTraining.zip"
    testing = path + "dataTesting.zip"

    data_training = pd.read_csv(training, encoding='UTF-8', index_col=0)
    data_testing = pd.read_csv(testing, encoding='UTF-8', index_col=0)

    # Create the object
    final_project = FinalProject(data_training, data_testing)

    # Run the model
    final_project.run(
        epochs=30,
        patience=5,
        learning_rate=0.00001,
        predictions_path="predictions/predictions.csv",
        model_path="models/model.h5",
        vectorizer_path="vectorizers/vectorizer.pkl"
    )
