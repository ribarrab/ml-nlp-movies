# Import libraries
import joblib
from flask import Flask
from flask_restplus import Api, Resource, fields
import pandas as pd
import tensorflow.keras as keras

# Get vectorizer
vectorizer = joblib.load("vectorizer.pkl")

# Get model
model = keras.models.load_model("model.h5")


# Get col names
cols = [
        "p_Action", "p_Adventure", "p_Animation",
        "p_Biography", "p_Comedy", "p_Crime",
        "p_Documentary", "p_Drama", "p_Family",
        "p_Fantasy", "p_Film-Noir", "p_History",
        "p_Horror", "p_Music", "p_Musical",
        "p_Mystery", "p_News", "p_Romance",
        "p_Sci-Fi", "p_Short", "p_Sport",
        "p_Thriller", "p_War", "p_Western"
]

app = Flask(__name__)

# Define API
api = Api(
            app,
            version="1.0",
            title="Movie genres predictor",
            description="Movie genres prediction API"
          )

ns = api.namespace("predict", description="Movie genres Classifier")

# Define API arguments
parser = api.parser()
parser.add_argument(
                    "Plot",
                    type=str,
                    required=True,
                    help="Movie plot",
                    location="args"
)

resource_fields = api.model(
                        "Resource",
                        {"result": fields.String}
)


@ns.route("/")
class MoviesgApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        new_row = [args["Plot"]]
        new_df = pd.DataFrame([new_row])
        new_df.columns = ["Plot"]

        # Vectorize new plots
        new_vectorized = vectorizer.transform(new_df["Plot"])

        # Predict
        result = model.predict(new_vectorized)[0]
        result = {movie: proba for movie, proba in zip(cols, result)}

        return {
         "result": result
        }, 200


app.run(host="0.0.0.0", port=3000)
