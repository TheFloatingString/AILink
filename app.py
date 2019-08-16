from urllib.parse import unquote
from flask import Flask
from vectorizer import vectorize_sentence
import numpy as np
from keras.models import load_model
import tensorflow as tf
graph = tf.get_default_graph()

# from keras import backend as K
# K.set_image_dim_ordering('th')

app = Flask(__name__)

model = load_model("static/feedforward_rt_sent.h5")


@app.route("/compute/<input_string>", methods=["GET"])
def compute(input_string):
	global graph
	input_string = unquote(input_string)
	X = vectorize_sentence(input_string)
	# input_array = input_array.reshape(1,2555,1)
	X = np.expand_dims(X, axis=2)

	with graph.as_default():
		return dict(zip([0,1,2,3,4],model.predict(X)[0].tolist()))

if __name__ == '__main__':
	app.run()