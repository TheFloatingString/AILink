import joblib

vectorizer = joblib.load('../static/count_vectorizer.joblib')

def vectorize_sentence(str_sentence):
	if type(str_sentence) != list:
		str_sentence = [str_sentence]
	return vectorizer.transform(str_sentence).toarray()
