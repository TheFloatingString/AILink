import matplotlib.pyplot as plt
import json 

with open('data/training_hist.json', 'r') as json_file:
	data = json.load(json_file)

	print(data)

	plt.title("Convolutional Neural Network Training Accuracies")
	plt.plot(data["acc"], label="training data")
	plt.plot(data["val_acc"], label="testing data")
	plt.xlabel("Epochs")
	plt.ylabel("Accuarcy (%)")
	plt.legend()
	plt.savefig("data/accuracy_performance.png")

	plt.close()

	plt.title("Convolutional Neural Network Training Loss")
	plt.plot(data["loss"], label="training data")
	plt.plot(data["val_loss"], label="testing data")
	plt.xlabel("Epochs")
	plt.ylabel("Loss (mean squared error)")
	plt.legend()
	plt.savefig("data/loss_performance.png")