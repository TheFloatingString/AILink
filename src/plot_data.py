import matplotlib.pyplot as plt
import json 

with open('data/training_hist.json', 'r') as json_file:
	data = json.load(json_file)

	print(data)

	plt.plot(data["acc"])
	plt.plot(data["val_acc"])
	plt.savefig("data/loss_performance.png")