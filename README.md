# AILink

## Quickstart

Go to `http://ailinkdev.herokuapp.com/compute/this is a sample text`

## Running Flask API 

```bash
pip3 install requirements.txt
python3 app.py
```

Then, enter in browser the following URL:
`http://localhost:5000/compute/<STRING HERE>`
which will return a JSON of probabilities of classes 0 (terrible) to 4 (excellent)
`{"0":0.1369616836309433,"1":0.2723471224308014,"2":0.4098854064941406,"3":0.0,"4":0.0}`

## Running Machine Learning Model

```bash
pip3 install requirements.txt
python3 src/train_cnn.py
```

## Running data visualization

```bash
pip3 install requirements.txt
python3 src/plot_data.py
```
