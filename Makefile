.PHONY: train app

train:
	python train.py

app:
	python app/gradio_app.py