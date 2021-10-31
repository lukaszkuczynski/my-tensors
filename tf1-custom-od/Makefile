IMAGE_NAME=tf1-object-detection-custom
SHELL := /bin/bash

build:
	docker build . -t $(IMAGE_NAME)

run:
	docker run --rm -ti $(IMAGE_NAME) /bin/bash

getmodel:
	source ./config.env && ./getmodel.sh