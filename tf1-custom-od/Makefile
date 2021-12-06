IMAGE_NAME=tf1-object-detection-custom
SHELL := /bin/bash

build:
	docker build . -t $(IMAGE_NAME)

run:
	source ./config.env && \
	docker run --rm -ti \
		-v $(PWD)/workspace:/root/workspace \
		-v $(PWD)/scripts:/root/scripts \
		--env-file ./docker.env \
		$(IMAGE_NAME) \
		/bin/bash

train:
	source ./config.env && \
	docker run --rm -ti \
		-v $(PWD)/workspace:/root/workspace \
		-v $(PWD)/scripts:/root/scripts \
		--env-file ./docker.env \
		$(IMAGE_NAME) \
		/root/scripts/train.sh

export:
	source ./config.env && \
	docker run --rm -ti \
		-v $(PWD)/workspace:/root/workspace \
		-v $(PWD)/scripts:/root/scripts \
		--env-file ./docker.env \
		$(IMAGE_NAME) \
		/root/scripts/export_graph.sh

getmodel:
	source ./config.env && ./getmodel.sh

tensorboard:
	tensorboard --logdir ./workspace/training/ssd_mobilenet_v2_coco_2018_03_29

install:
	pip install -r requirements-tensorboard.txt

rsync:
	rsync workspace/export/frozen_inference_graph.pb jetson:/home/lukjestson/Downloads/