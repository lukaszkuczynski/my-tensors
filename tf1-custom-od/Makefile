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

getmodel:
	source ./config.env && ./getmodel.sh