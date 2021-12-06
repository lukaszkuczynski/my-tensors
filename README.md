The idea for this repo is to create a reproducible environment for Tensorflow 1 Object detection custom training.

## Step 1 . Collect images
take pictures
resize them to some normal size easily uploaded like max 2k px width

## 2. Create rectangles / object boundaries
I am using makesense.ai and with that to produce YOLO annotations

with XMLs ready that should look like the following:

```xml
<annotation>
	<folder>oniongarlic</folder>
	<filename>IMG_0148.jpg</filename>
	<path>/oniongarlic/IMG_0148.jpg</path>
	<source>
		<database>Unspecified</database>
	</source>
	<size>
		<width>1778</width>
		<height>1000</height>
		<depth>3</depth>
	</size>
	<object>
		<name>onion</name>
		<pose>Unspecified</pose>
		<truncated>Unspecified</truncated>
		<difficult>Unspecified</difficult>
		<bndbox>
			<xmin>950</xmin>
			<ymin>440</ymin>
			<xmax>1111</xmax>
			<ymax>617</ymax>
		</bndbox>
	</object>
</annotation>
```
copy files to the custom folder in `workspace/images`
XMLs and images should be put together

```bash
luk@luk-ThinkPad-T490s:~/othprj/my-tensors/tf1-custom-od/workspace/images/oniongarlic$ ls | head -n4
IMG_0047.jpg
IMG_0047.xml
IMG_0048.jpg
IMG_0048.xml
```
## Train
To prepare images we will use Docker image prepared here.
Change `docker.env` property `IMAGE_PROJECT_NAME` to your project name from `makesense.ai`

Adjust your labels.pbtxt file and put your labels there

