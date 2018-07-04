---
layout: post
title: Automatic Image Captions
---

The following post tries to mimic the following idea by google AI [LINK](https://ai.googleblog.com/2014/11/a-picture-is-worth-thousand-coherent.html)
and also takes a lot of tips from the final project of the Deep Learning course from the Advanced Machine Learning course on Coursera.

The full script for the project is available in a jupyter notebook [here](https://github.com/arroqc/ImageCaptioning/blob/master/Image_captioning.ipynb).

## Idea

This experiment uses the [common objects in context dataset](http://cocodataset.org/#download.) from 2017. We have around 100K images for train and
5K Validation images. For each of those, humans have given some captions (5 captions per images). Here is an example:

![_config.yml]({{ site.baseurl }}/images/caption_example.PNG)  

The task is to make a machine learning algorithm that gets as an input the image and can generate a caption for that image.

## Neural Network Architecture

Obviously, for such an experimental task many advanced approaches are possible. The one I try here is pretty simple:
1. First, use a pretrained ImageNet algorithm to get encodings out of its final layer for an image (transfer learning)
2. Use that encoding as an initial state for a RNN model (here LSTM) that generates the caption

It's a standard encoder-decoder architecture that is used often in seq to seq models. One network encodes an object and the other network
is trained to decode that object.

![_config.yml]({{ site.baseurl }}/images/caption_architecture.PNG)

## CNN Encoder

For the encoder part I use the ImageNet InceptionV3 model available in Keras:

```python
preprocessing = tf.keras.applications.inception_v3.preprocess_input
model = tf.keras.applications.InceptionV3(include_top=False) #I use inception
model = tf.keras.models.Model(model.inputs, tf.keras.layers.GlobalAveragePooling2D()(model.output))
```

Each image out of the Zipfile is also centered, croped and resized to what that model needs (299x299 pixels).

```python
img = bytearray(zf.read(f)) #raw bytes
img = cv2.imdecode(np.array(img),1) #this returns an image in BGR format
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #Convert it to RGB (helps displays)

#The inception model requires square images so we must first crop it to a centered square
h,w,_ = img.shape
if h > w:
	d = h - w #Need to crop the height
	img = img[d-d//2:h-d//2,:,:]
else:
	d = w - h #Need to crop width
	img = img[:,d-d//2:w-d//2,:]

#Need to be resized for inception
img = cv2.resize(img, (299,299)).astype("float32")  # resize for our model
```

Once the images are preprocessed they are predicted in batch by the CNN encoder and we get a 2048 size vector for each images. This is in
essence a learned numeric representation of the image by a neural network that we will feed to another. All this is then stored in a 
pretrained matrix which will be used as inputs for each batch of example during training.

## LSTM Decoder

For the decoder, a similar technique to machine translation is used. The caption is used both as an input and an output. In a single batch,
all captions have to be of the same length so they are padded to match the longest caption with 0s. Finally, a START and END token are added
to each sentences to make the model understand what is the beginning and end of a caption.

When the caption is used as an input, the caption starts at the START token until the last word. The purpose is at each word to predict 
the next word (generative model). Therefore, from the START word and the image the algorithm must try to output the first word of the
caption. With the real first word, the second word etc. As a result the output starts at the first word and ends at the END keyword. The
model is then trained to minimize the error at each prediction step.

## Prediction

The model created is a generative one. We don't have any caption word so from the START token and the image, the model must output the whole
sequence when doing predictions. The way it is done is by starting with the START token and then using each output word as the input of the next step. The loop stops whenever
the algorithm predicts an END token (or reaches the maximum length).

![_config.yml]({{ site.baseurl }}/images/captioning_predict.PNG)

## Examples of predictions

Here are a few examples on the **validation** dataset. Some show great results for this simple technique ! A lot more could be down for both
the CNN and RNN parts. The CNN could be trained for that specific task or even use a more modern version, the RNN should probably
be combined with a language model instead of learning the language only from captions etc.

#### Some good examples
![_config.yml]({{ site.baseurl }}/images/good1.PNG)

![_config.yml]({{ site.baseurl }}/images/good2.PNG)

![_config.yml]({{ site.baseurl }}/images/good3.PNG)

![_config.yml]({{ site.baseurl }}/images/good4.PNG)

![_config.yml]({{ site.baseurl }}/images/good5.PNG)

#### Examples with mistakes
![_config.yml]({{ site.baseurl }}/images/average1.PNG)

![_config.yml]({{ site.baseurl }}/images/average2.PNG)

![_config.yml]({{ site.baseurl }}/images/average3.PNG)

#### Bad Examples
![_config.yml]({{ site.baseurl }}/images/bad1.PNG)

![_config.yml]({{ site.baseurl }}/images/bad2.PNG)

![_config.yml]({{ site.baseurl }}/images/bad3.PNG)