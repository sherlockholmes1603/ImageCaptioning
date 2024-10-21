# Image Caption With PyTorch

This repository is still under construction.

This repository is created to show how to make neural network using pytorch to generate a caption from an image. The dataset that I use in this repository is Flickr8k and Flikcr30k Image Caption dataset. The model is divided into encoder and decoder to make it more clear to read the code. 
<br/> <br/>

This project demonstrates how to generate captions for images using a neural network built with PyTorch. The model is trained using the Flickr8k and Flickr30k datasets and consists of an encoder-decoder architecture.
<br/> <br/>

<img src="assets/encoder-decoder.png"/>
Image Source : Encoder Decoder Image from Udacity Computer Vision Nanodegree Project
<br/><br/>

The image on the top is the ilustration of the network, but not very similar to what I do in this project, that image is just only the illustration of what the network do. <br/>




# Project Overview

The image captioning model is implemented in a single Jupyter Notebook (image_captioning.ipynb). It includes data loading, model creation, training, and testing steps in one place for simplicity.

# Usage

1. Clone the repository
```bash
git clone https://github.com/your-username/image-caption-pytorch.git
cd image-caption-pytorch
```

2. Download the dataset
Download the Flickr8k or Flickr30k datasets and place them in the data/ directory.
The dataset that I use for this repository can be downloaded from this dataset repository :

Flickr8k Dataset  : https://www.kaggle.com/nunenuh/flickr8k <br/>
Flickr30k Dataset : https://www.kaggle.com/nunenuh/flickr30k

3. Run the notebook
Run the Jupyter Notebook to train and test the model:

```bash
jupyter notebook image_captioning.ipynb
```



