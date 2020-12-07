# DS-Project4

Udacity Data Science Nanodegree Project 4 - Dog Breed Classifier Project


### 1. Project Summary
The motivation of this project is to experiment with deep learning in image recognition, specifically by building a convolutional neural network that can identify a dog's breed from an image. The notebook begins with the steps to create a human face detector and dog face detector. 

This is followed by the development of a convolutional neural network from scratch to predict a dog's breed from an image. Next is a second CNN developed using transfer learning in order to get an accuracy of at least 60% for the classifier.

The final step is to bring together the human and dog face detectors with the dog breed classifier to build an algorithm where the user can input an image and the closest dog breed is returned if the image contains a human or a dog.


### 2. Libraries Required
In order to run the project you will need to install the libraries and their corresponding minimum versions outlined in the file requirements/requirements.txt
To run the project on a GPU the requirements are within requirements/requirements-gpu.txt


### 3. Project Files/Folders:
* Dog App - the notebook containing the each step to develop and train the dog breed detector in ipynb and html format
* Requirements - the yml files for setting up a virtual conda env in linux/mac/windows along with gpu capabilities for each. 2 txt files containg the required libraries as above
* Haarcascades - OpenCV's implementation of Haar feature-based cascade classifiers in xml format used for human face detection
* Bottleneck Features - npz files containing the weights for the pre-trained CNN that we use (only ResNet50 is stored - others can be downloaded)
* Saved Models - hdf5 files containing the best weights for the models being trained in the dog_app notebook


### 4. Running Instructions:
Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`. The ResNet-50 bottleneck features are already in this directory

(Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

(Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

(Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
(Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

(Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

Switch to the dog-project directory and open the notebook.
```
jupyter notebook dog_app.ipynb
```

### 5. Results and Analysis
The first attempt at building a CNN yielded an accuracy of 3%. This was trained over 5 epochs in less than 2 minutes and has three convolutional layers, three max pooling layers and one global average pooling layer. The optimizer used was RMSprop, the loss function is categorical crossentropy and the only measured metric is accuracy.

For the case of transfer learning using the ResNet-50 model, we achieve an accuracy of 80% after training the model for 10 epochs. The original weights for the ResNet-50 bottleneck features were maintained and the model was trained with an additional global average pooling layer before the fully connected output layer.

The majority of dogs tested with the algorithm were identified correctly however there are limitations when using the the algorithm to identify the breeds of puppies and when submitting other animals such as cats as these can be mistaken for human faces.

### 6. Acknowledgements
Thank you to Udacity along with their suppliers and contributors for providing the baseline code for the notebook and the images used in the development of the dog breed classifier.

Also thank you to my pals Lucia and Shaylen for providing selfies to test the classifier on human faces along with Frank the pug and Coco the cavalier for being wonderful models too. Finally to my cat Siri for also reluctantly modelling for this notebook.
