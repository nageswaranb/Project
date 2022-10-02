# USAGE
# python C:\Users\Nageswaran B\Documents\machine_learning\machine_learning_project\simplilearn\cnn\resnet.py --dataset E:\delete_junk\train\train-stratified --weights C:\Users\Nageswaran B\Documents\machine_learning\machine_learning_project\simplilearn\cnn\weights --output C:\Users\Nageswaran B\Documents\machine_learning\machine_learning_project\simplilearn\cnn\gpu.png

# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from works.preprocessing import ImageToArrayPreprocessor
from works.preprocessing import SimplePreprocessor
from works.datasets import SimpleDatasetLoader
from works.nn.conv import ShallowNet
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.compat.v2.keras.utils import multi_gpu_model
#from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
import os
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from works.nn.conv import ResNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
#ap.add_argument("-m", "--model", required=True,
#	help="path to output model")
ap.add_argument("-w", "--weights", required=True,
	help="path to weights directory")
ap.add_argument("-o", "--output", required=True,
	help="path to output plot")
ap.add_argument("-g", "--gpus", type=int, default=1,
	help="# of GPUs to use for training")    
args = vars(ap.parse_args())

# grab the number of GPUs and store it in a conveience variable
G = args["gpus"]

# grab the list of images that we'll be describing
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessors
sp = SimplePreprocessor(128, 128)
iap = ImageToArrayPreprocessor()

# load the dataset from disk then scale the raw pixel intensities
# to the range [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# construct the image generator for data augmentation
aug = ImageDataGenerator(width_shift_range=0.1,
	height_shift_range=0.1, horizontal_flip=True,
	fill_mode="nearest")

# check to see if we are compiling using just a single GPU
if G <= 1:
    print("[INFO] training with 1 GPU...")
    # initialize the optimizer and model
    print("[INFO] compiling model...")
    #opt = SGD(lr=1e-1)
    model = ResNet.build(128, 128, 3, 5, (9, 9, 9),#change 10 to 62
            (64, 64, 128, 256), reg=0.0005)
            
# otherwise, we are compiling using multiple GPUs
else:
	# disable eager execution
	tf.compat.v1.disable_eager_execution()
	print("[INFO] training with {} GPUs...".format(G))
	# we'll store a copy of the model on *every* GPU and then combine
	# the results from the gradient updates on the CPU
	with tf.device("/cpu:0"):
		# initialize the model
		model = ResNet.build(128, 128, 3, 5, (9, 9, 9), #change 10 to 62
			(64, 64, 128, 256), reg=0.0005)
	
	# make the model parallel
	model = multi_gpu_model(model, gpus=G)

# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# tf.config.list_physical_devices('GPU')

# gpus = tf.config.list_physical_devices('GPU')                 # if GPU then returns info

# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
#   try:
#     tf.config.set_logical_device_configuration(
#                                               gpus[0],
#                                               [tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                                                tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                                                tf.config.LogicalDeviceConfiguration(memory_limit=1024),
#                                                tf.config.LogicalDeviceConfiguration(memory_limit=1024)
#                                               ])
    
#     logical_gpus = tf.config.list_logical_devices('GPU')

#     print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
    
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

# tf.debugging.set_log_device_placement(True)

# gpus     = tf.config.list_logical_devices('GPU')
# strategy = tf.distribute.MirroredStrategy(gpus)

# print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

# with strategy.scope():
# 	model = ResNet.build(512, 512, 3, 5, (9, 9, 9), #change 10 to 62
# 		(64, 64, 128, 256), reg=0.0005)  

# initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])



#added for checkpoint saving
# construct the callback to save only the *best* model to disk
# based on the validation loss
fname = os.path.sep.join([args["weights"],
	"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])
checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
	save_best_only=True, verbose=1)
callbacks = [checkpoint]
#added for checkpoint saving



# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
	batch_size=3, epochs=300, verbose=1, callbacks=callbacks)#32

## grab the history object dictionary
#H = H.history

## save the network to disk
#print("[INFO] serializing network...")
#model.save(args["model"])

# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=3)#32
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=["0", "1", "2","3", "4"]))

# grab the history object dictionary
H = H.history

# plot the training loss and accuracy
N = np.arange(0, len(H["loss"]))
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H["loss"], label="train_loss")
plt.plot(N, H["val_loss"], label="test_loss")
plt.plot(N, H["accuracy"], label="train_acc")
plt.plot(N, H["val_accuracy"], label="test_acc")
plt.title("Resnet on eye")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()

# save the figure
plt.savefig(args["output"])
plt.close()

