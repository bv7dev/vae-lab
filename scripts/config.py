MODEL_NAME = "vae_mnist_2"
MODEL_DIR = "./models/"

DATASET_NAME = "MNIST"
DATASET_DIR = "../pytorch-datasets/"
SUBSET_SIZE = 0 # set to zero for full dataset

BATCH_SIZE = 16
IMG_DEPTH  = 1
IMG_HEIGHT = 64
IMG_WIDTH  = 64

INPUT_SHAPE = [BATCH_SIZE, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH]

CONV_CHANNELS = [2, 4, 8, 16]


EPOCHS        = 2
LATENT_SIZE   = 128
LEARNING_RATE = 1 / 2048
