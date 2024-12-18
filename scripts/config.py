MODEL_NAME = "vae_mnist_1"
MODEL_DIR = "./models/"

DATASET_NAME = "MNIST"
DATASET_DIR = "../pytorch-datasets/"
SUBSET_SIZE = 8192

BATCH_SIZE = 16
IMG_DEPTH  = 1
IMG_HEIGHT = 64
IMG_WIDTH  = 64

INPUT_SHAPE = [BATCH_SIZE, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH]

EPOCHS        = 2
LATENT_SIZE   = 128
LEARNING_RATE = 1 / 2048
CONV_CHANNELS = [2, 4, 8, 16]