MODEL_NAME = "vae_celeba_1"
MODEL_DIR  = "./models/"

DATASET_NAME = "CelebA"
DATASET_DIR  = "../pytorch-datasets/"
SUBSET_SIZE  = 0 # set to zero for full dataset

EPOCHS        = 4
LEARNING_RATE = 1 / 2048 

MSE_WEIGHT = 1
KLD_WEIGHT = 1 / 32768

BATCH_SIZE = 64
IMG_DEPTH  = 3
IMG_HEIGHT = 64
IMG_WIDTH  = 64

INPUT_SHAPE = [BATCH_SIZE, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH]

CONV_CHANNELS = [2, 4, 8, 16]
FFN_LAYERS = [128, 64, 32]
LATENT_SIZE = 16

# adjust for image depth to get more consistent results beteween datasets
CONV_CHANNELS = [x * IMG_DEPTH for x in CONV_CHANNELS]
FFN_LAYERS = [x * IMG_DEPTH for x in FFN_LAYERS]
LATENT_SIZE = IMG_DEPTH * LATENT_SIZE