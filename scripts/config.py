MODEL_NAME = "vae-celeba-64x64"
MODEL_DIR  = "./models/"

DATASET_NAME = "CelebA"
DATASET_DIR  = "../pytorch-datasets/"

# set to zero for full dataset
SUBSET_SIZE = 16384

EPOCHS        = 4
LEARNING_RATE = 1e-4

SCHEDULER_STEP = 1
SCHEDULER_GAMMA = 0.9375

MSE_WEIGHT = 1
KLD_WEIGHT = 1 / 65536

BATCH_SIZE = 64
IMG_DEPTH  = 3
IMG_HEIGHT = 64
IMG_WIDTH  = 64

CONV_CHANNELS = [3, 9, 27, 81]
FFN_LAYERS = [1024, 768]
LATENT_SIZE = 512

# pytorch standard image batch tensor shape
INPUT_SIZE = [BATCH_SIZE, IMG_DEPTH, IMG_HEIGHT, IMG_WIDTH]

# adjust for image depth to get more consistent results between datasets
CONV_CHANNELS = [x * IMG_DEPTH for x in CONV_CHANNELS]
FFN_LAYERS = [x * IMG_DEPTH for x in FFN_LAYERS]
LATENT_SIZE = IMG_DEPTH * LATENT_SIZE
