import tensorflow as tf
# Path to the images
#IMAGES_PATH = "Datasets/Flicker8k/Flicker8k_Dataset"
#IMAGES_PATH = "Datasets/VG/Extract/ImageObj"
IMAGES_PATH = "Datasets/VG/Extract/Images"

# Desired image dimensions
#IMAGE_SIZE = (128, 128)

IMAGE_SIZE = (224, 224)

# Vocabulary size
VOCAB_SIZE = 10000

# Fixed length allowed for any sequence
SEQ_LENGTH = 125

# Dimension for the image embeddings and token embeddings
EMBED_DIM = 512

# Per-layer units in the feed-forward network
FF_DIM = 512

# Other training parameters
BATCH_SIZE = 64
EPOCHS = 30
AUTOTUNE = tf.data.AUTOTUNE

strip_chars = "!\"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"
strip_chars = strip_chars.replace("<", "")
strip_chars = strip_chars.replace(">", "")