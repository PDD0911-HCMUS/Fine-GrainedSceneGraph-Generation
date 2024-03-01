from KerasBaseLine.Config import *
from KerasBaseLine.ModelController import *
from KerasBaseLine.DatasetPreparingController import *
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
import os

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def process_input(img_path, captions):
    return decode_and_resize(img_path), vectorization(captions)


def make_dataset(images, captions):
    dataset = tf.data.Dataset.from_tensor_slices((images, captions))
    dataset = dataset.shuffle(BATCH_SIZE * 8)
    dataset = dataset.map(process_input, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)

    return dataset


# Load the dataset
#captions_mapping, text_data = load_captions_data(os.path.join("Datasets/Flicker8k","Flickr8k.token.txt"))
# captions_mapping, text_data = load_captions_data(os.path.join("Datasets/VG/Extract","Attr.token.txt"))
captions_mapping, text_data = load_captions_data(os.path.join("Datasets/VG/Extract","Bbox.txt"))

# Split the dataset into training and validation sets
train_data, valid_data = train_val_split(captions_mapping)
print("Number of training samples: ", len(train_data))
print("Number of validation samples: ", len(valid_data))

vectorization = TextVectorization(
    max_tokens=VOCAB_SIZE,
    output_mode="int",
    output_sequence_length=SEQ_LENGTH,
    standardize=custom_standardization,
)
vectorization.adapt(text_data)

# Data augmentation for image data
image_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.2),
        layers.RandomContrast(0.3),
    ]
)   

# Pass the list of images and the list of corresponding captions
train_dataset = make_dataset(list(train_data.keys()), list(train_data.values()))

valid_dataset = make_dataset(list(valid_data.keys()), list(valid_data.values()))

cnn_model = get_cnn_model()
encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
caption_model = ImageCaptioningModel(
    cnn_model=cnn_model,
    encoder=encoder,
    decoder=decoder,
    image_aug=image_augmentation,
)

# Define the loss function
cross_entropy = keras.losses.SparseCategoricalCrossentropy()

# EarlyStopping criteria
early_stopping = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)


# Learning Rate Scheduler for the optimizer
# class LRSchedule(keras.optimizers.schedules.LearningRateSchedule):
#     def __init__(self, post_warmup_learning_rate, warmup_steps):
#         super().__init__()
#         self.post_warmup_learning_rate = post_warmup_learning_rate
#         self.warmup_steps = warmup_steps

#     def __call__(self, step):
#         global_step = tf.cast(step, tf.float32)
#         warmup_steps = tf.cast(self.warmup_steps, tf.float32)
#         warmup_progress = global_step / warmup_steps
#         warmup_learning_rate = self.post_warmup_learning_rate * warmup_progress
#         return tf.cond(
#             global_step < warmup_steps,
#             lambda: warmup_learning_rate,
#             lambda: self.post_warmup_learning_rate,
#         )
    
# Create a learning rate schedule
num_train_steps = len(train_dataset) * EPOCHS
num_warmup_steps = num_train_steps // 15
# lr_schedule = LRSchedule(post_warmup_learning_rate=1e-4, warmup_steps=num_warmup_steps)

lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=7, mode='min', verbose=1)

# Compile the model
caption_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=cross_entropy)
# Fit the model
caption_model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    callbacks=[lr_schedule
               #, early_stopping
               ],
)

caption_model.save_weights("KerasBaseLine/SaveModel4/my_model",save_format='tf')