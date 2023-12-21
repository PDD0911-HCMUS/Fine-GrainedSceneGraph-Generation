import os
from DETRController import *
from PIL import Image
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from KerasBaseLine.Config import *
from KerasBaseLine.ModelController import *
from KerasBaseLine.DatasetPreparingController import *

import matplotlib.pyplot as plt
import torch
torch.set_grad_enabled(False)

def decode_and_resize(img_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    print(type(img))
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def NomarlizeImage(img):
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.convert_to_tensor(img)
    #img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

def CreateModel():
    image_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.2),
            layers.RandomContrast(0.3),
        ]
    )   

    cnn_model = get_cnn_model()
    encoder = TransformerEncoderBlock(embed_dim=EMBED_DIM, dense_dim=FF_DIM, num_heads=1)
    decoder = TransformerDecoderBlock(embed_dim=EMBED_DIM, ff_dim=FF_DIM, num_heads=2)
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder,
        image_aug=image_augmentation,
    )

    caption_model.load_weights("KerasBaseLine/SaveModel3/my_model").expect_partial()
    return caption_model

def MainGenerateAttr(imageUrl):
    captions_mapping, text_data = load_captions_data(os.path.join("Datasets/VG/Extract","Attr.token.txt"))

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    caption_model = CreateModel()
    imageName = imageUrl
    print(imageName)
    imageUrl = decode_and_resize(imageUrl)
    img = imageUrl.numpy().clip(0, 255).astype(np.uint8)
    imgReturn = img

    # Pass the image to the CNN
    img = tf.expand_dims(imageUrl, 0)
    img = caption_model.cnn_model(img)

    # Pass the image features to the Transformer encoder
    encoded_img = caption_model.encoder(img, training=False)

    # Generate the caption using the Transformer decoder
    decoded_caption = "<start> "
    for i in range(max_decoded_sentence_length):
        tokenized_caption = vectorization([decoded_caption])[:, :-1]
        mask = tf.math.not_equal(tokenized_caption, 0)
        predictions = caption_model.decoder(
            tokenized_caption, encoded_img, training=False, mask=mask
        )
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]
        if sampled_token == "<end>":
            break
        decoded_caption += " " + sampled_token

    decoded_caption = decoded_caption.replace("<start> ", "")
    decoded_caption = decoded_caption.replace(" <end>", "").strip()
    print("Predicted Caption: ", decoded_caption)

    return imgReturn, decoded_caption, imageName

def LoadAttribute():
    captions_mapping, text_data = load_captions_data(os.path.join("Datasets/VG/Extract","Attr.token.txt"))

    vectorization = TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=SEQ_LENGTH,
        standardize=custom_standardization,
    )
    vectorization.adapt(text_data)

    vocab = vectorization.get_vocabulary()
    index_lookup = dict(zip(range(len(vocab)), vocab))
    max_decoded_sentence_length = SEQ_LENGTH - 1

    return vectorization, index_lookup, max_decoded_sentence_length

def MainGenerateAttr2(image):
    imgReturn = image
    scores, boxes, listImObj = MainDetect(image)
    vectorization, index_lookup, max_decoded_sentence_length = LoadAttribute()
    caption_model = CreateModel()

    for item in listImObj:
        imageUrl = NomarlizeImage(item)
        img = imageUrl.numpy().clip(0, 255).astype(np.uint8)
        #imgReturn = img

        # Pass the image to the CNN
        img = tf.expand_dims(imageUrl, 0)
        img = caption_model.cnn_model(img)

        # Pass the image features to the Transformer encoder
        encoded_img = caption_model.encoder(img, training=False)

        # Generate the caption using the Transformer decoder
        decoded_caption = "<start> "
        for i in range(max_decoded_sentence_length):
            tokenized_caption = vectorization([decoded_caption])[:, :-1]
            mask = tf.math.not_equal(tokenized_caption, 0)
            predictions = caption_model.decoder(
                tokenized_caption, encoded_img, training=False, mask=mask
            )
            sampled_token_index = np.argmax(predictions[0, i, :])
            sampled_token = index_lookup[sampled_token_index]
            if sampled_token == "<end>":
                break
            decoded_caption += " " + sampled_token

        decoded_caption = decoded_caption.replace("<start> ", "")
        decoded_caption = decoded_caption.replace(" <end>", "").strip()
        print("Predicted Caption: ", decoded_caption)

    return imgReturn, decoded_caption

if __name__=="__main__":
    imgUrl = 'Datasets/VG/VG_100K/200.jpg'
    image = Image.open(imgUrl)
    imgReturn, decoded_caption = MainGenerateAttr2(image)
    
    # imgReturn, decoded_caption, imageName = MainGenerateAttr('Datasets/VG/Extract/ImageObj/964_jacket.jpg')
    # plt.title(decoded_caption)
    # plt.imshow(imgReturn)
    # plt.show()
