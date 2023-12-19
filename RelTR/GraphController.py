import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 
from PIL import Image
import matplotlib.pyplot as plt 
from SourceBaseline.RelTR.models.backbone import Backbone, Joiner
from SourceBaseline.RelTR.models.position_encoding import PositionEmbeddingSine
from SourceBaseline.RelTR.models.transformer import Transformer
from SourceBaseline.RelTR.models.reltr import RelTR
from SourceBaseline.RelTR.Config import CLASSES, REL_CLASSES

position_embedding = PositionEmbeddingSine(128, normalize=True)
backbone = Backbone('resnet50', False, False, False)
backbone = Joiner(backbone, position_embedding)
backbone.num_channels = 2048

transformer = Transformer(d_model=256, dropout=0.1, nhead=8, 
                          dim_feedforward=2048,
                          num_encoder_layers=6,
                          num_decoder_layers=6,
                          normalize_before=False,
                          return_intermediate_dec=True)

# Some transformation functions
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
          (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def LoadImage(imPath):
    # url = 'data/Incidents/incidents_cleaned/test/snow/0D49EE4F095FE124581697B063CF71C31D4C04F0.jpg'
    im = Image.open(imPath)
    #plt.imshow(im)
    img = transform(im).unsqueeze(0)
    return im, img

def ModelCreate():
    model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
              num_entities=100, num_triplets=200)

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.hub.load_state_dict_from_url(
        url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
        map_location='cpu', check_hash=True)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model

def ProcessDemo(imPath, topK):
    model = ModelCreate()
    #eg: imPath = 'data/Incidents/incidents_cleaned/test/snow/0D49EE4F095FE124581697B063CF71C31D4C04F0.jpg'
    im, img = LoadImage(imPath)
    # propagate through the model
    outputs = model(img)

    # keep only predictions with >0.3 confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))


    # convert boxes from [0; 1] to image scales
    sub_bboxes_scaled = rescale_bboxes(outputs['sub_boxes'][0, keep], im.size)
    obj_bboxes_scaled = rescale_bboxes(outputs['obj_boxes'][0, keep], im.size)

    topk = topK # display up to 10 images
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]


    # save the attention weights
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )]


    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        # get the feature map shape
        h, w = conv_features['0'].tensors.shape[-2:]
        im_w, im_h = im.size

        fig, axs = plt.subplots(ncols=len(indices), nrows=3, figsize=(16, 9))
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in \
                zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            ax = ax_i[0]
            ax.imshow(dec_attn_weights_sub[0, idx].view(h, w))
            ax.axis('off')
            ax.set_title(f'query id: {idx.item()}')
            ax = ax_i[1]
            ax.imshow(dec_attn_weights_obj[0, idx].view(h, w))
            ax.axis('off')
            ax = ax_i[2]
            ax.imshow(im)
            ax.add_patch(plt.Rectangle((sxmin, symin), sxmax - sxmin, symax - symin,
                                        fill=False, color='blue', linewidth=2.5))
            ax.add_patch(plt.Rectangle((oxmin, oymin), oxmax - oxmin, oymax - oymin,
                                        fill=False, color='orange', linewidth=2.5))

            ax.axis('off')
            ax.set_title(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()], fontsize=7)

        fig.tight_layout()
        plt.show() # show the output


def Process(imPath, topK):
    resSG = []
    model = ModelCreate()
    im, img = LoadImage(imPath)
    # propagate through the model
    outputs = model(img)

    # keep only predictions with >0.3 confidence
    probas = outputs['rel_logits'].softmax(-1)[0, :, :-1]
    probas_sub = outputs['sub_logits'].softmax(-1)[0, :, :-1]
    probas_obj = outputs['obj_logits'].softmax(-1)[0, :, :-1]
    keep = torch.logical_and(probas.max(-1).values > 0.3, torch.logical_and(probas_sub.max(-1).values > 0.3,
                                                                            probas_obj.max(-1).values > 0.3))

    topk = topK # display up to 10 images
    keep_queries = torch.nonzero(keep, as_tuple=True)[0]
    indices = torch.argsort(-probas[keep_queries].max(-1)[0] * probas_sub[keep_queries].max(-1)[0] * probas_obj[keep_queries].max(-1)[0])[:topk]
    keep_queries = keep_queries[indices]


    # save the attention weights
    conv_features, dec_attn_weights_sub, dec_attn_weights_obj = [], [], []
    hooks = [
        model.backbone[-2].register_forward_hook(
            lambda self, input, output: conv_features.append(output)
        ),
        model.transformer.decoder.layers[-1].cross_attn_sub.register_forward_hook(
            lambda self, input, output: dec_attn_weights_sub.append(output[1])
        ),
        model.transformer.decoder.layers[-1].cross_attn_obj.register_forward_hook(
            lambda self, input, output: dec_attn_weights_obj.append(output[1])
        )]

    with torch.no_grad():
        # propagate through the model
        outputs = model(img)

        for hook in hooks:
            hook.remove()

        # don't need the list anymore
        conv_features = conv_features[0]
        dec_attn_weights_sub = dec_attn_weights_sub[0]
        dec_attn_weights_obj = dec_attn_weights_obj[0]

        for idx in zip(keep_queries):
            resSG.append(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()])
            #print(CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()])
    return resSG
# ProcessDemo('Datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg')
# Process('Datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg')