from AttrGenController import GenerateAttr
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T 
import pandas as pd

from PIL import Image
import requests
import matplotlib.pyplot as plt 

from RelTR.models.backbone import Backbone, Joiner
from RelTR.models.position_encoding import PositionEmbeddingSine
from RelTR.models.transformer import Transformer
from RelTR.models.reltr import RelTR
import io

CLASSES = [ 'N/A', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike',
                'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
                'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup',
                'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy',
                'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
                'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
                'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post',
                'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt',
                'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow',
                'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel',
                'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
                'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']

REL_CLASSES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind',
                'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for',
                'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on',
                'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over',
                'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
                'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

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

def CreateModelReltr():

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

    model = RelTR(backbone, transformer, num_classes=151, num_rel_classes = 51,
                num_entities=100, num_triplets=200)

    # The checkpoint is pretrained on Visual Genome
    ckpt = torch.hub.load_state_dict_from_url(
        url='https://cloud.tnt.uni-hannover.de/index.php/s/PB8xTKspKZF7fyK/download/checkpoint0149.pth',
        map_location='cpu', check_hash=True)
    model.load_state_dict(ckpt['model'])

    return model

def MainProcess(imageUrl, topk = 15):

    lstNewGraph = []
    lstOrgGraph = []

    model = CreateModelReltr()
    model.eval()

    # Some transformation functions
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #imageUrl = 'Datasets/VG/VG_100K/150.jpg'
    im = Image.open(imageUrl)
    # plt.imshow(im)
    imageReturn = im
    img = transform(im).unsqueeze(0)

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
    sub_bboxes = []
    obj_bboxes = []

    #topk = 15 # display up to 10 images
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
        for idx, ax_i, (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in zip(keep_queries, axs.T, sub_bboxes_scaled[indices], obj_bboxes_scaled[indices]):
            sub_bboxes.append((sxmin.item(), symin.item(), sxmax.item(), symax.item()))
            obj_bboxes.append((oxmin.item(), oymin.item(), oxmax.item(), oymax.item()))
            
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
            
            imCropSub = imageReturn.crop((sxmin.item(), symin.item(), sxmax.item(), symax.item()))
            attSub = GenerateAttr(imCropSub)
            imCropObj = imageReturn.crop((oxmin.item(), oymin.item(), oxmax.item(), oymax.item()))
            attObj = GenerateAttr(imCropObj)
            newGraph = {
                'AttSub': attSub.split(' is ')[-1],
                'Sub': CLASSES[probas_sub[idx].argmax()],
                'Rel': REL_CLASSES[probas[idx].argmax()],
                'AttObj': attObj.split(' is ')[-1],
                'Obj': CLASSES[probas_obj[idx].argmax()]
            }
            orgGraph = {
                'Sub': CLASSES[probas_sub[idx].argmax()],
                'Rel': REL_CLASSES[probas[idx].argmax()],
                'Obj': CLASSES[probas_obj[idx].argmax()]
            }
            lstNewGraph.append(newGraph)
            lstOrgGraph.append(orgGraph)
            ax.axis('off')
            print('New graph: ', attSub.split(' is ')[-1] + ' ' + CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+ ' ' + attObj.split(' is ')[-1] + ' '+CLASSES[probas_obj[idx].argmax()])
            print('Origin graph: ', CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+' '+CLASSES[probas_obj[idx].argmax()])
            print(20*'=')
            ax.set_title(attSub + ' ' + CLASSES[probas_sub[idx].argmax()]+' '+REL_CLASSES[probas[idx].argmax()]+ ' ' + attObj + ' '+CLASSES[probas_obj[idx].argmax()], fontsize=10)

        # fill dataframe with one row per object, one attribute per column
        df = pd.DataFrame([t.__dict__ for t in lstNewGraph ])

        print(df)
        fig.tight_layout()
        plt.show() # show the output
    return sub_bboxes, obj_bboxes, imageReturn

if __name__=="__main__":
    imageUrl = 'Datasets/VG/VG_100K/235.jpg'
    sub_bboxes_scaled,obj_bboxes_scaled,imageReturn  = MainProcess(imageUrl)
    # print(imageReturn)
    # plt.imshow(imageReturn)
    # plt.show()
    # for (sxmin, symin, sxmax, symax), (oxmin, oymin, oxmax, oymax) in zip(sub_bboxes_scaled, obj_bboxes_scaled):
    #     imCropSub = imageReturn.crop((sxmin, symin, sxmax, symax))
    #     GenerateAttr(imCropSub)
    #     imCropObj = imageReturn.crop((oxmin, oymin, oxmax, oymax))
    #     GenerateAttr(imCropObj)
        # plt.imshow(imCropSub)
        # plt.show()
        # plt.imshow(imCropObj)
        # plt.show()