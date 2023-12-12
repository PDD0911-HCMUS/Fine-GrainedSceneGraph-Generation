import json
from PIL import Image
import matplotlib.pyplot as plt
import os
datasetRooot ='Datasets/VG/'
datasetAttr = datasetRooot + 'Annotation/attributes.json'
dataIm = datasetRooot + 'VG_100K/'
# dataExtractImSave = datasetRooot + 'ExtractDemo/ImageObj/'
# dataExtractAttrSave = datasetRooot + 'ExtractDemo/AttrObj/'

dataExtractImSave = datasetRooot + 'Extract/ImageObj/'
dataExtractAttrSave = datasetRooot + 'Extract/AttrObj/'

def ExtractObjectsAndAttr(image, imageId, coordiates, attrs, names):
    try:
        imCrop = image.crop(coordiates)
        width, height = imCrop.size
        if(width >= 32 and height >= 32):
            nameImSave = imageId + '_' + names
            imCrop.save(dataExtractImSave + nameImSave + '.jpg')
            with open(dataExtractAttrSave + nameImSave + '.txt', 'w') as f:
                f.write(
                    names + ' is ' + str(attrs)
                    .replace("[","")
                    .replace("'","")
                    .replace("]","")
                    .replace(","," and "))
    except Exception as e:
        print(e)
    return

def Extract():
    attr = json.load(open(datasetAttr))
    # print(len(attr))
    for item in attr[:50000]:
        image = Image.open(dataIm + str(item['image_id']) + '.jpg')
        for attItem in item['attributes']:
            try:
                if('attributes' in attItem.keys()):
                    coordinates = (
                        attItem['x'],
                        attItem['y'],
                        attItem['x'] + attItem['w'],
                        attItem['y'] + attItem['h']
                    )
                    names = attItem['names'][0].replace(" ", "_")
                    attr = attItem['attributes']
                    ExtractObjectsAndAttr(
                        image,
                        str(item['image_id']),
                        coordinates,
                        attr,
                        names
                    )
            except Exception as e:
                print(names)
                print(e)
                continue
        #     break
        # break

def ExtractTokenByImg():
    tokenByImg = []

    with open(os.path.join("Datasets/VG/Extract", 'Attr.token.txt'), 'w') as fw:
        for itemTxt in os.listdir("Datasets/VG/Extract/AttrObj"):
            with open(os.path.join("Datasets/VG/Extract/AttrObj", itemTxt)) as f:
                lines = f.readlines()
            line = itemTxt.replace('.txt','.jpg')+ '#0' + '\t' + str(lines).replace("[","").replace("'","").replace(","," ").replace("]","") + ' .' + '\n'
            #print(line)
            fw.write(line)

#Extract()

ExtractTokenByImg()