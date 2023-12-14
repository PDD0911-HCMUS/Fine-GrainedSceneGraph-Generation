import json
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil
datasetRooot ='Datasets/VG/'
datasetAttr = datasetRooot + 'Annotation/attributes.json'
datasetSG = datasetRooot + 'Annotation/scene_graphs.json'
dataIm = datasetRooot + 'VG_100K/'
# dataExtractImSave = datasetRooot + 'ExtractDemo/ImageObj/'
# dataExtractAttrSave = datasetRooot + 'ExtractDemo/AttrObj/'

dataExtractImSave = datasetRooot + 'Extract/ImageObj/'
dataExtractAttrSave = datasetRooot + 'Extract/AttrObj/'
dataExtractRootImSave = datasetRooot + 'Extract/Img/'

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
    for item in attr[:9000]:
        image = Image.open(dataExtractRootImSave + str(item['image_id']) + '.jpg')
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

def ExtractTokenAttrByImg():
    tokenByImg = []

    with open(os.path.join("Datasets/VG/Extract", 'Attr.token.txt'), 'w') as fw:
        for itemTxt in os.listdir("Datasets/VG/Extract/AttrObj"):
            with open(os.path.join("Datasets/VG/Extract/AttrObj", itemTxt)) as f:
                lines = f.readlines()
            line = itemTxt.replace('.txt','.jpg')+ '#0' + '\t' + str(lines).replace("[","").replace("'","").replace(","," ").replace("]","") + ' .' + '\n'
            #print(line)
            fw.write(line)

def GetObjectNameSG(objListJS, subjectId, objectId):
    subject = [item for item in objListJS if item['object_id'] == subjectId][0]
    object = [item for item in objListJS if item['object_id'] == objectId][0]

    if('attributes' in subject.keys()):
        attr = str(subject['attributes']).replace("[","").replace("'","").replace("]"," ").replace(","," and ")
        names = subject['names'][0]
        nameSubject = attr + names
    else:
        names = subject['names'][0]
        nameSubject = names
    
    if('attributes' in object.keys()):
        attr = str(object['attributes']).replace("[","").replace("'","").replace("]"," ").replace(","," and ")
        names = object['names'][0]
        nameObject = attr + names
    else:
        names = object['names'][0]
        nameObject = names
    #print(nameSubject, nameObject)
    return nameSubject, nameObject

def ExtractSG():
    itemTxt = ''
    sg = json.load(open(datasetSG))
    print(len(sg))
    with open(os.path.join("Datasets/VG/Extract", 'SG.token.txt'), 'w') as fw:
        for item in sg[:9000]:
            objListJS = item['objects']
            for i, itemSg in enumerate(item['relationships'][:10], start=0):
                nameSubject, nameObject = GetObjectNameSG(objListJS, itemSg['subject_id'], itemSg['object_id'])
                #print(nameSubject + " " + itemSg['predicate'] + " " + nameObject)
                lines = nameSubject + " " + itemSg['predicate'] + " " + nameObject
                line = str(item['image_id']) + '.jpg' + '#' + str(i) + '\t' + lines + ' .' + '\n'
                fw.write(line)
            shutil.copyfile(dataIm + '/'+ str(item['image_id']) + '.jpg', dataExtractRootImSave +'/'+ str(item['image_id']) + '.jpg')
            #print(20*'=')
    return

#STEP 1
#ExtractSG()

#STEP 2
#Extract()

#STEP 3
ExtractTokenAttrByImg()

