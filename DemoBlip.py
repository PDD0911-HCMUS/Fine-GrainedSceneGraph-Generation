# from PIL import Image
# import requests
# import torch
# from torchvision import transforms
# from torchvision.transforms.functional import InterpolationMode
# from BLIP.models.blip import blip_decoder


# print(torch.cuda.is_available(),
# torch.cuda.device_count(),
# torch.cuda.current_device(),
# torch.cuda.device(0),
# torch.cuda.get_device_name(0)
# )

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
print(model.eval())

# img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
# raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

img_url = 'Datasets/VG/VG_100K/1.jpg' 
raw_image = Image.open(img_url).convert('RGB')

# conditional image captioning
text = "a photography of"
inputs = processor(raw_image, text, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# unconditional image captioning
inputs = processor(raw_image, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))