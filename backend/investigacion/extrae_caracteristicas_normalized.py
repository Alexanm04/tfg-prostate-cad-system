import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import os
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from  PIL import Image
import timm 
from timm.layers import SwiGLUPacked
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from transformers import AutoImageProcessor, AutoModel,CLIPProcessor, CLIPModel
from conch.open_clip_custom import create_model_from_pretrained

model_selected = "dinov3"
phase = "test"
csv = "../CROWDGLEASON/test.csv"

if phase in ["train", "val"]:
   images = "./dataset_norm/SICAPv2"
else:
   images = "./dataset_norm/CrowdGleason/test"

output_dir = f"./features/{phase}"
os.makedirs(output_dir, exist_ok=True)
output = f"{output_dir}/features_{model_selected}{phase}_norm.npy"

if  model_selected == "resnet50":
   model = models.resnet50(weights = ResNet50_Weights.IMAGENET1K_V2)
   model.fc  = nn.Identity()
   transformation = transforms.Compose([ transforms.Resize(256),
      transforms.CenterCrop(224), 
      transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),])

elif model_selected == "uni2":
   timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
   model = timm.create_model("hf_hub:MahmoodLab/uni2-h", pretrained=True, **timm_kwargs)
   transformation = transforms.Compose([ transforms.Resize(224),
      transforms.ToTensor(), 
      transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

elif model_selected == "gigapath":
   model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
   transformation = transforms.Compose([
      transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
   ])

elif model_selected == "conch":
   token="tokenfalso"
   model, transformation  = create_model_from_pretrained('conch_ViT-B-16', "hf_hub:MahmoodLab/conch", hf_auth_token= token)

elif model_selected == "virchow2":
   model = timm.create_model("hf_hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
   transformation = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))

elif model_selected == "phikon2":
   processor = AutoImageProcessor.from_pretrained("owkin/phikon-v2")
   model = AutoModel.from_pretrained("owkin/phikon-v2")
   transformation = processor

elif model_selected == "plip":
   processor = CLIPProcessor.from_pretrained("vinid/plip")
   model = CLIPModel.from_pretrained("vinid/plip")
   transformation = processor

elif model_selected == "dinov3":
   processor = AutoImageProcessor.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
   model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
   transformation = processor

else:
   print("Not detected model")
   exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

df = pd.read_csv(csv)
features_l=[]
total_images = df.shape[0]

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
   for index, row  in df.iterrows():
      if 'image_name' in row:
         name = str(row['image_name'])
      elif 'Patch filename' in row:
         name = str(row['Patch filename'])
         if not name.endswith('.jpg'):
            name += '.jpg'
      else:
         print("Not detected column name")
         break

      path = os.path.join(images, name)
      img = Image.open(path)
      if model_selected == "phikon2":
         inputs = processor(img, return_tensors="pt").to(device)
         outputs = model(**inputs)
         features = outputs.last_hidden_state[:, 0, :]
         features_l.append(features.cpu().float().numpy().flatten())
      elif model_selected == "plip":
         inputs = transformation(images=img, return_tensors="pt")
         pixel_values = inputs['pixel_values'].to(device)
         outputs = model.vision_model(pixel_values=pixel_values)
         features = outputs.pooler_output
         features_l.append(features.cpu().float().numpy().flatten())
      elif model_selected == "dinov3":
         inputs = transformation(images=img, return_tensors="pt").to(device)
         outputs = model(**inputs)
         features = outputs.pooler_output
         features_l.append(features.cpu().float().numpy().flatten()) 
      elif model_selected == "conch":
         img_tensor = transformation(img).unsqueeze(0).to(device)
         features = model.encode_image(img_tensor, proj_contrast=False, normalize=False)
         features_l.append(features.cpu().float().numpy().flatten()) 
      else:
         img_tensor = transformation(img).unsqueeze(0).to(device)
         features = model(img_tensor)

         if model_selected == "virchow2":
            class_token = features[:,0]
            patch_tokens = features[:,5:]
            embedding =torch.cat([class_token, patch_tokens.mean(1)], dim=-1)
            features_l.append(embedding.cpu().float().numpy().flatten())
         else:
            features_l.append(features.cpu().float().numpy().flatten())
      
      if (index + 1) % 5000 == 0 or (index + 1) == total_images:
         print(f"Extracted {index + 1} of {total_images} images")

np.save(output, np.array(features_l))
print(f"Array with shape: {np.array(features_l).shape}")


