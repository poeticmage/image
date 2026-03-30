# Fundus Image Processing API

## Overview https://funduseyefrontend12.onrender.com/
## This API accepts a fundus image, validates it using one external service, and if accepted, sends it to another service for prediction. | Node.js Express Axios FormData CORS

### Client uploads an image
### Image is sent to API1 for validation
### If rejected, response is returned immediately
### If accepted, image is sent to API2
### Final prediction is returned

## API1: Checks whether the image is a Fundus image at all based on an ML application on DBSCAN clustering https://lioninthestreets-fundusimagegate.hf.space/check
## API2: Studies the image, extracts information, matches with the trained weights of a Light MaxViT based algorithm. https://lioninthestreets-maxvitgradcam.hf.space/predict
```
import express from "express";
import axios from "axios";
import FormData from "form-data";
import multer from "multer";
import serverless from "serverless-http"; 
import cors from "cors";

const app = express();
app.use(cors());
const upload = multer();

const API1 = "https://lioninthestreets-fundusimagegate.hf.space/check";
const API2 = "https://lioninthestreets-maxvitgradcam.hf.space/predict";

app.post("/image", upload.single("file"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400);
    }
    const form = new FormData();
    form.append("file", req.file.buffer, req.file.originalname);
    const response1 = await axios.post(API1, form, {
      headers: form.getHeaders(),
    });
    console.log("API1:", response1.data);
    const status1 = response1.data.status?.toString().toLowerCase();
    if (status1 ==="reject") {
      return res.json({
        result: response1.data
      });
    }
    const form2 = new FormData();
    form2.append("file", req.file.buffer, req.file.originalname);
    const response2 = await axios.post(API2, form2, {
      headers: form2.getHeaders(),
    });
    console.log("API2:", response2.data);
    return res.json({
      result: response2.data,
    });
  } catch (e) {
    console.error(e);
    return res.status(500);
  }
});



const port = process.env.PORT || 3000;
app.listen(port, () => console.log("Running on", port));



// app.listen(3000, () => {
//   console.log("Server running on port 3000");
// });
```
## API1: https://lioninthestreets-fundusimagegate.hf.space/check This is a DBSCAN algorithm that act as a gate. It is deployed in Hugging Face Spaces using Docker
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances
from fastapi import FastAPI, UploadFile
import uvicorn



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
backbone = nn.Sequential(*list(backbone.children())[:-1])
backbone.eval()
backbone.to(device)


transform=transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor(),])
artifact = pickle.load(open("./dbscan_clusters.pkl","rb"))
core_points = artifact["core_points"]
core_labels = artifact["core_labels"]
eps = artifact["eps"]
centroid = np.mean(core_points, axis=0, keepdims=True)
threshold = 0.1


def get_embedding(img):
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = backbone(img)
        emb = emb.view(emb.size(0), -1)
        emb = F.normalize(emb, dim=1)
    return emb.cpu().numpy()[0]

def distance_gate(embedding):
    dist = cosine_distances(embedding.reshape(1,-1), centroid)[0,0]
    if dist <= threshold:
        return "accept"
    return "reject"


def check_image(img):
    return distance_gate(get_embedding(img))


# img = Image.open("myreactapp/src/Non Referable/EyePACS-DEV-NRG-1.jpg").convert("RGB")
# result = check_image(img)
# print(result) 
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

app = FastAPI(title="Fundus Image Gate")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/check")
async def check_file(file: UploadFile = File(...)):
    try:
        img = Image.open(file.file).convert("RGB")
        result = check_image(img)
        return JSONResponse({"status": result})
    except Exception as e:
        return JSONResponse({"status": "Error", "detail": str(e)})
    

```
### API2:  https://lioninthestreets-maxvitgradcam.hf.space/predict 
#### This code builds a deep learning image classifier using a hybrid CNN–Transformer architecture inspired by MaxViT, where convolution layers first extract spatial features and attention mechanisms then model both local and global relationships using window-based and grid-based self-attention. It defines custom components like a 2D LayerNorm, squeeze-and-excitation blocks for channel attention, and transformer-style encoder blocks, all combined into stacked stages that progressively downsample and enrich features before classification. The model is wrapped using PyTorch Lightning for structure, and Grad-CAM is implemented to visualize which parts of the image influence predictions by using gradients and activations. Finally, the system is deployed through a FastAPI server that accepts an image, runs inference, generates a heatmap overlay, and returns the prediction, confidence scores, and visualization as a base64-encoded image.
```
import torch
import lightning as L
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_distances
from fastapi import FastAPI, UploadFile
import uvicorn
import cv2
import io
import base64


class LayerNorm2d(nn.Module):
    def __init__(self,dim,eps=1e-6):
       super().__init__() 
       self.norm=nn.LayerNorm(dim,eps=eps)
    def forward(self,x):
        x=x.permute(0,2,3,1)
        x=self.norm(x)
        return x.permute(0,3,1,2)

class SEBlock(nn.Module):
    def __init__(self,dim,reduction=4):
        super().__init__()
        self.fc1=nn.Linear(dim,dim//reduction)
        self.fc2=nn.Linear(dim//reduction,dim)
    def forward(self,x):
        b,c,h,w=x.shape
        y=x.mean(dim=(2,3))
        y=F.relu(self.fc1(y))
        y=torch.sigmoid(self.fc2(y))
        return x*y.view(b,c,1,1)

class MLP(nn.Module):
    def __init__(self,dim,hidden_dim):
        super().__init__()
        self.fc1=nn.Linear(dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,dim)
    def forward(self,x):
        return self.fc2(F.gelu(self.fc1(x)))

def window_partition(x,window_size):
    B,C,H,W=x.shape
    x=x.view(B,C,H//window_size,window_size,W//window_size,window_size)
    x=x.permute(0,2,4,3,5,1)
    return x.reshape(-1,window_size*window_size,C)
def window_reverse(x,window_size,H,W,B,C):
    x=x.view(B,H//window_size,W//window_size,window_size,window_size,-1)
    x=x.permute(0,5,1,3,2,4)
    return x.reshape(B,C,H,W)

class AttentionEncoder(nn.Module):
    def __init__(self,dim,num_heads,mlp_ratio=4):
        super().__init__()
        self.norm1=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim,num_heads,batch_first=True)
        self.norm2=nn.LayerNorm(dim)
        self.mlp=MLP(dim,dim*mlp_ratio)
    def forward(self,x):
        x=x+self.attn(self.norm1(x),self.norm1(x),self.norm1(x))[0]
        x=x+self.mlp(self.norm2(x))
        return x

class MaxVitBlock(nn.Module):
    def __init__(self,dim,window_size=8,num_heads=8):
        super().__init__()
        self.conv1=nn.Conv2d(dim,dim,1) # Conv-block
        self.conv2=nn.Conv2d(dim,dim,3,padding=1,groups=dim)
        self.se=SEBlock(dim)
        self.conv3=nn.Conv2d(dim,dim,1)
        self.norm=LayerNorm2d(dim)
        self.block_attn=AttentionEncoder(dim,num_heads)
        self.grid_attn=AttentionEncoder(dim,num_heads)
        self.window_size=window_size
    def forward(self,x):
        B,C,H,W=x.shape
        residual=x
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.se(x)
        x=self.conv3(x)
        x=self.norm(x)
        x=x+residual
        xb=window_partition(x,self.window_size)
        xb=self.block_attn(xb)
        x=window_reverse(xb,self.window_size,H,W,B,C)
        xg=x.permute(0,1,3,2)
        xg=window_partition(xg,self.window_size)
        xg=self.grid_attn(xg)
        xg=window_reverse(xg,self.window_size,W,H,B,C)
        x=xg.permute(0,1,3,2)
        return x

class MaxVIT(nn.Module):
    def __init__(self,num_classes=2,dims=(64,128,256,512)):
        super().__init__()
        self.stem=nn.Sequential(
            nn.Conv2d(3,dims[0],3,stride=2,padding=1),
            nn.Conv2d(dims[0],dims[0],3,stride=2,padding=1),
            LayerNorm2d(dims[0])
        )
        self.stages=nn.ModuleList()
        for i in range(4):
            blocks=nn.Sequential(*[MaxVitBlock(dims[i]) for _ in range(2)])
            self.stages.append(blocks)
            if i<3:
                self.stages.append(nn.Conv2d(dims[i],dims[i+1],3,stride=2,padding=1))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(dims[-1],num_classes)
    def forward(self,x):
        x=self.stem(x)
        for l in self.stages:
                x=l(x)
        x=self.pool(x).flatten(1)
        return self.fc(x)
class CNNAttention(L.LightningModule):
    def __init__(self):
        super().__init__()
        L.seed_everything(42)
        self.model=MaxVIT(num_classes=2)
    def forward(self,x):
        return self.model(x)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activations = output
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_tensor)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[:, class_idx]
        score.backward()
        gradients = self.gradients
        activations = self.activations
        weights = gradients.mean(dim=(2,3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam
    

    
import torch
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# MODEL AND CHECKPOINT PLEASE LOOK HERE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1!!!!!11223!!!!!!
model = CNNAttention()
ckpt = torch.load("./epoch=14-step=7500.ckpt", map_location=device)


model.load_state_dict(ckpt["state_dict"])
model.to(device)
model.eval()
transform=transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor(),])

gradcam = GradCAM(
    model=model,
    target_layer=model.model.stages[-1][-1]
)

def predict(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
        prob = torch.softmax(out, dim=1)
        pred = prob.argmax(dim=1).item()
    img_cam = img.clone().detach()
    img_cam.requires_grad = True
    with torch.enable_grad():
        out_cam = model(img_cam)
        score = out_cam[:, pred]
        model.zero_grad()
        score.backward()
        cam = gradcam.generate(img_cam, pred)
    return pred, prob.cpu().numpy().tolist(), cam

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_api(file: UploadFile):

    image = Image.open(file.file).convert("RGB")

    pred, prob, cam = predict(image)
    img_resized = image.resize((512,512))
    img_np = np.array(img_resized) / 255.0
    heatmap = cv2.applyColorMap(
        np.uint8(255 * cam),
        cv2.COLORMAP_JET
    )
    heatmap = heatmap[..., ::-1] / 255.0
    overlay = (0.65 * img_np) + (0.35 * heatmap)
    overlay = np.clip(overlay, 0, 1)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    pil_overlay = Image.fromarray(overlay_uint8)
    buffer = io.BytesIO()
    pil_overlay.save(buffer, format="PNG")
    overlay_b64 = base64.b64encode(buffer.getvalue()).decode()
    return {
        "prediction": pred,
        "confidence": prob,
        "gradcam": overlay_b64
    }
```
  
