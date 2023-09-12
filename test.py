import boto3
import torch
from PIL import Image

#CodeWishperer: Create a nn for image classification

def image_classification(): 
    #load the model
    model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
    model.eval()
    #load the image
    img = Image.open('cat.jpg')
    #preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])  
    img = preprocess(img)
    img = torch.unsqueeze(img, 0)
    #make prediction
    with torch.no_grad():
        output = model(img)
    #get the index of the highest probability
    pred = output.argmax(dim=1, keepdim=True)
    print(pred)