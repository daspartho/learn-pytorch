import torchvision
import argparse
import torch
import requests
import model_builder

parser = argparse.ArgumentParser()

parser.add_argument(
    "--url",
    type=str,
    help="image url"
    )

args = parser.parse_args()

url= args.url

class_names= ["pizza", "steak", "sushi"]
transform= torchvision.transforms.Resize((64, 64))
device = "cuda" if torch.cuda.is_available() else "cpu"

model = model_builder.TinyVGG(3, 10, 3).to(device)
model.load_state_dict(torch.load("models/tinyvgg_model.pth"))

with open("test.jpeg", "wb") as f:
        request = requests.get(url)
        f.write(request.content)

img = torchvision.io.read_image("test.jpeg").type(torch.float32)
img = img/255
img = transform(img)

model.to(device)
model.eval()
with torch.inference_mode():
    logits = model(img.unsqueeze(dim=0).to(device))
    pred_label = torch.argmax(torch.softmax(logits, dim=1), dim=1)

print(f"Pred: {class_names[pred_label.cpu()]}")
