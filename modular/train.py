import data_setup, model_builder, engine, utils
import torch
import argparse

from torchvision import transforms

parser = argparse.ArgumentParser(description="hyperparameters")

parser.add_argument(
    "--num_epochs",
    default=10,
    type=int,
    help="number of epochs to train for"
    )
parser.add_argument(
    "--batch_size",
    default=32,
    type=int,
    help="number of samples per batch"
    )

parser.add_argument(
    "--lr",
    default=1e-3,
    type=float,
    help="learning rate"
    )

args = parser.parse_args()

BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.num_epochs
LEARNING_RATE = args.lr

train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
])

train_dataloader, test_dataloader = data_setup.create_dataloaders(
    train_dir,
    test_dir,
    data_transform,
    BATCH_SIZE,
)

model = model_builder.TinyVGG(3, 10, 3).to(device)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

engine.train(model,
             train_dataloader,
             test_dataloader,
             loss_fn,
             optimizer,
             NUM_EPOCHS,
             device)

utils.save_model(model,  "models", "tinyvgg_model.pth")
