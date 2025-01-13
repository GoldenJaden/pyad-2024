import torch
import torchvision
import torchvision.transforms as transforms
import timm
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

# Hyperparameters
batch_size = 64
lr = 0.001
epochs = 20
model_name = "resnet50"

# Fashion-MNIST Dataset
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(28, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Model
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()

        self.backbone = timm.create_model("resnet34", pretrained=True, num_classes=10)

        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.backbone.conv1.out_channels,
            kernel_size=self.backbone.conv1.kernel_size,
            stride=self.backbone.conv1.stride,
            padding=self.backbone.conv1.padding,
            bias=self.backbone.conv1.bias is not None,
        )

    def forward(self, x):
        return self.backbone(x)

model = FashionMNISTModel()

model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=lr)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Training Loop
best_acc = 0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    train_loss, train_acc = train_one_epoch(model, trainloader, optimizer, criterion)
    val_loss, val_acc = validate_one_epoch(model, testloader, criterion)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}, Validation Acc: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_fashionmnist_model.pt")
        print("Saved Best Model!")

print("Training Completed. Best Validation Accuracy: {:.2f}%".format(best_acc))

# Load and Evaluate Best Model
model.load_state_dict(torch.load("best_fashionmnist_model.pt"))
model.eval()
test_loss, test_acc = validate_one_epoch(model, testloader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
