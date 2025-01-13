import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import nn
import timm  # Для использования предобученной ResNet

# Гиперпараметры
batch_size = 64
lr = 0.001
device = "cuda" if torch.cuda.is_available() else "cpu"

# Трансформации для тестового набора
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Тестовый набор
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Определение модели
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

# Загрузка обученной модели
model.load_state_dict(torch.load("best_fashionmnist_model.pt", map_location=device))
model.eval()  # Перевод модели в режим оценки

# Функция для валидации
def validate_one_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc

# Задаем функцию потерь
criterion = nn.CrossEntropyLoss()

# Проверка на тестовой выборке
test_loss, test_acc = validate_one_epoch(model, testloader, criterion)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

# Словарь с названиями классов Fashion-MNIST
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Берем одно изображение из тестового набора
image, label = testset[1]  # Извлекаем первое изображение и метку
plt.imshow(image.squeeze(), cmap="gray")  # Отображаем изображение
plt.title(f"Actual: {class_labels[label]}")
plt.show()
# Преобразуем изображение и делаем предсказание
image = image.unsqueeze(0).to(device)  # Добавляем размер батча и отправляем на устройство
output = model(image)
_, predicted = torch.max(output, 1)  # Индекс класса с максимальной вероятностью
predicted_label = class_labels[predicted.item()]

print(f"Predicted: {predicted_label}")
