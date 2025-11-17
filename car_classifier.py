# ==============================================================================
# БИНАРНЫЙ КЛАССИФИКАТОР ДЛЯ ДЕТЕКЦИИ МАШИН
# Простая CNN на PyTorch
# ==============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np

# ==============================================================================
# 1. ПОДГОТОВКА ДАННЫХ
# ==============================================================================

class CarDataset(Dataset):
    """
    Датасет для загрузки изображений из папок.
    Структура папок должна быть:
    dataset/
        train/
            car/       <- изображения с машинами
            no_car/    <- изображения без машин
        val/
            car/
            no_car/
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Загружаем изображения с машинами (класс 1)
        car_dir = os.path.join(root_dir, 'car')
        if os.path.exists(car_dir):
            for img_name in os.listdir(car_dir):
                print(str(img_name))
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(car_dir, img_name))
                    self.labels.append(1)
        else:
            print(f'такого пути нет, {car_dir}')
        # Загружаем изображения без машин (класс 0)
        no_car_dir = os.path.join(root_dir, 'no_car')
        if os.path.exists(no_car_dir):
            for img_name in os.listdir(no_car_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(no_car_dir, img_name))
                    self.labels.append(0)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


# ==============================================================================
# 2. ОПРЕДЕЛЕНИЕ МОДЕЛИ CNN
# ==============================================================================

class SimpleCarClassifier(nn.Module):
    """
    Простая свёрточная нейронная сеть для бинарной классификации
    """
    def __init__(self):
        super(SimpleCarClassifier, self).__init__()
        
        # Блок 1: Conv -> ReLU -> MaxPool
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Блок 2: Conv -> ReLU -> MaxPool
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Блок 3: Conv -> ReLU -> MaxPool
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Полносвязные слои
        # Размер после трёх MaxPool (каждый /2): 128x128 -> 16x16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.relu4 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 1)  # Бинарная классификация
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Прямое распространение
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x


# ==============================================================================
# 3. ФУНКЦИИ ОБУЧЕНИЯ И ВАЛИДАЦИИ
# ==============================================================================

def train_model(model, train_loader, criterion, optimizer, device):
    """Одна эпоха обучения"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        # Обнуляем градиенты
        optimizer.zero_grad()
        
        # Прямое распространение
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Обратное распространение и оптимизация
        loss.backward()
        optimizer.step()
        
        # Статистика
        running_loss += loss.item()
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_model(model, val_loader, criterion, device):
    """Валидация модели"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


# ==============================================================================
# 4. ГЛАВНАЯ ФУНКЦИЯ
# ==============================================================================

def main():
    # Параметры
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    IMAGE_SIZE = 128  # Размер входного изображения
    
    # Путь к датасету (ИЗМЕНИ НА СВОЙ!)
    TRAIN_DIR = 'C:/Users/chuva/Documents/Python Scripts/binary_car_classieir/dataset/train'  # <- ЗДЕСЬ ТВОЯ ПАПКА С TRAIN
    VAL_DIR = 'C:/Users/chuva/Documents/Python Scripts/binary_car_classieir/dataset/val'      # <- ЗДЕСЬ ТВОЯ ПАПКА С VAL
    
    # Устройство (GPU или CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Используется устройство: {device}')
    
    # Трансформации для аугментации и нормализации
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка данных
    train_dataset = CarDataset(TRAIN_DIR, transform=train_transform)
    val_dataset = CarDataset(VAL_DIR, transform=val_transform)
    
    print(f'Количество обучающих изображений: {len(train_dataset)}')
    print(f'Количество валидационных изображений: {len(val_dataset)}')
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Создание модели
    model = SimpleCarClassifier().to(device)
    
    # Функция потерь и оптимизатор
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    print('\nНачинаем обучение...\n')
    
    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_model(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        print(f'Эпоха [{epoch+1}/{NUM_EPOCHS}]')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_car_classifier.pth')
            print(f'  -> Сохранена новая лучшая модель (Val Acc: {val_acc:.2f}%)')
        print()
    
    print(f'Обучение завершено! Лучшая точность на валидации: {best_val_acc:.2f}%')
    
    # Построение графиков
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Функция потерь')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Точность')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print('График обучения сохранён: training_history.png')


# ==============================================================================
# 5. ФУНКЦИЯ ДЛЯ ИНФЕРЕНСА (ПРЕДСКАЗАНИЯ НА НОВОМ ИЗОБРАЖЕНИИ)
# ==============================================================================

def predict_image(image_path, model_path='best_car_classifier.pth', image_size=128):
    """
    Предсказание для одного изображения
    
    Args:
        image_path: путь к изображению
        model_path: путь к сохранённой модели
        image_size: размер входного изображения
    
    Returns:
        prediction: 1 если машина, 0 если нет
        confidence: уверенность модели (0-1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    model = SimpleCarClassifier().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Загружаем и обрабатываем изображение
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)
        confidence = output.item()
        prediction = 1 if confidence > 0.5 else 0
    
    return prediction, confidence


if __name__ == '__main__':
    main()
    