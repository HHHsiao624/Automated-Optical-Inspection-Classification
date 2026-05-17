import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from time import time as timer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
#print("PyTorch version:", torch.__version__)  
##print("CUDA available:", torch.cuda.is_available())  
#print("CUDA device count:", torch.cuda.device_count())  
#print("Current CUDA device:", torch.cuda.current_device())  
#print("Current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))  

'''資料的路徑'''
train_csv = r'data\train.csv'
test_csv = r'data\test.csv'
train_images = r'data\train_images'
test_images = r'data\test_images'

train_data = pd.read_csv(train_csv)
test_data = pd.read_csv(test_csv)

'''資料預處理'''
# EfficientNet參考:https://spingence.medium.com/efficientnet-%E6%A8%A1%E5%9E%8B%E6%94%BE%E5%A4%A7%E7%9A%84%E6%96%B0%E6%80%9D%E7%B6%AD-bef2062ff070
transform = transforms.Compose([
    #transforms.Resize((224, 224)),  # 先將大小改為EfficientNet-B0输入大小
    transforms.Resize((240, 240)), #EfficientNet-B1
    transforms.RandomHorizontalFlip(),  #水平翻轉
    transforms.RandomVerticalFlip(),  #垂直翻轉
    transforms.RandomRotation(10),   #旋轉角度在-10~10
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  #改變圖片亮度
    transforms.ToTensor(), #轉變pytorch張量
])
class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self): 
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB') 
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label)  


train_dataset = Dataset(train_data, train_images, transform)
test_dataset = Dataset(test_data, test_images, transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#print(train_data.head(5))
#print(test_data.head(5))

# 計算每個類別的數量
class_counts = train_data['Label'].value_counts().reindex(range(6), fill_value=0)
for label, count in class_counts.items():
    print(f'Label {label}: {count}')

#print(torch.__version__) 
#print(torch.cuda.is_available())  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #GPU
#print(device)  

'''模型'''
#ef_model = models.efficientnet_b0(weights='DEFAULT')
ef_model = models.efficientnet_b1(weights='DEFAULT')
num_classes = 6  #類別:0,1,2,3,4,5
ef_model.classifier[1] = nn.Linear(ef_model.classifier[1].in_features, num_classes) #最後一層輸出
ef_model = ef_model.to(device)  #GPU

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(ef_model.parameters(), lr=0.001)

epochs = 15  #輪數
start = timer()

# 確保 weights 資料夾存在
os.makedirs('weight', exist_ok=True) #存權重的資料夾weight
os.makedirs('pic', exist_ok=True) #存圖片


'''訓練'''
#'''
for epoch in range(epochs):
    train_loss = 0.0
    train_correct = 0  # 正確預測的數量
    total_samples = 0   # 總樣本數
    ef_model.train()  # 將模型設置為訓練模式
    #print(f'第{epoch + 1}輪')
    for images, labels in tqdm(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  
        outputs = ef_model(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step()  # 更新權重
        train_loss += loss.item()  
        # 計算準確率
        _, predicted = torch.max(outputs, 1)  
        train_correct += (predicted == labels).sum().item()  
        total_samples += labels.size(0)  # 樣本數


    avg_loss = train_loss / len(train_loader)  #損失率
    accuracy = (train_correct / total_samples) * 100  # 計算準確率

    print(f'第{epoch + 1}輪 損失:{avg_loss:.4f}, 準確率: {accuracy:.2f} %')
    torch.save(ef_model.state_dict(), f'weight/test6/model_epoch_{epoch+1}.pth')

torch.save(ef_model.state_dict(), 'weight/test6/model_all_epochs.pth')

end_time = timer()
total_time = end_time - start
print(f'時間: {total_time:.2f} 秒')
#'''

'''提取特徵'''
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.model = model
        self.feature_layer = nn.Sequential(*list(model.children())[:-1])  # 沒有包含分類器

    def forward(self, x):
        return self.feature_layer(x)

feature_extractor = FeatureExtractor(ef_model).to(device)

def extract_features(data_loader, model):
    model.eval()  
    all_features = []
    all_labels = []
    with torch.no_grad():
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            features = model(images)
            features = features.view(features.size(0), -1)  
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)

train_features, train_labels = extract_features(train_loader, feature_extractor)

'''畫圖'''
#訓練集
tsne = TSNE(n_components=2, random_state=42)
train_tsne = tsne.fit_transform(train_features) 

plt.figure(figsize=(10, 8))
plt.title('t-SNE of Training Features') 
scatter = plt.scatter(train_tsne[:, 0], train_tsne[:, 1], c=train_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='label')
plt.savefig('pic/t-SNE_final6_train.jpg')
plt.show()

'''進行預測'''
ef_model.load_state_dict(torch.load('weight/test2/model_all_epochs.pth', weights_only=True)) #加載先前訓練好的權重
ef_model.eval()
predictions = []

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        images = images.to(device)  
        #print(type(images), images.shape)    #class 'torch.Tensor'> torch.Size([4, 3, 224, 224])
        outputs = ef_model(images)
        _, predicted = torch.max(outputs, 1)
        predictions.append(predicted)

predictions = torch.cat(predictions).cpu().numpy()
sub_df = pd.DataFrame({'ID': test_data['ID'], 'Label': predictions}) #輸出預測的答案
sub_df.to_csv('final6.csv', index=False)  #結果+原test寫新的csv


#要用新生成的CSV檔案去畫test data的圖
new_test_csv = r'final6.csv' 
test_data = pd.read_csv(new_test_csv)

test_dataset = Dataset(test_data, test_images, transform)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

test_features, test_labels = extract_features(test_loader, feature_extractor)
'''畫圖'''
#測試集
test_tsne = tsne.fit_transform(test_features)
plt.figure(figsize=(10, 8))
plt.title('t-SNE of Testing Features') 
scatter_test = plt.scatter(test_tsne[:, 0], test_tsne[:, 1], c=test_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter_test, label='label')
plt.savefig('pic/t-SNE_final6_test.jpg')
plt.show()
