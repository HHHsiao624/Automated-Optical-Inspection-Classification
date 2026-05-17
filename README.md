# Automated-Optical-Inspection-Classification
利用深度學習對工業製造影像進行瑕疵分類，於工研院 AIDEA 平台測試達到 **99.2%** 辨識準確度。

## 資料集

來源：[AIDEA 平台](https://www.aidea-web.tw/topic/285ef3be-44eb-43dd-85cc-f0388bf85ea4)

| 檔案 | 內容 |
|---|---|
| `train_images.zip` | 2,528 張訓練影像（.png）|
| `train.csv` | 訓練資料 ID 與 Label |
| `test_images.zip` | 10,142 張測試影像（.png）|
| `test.csv` | 測試資料 ID |

## 分類類別

| Label | 說明 |
|---|---|
| 0 | 正常 |
| 1 | 瑕疵程度 1 |
| 2 | 瑕疵程度 2 |
| 3 | 瑕疵程度 3 |
| 4 | 瑕疵程度 4 |
| 5 | 瑕疵程度 5 |

## 模型

- **EfficientNet-B1**

## 結果

- AIDEA 平台測試準確度：**99.2%**
