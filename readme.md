下面是完整的 README，包括如何使用权重文件进行测试的说明。

---

# GAN生成的CIFAR-100图像分类

本项目使用生成对抗网络（GAN）生成CIFAR-100数据集的图像，并使用ResNet-18模型进行分类。以下是如何运行该项目的详细说明。

## 目录

- [依赖项](#依赖项)
- [数据集](#数据集)
- [训练和测试](#训练和测试)
- [使用权重文件](#使用权重文件)
- [模型评估](#模型评估)

## 依赖项

Python 3.8.10

请确保安装了以下依赖项。可以使用 `requirements.txt` 文件进行安装：

```bash
pip install -r requirements.txt
```

`requirements.txt` 文件内容如下：

```
torch==2.4.1+cu121
torchvision==0.19.1+cu121
numpy==1.24.4

```

## 数据集

本项目使用CIFAR-100数据集。第一次运行时，代码会自动下载该数据集，下载路径为 `./data`。

## 训练和测试

### 1. 训练GAN

要训练生成对抗网络（GAN），运行以下代码：

```python
# 训练GAN
train_gan(generator, discriminator, train_loader, num_epochs=50, device=device)
```

这将训练GAN模型，生成CIFAR-100图像。

### 2. 使用生成的数据训练ResNet-18

在训练GAN后，您可以使用生成的图像训练ResNet-18模型：

```python
# 初始化ResNet-18
resnet18 = models.resnet18(weights='IMAGENET1K_V1')  # 使用ImageNet预训练权重
resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)  # 修改输出层
resnet18 = resnet18.to(device)

# 使用生成的数据训练ResNet-18
train_resnet(resnet18, train_loader, test_loader, num_epochs=50, device=device)
```

### 3. 评估模型

在训练完成后，您可以使用以下代码评估模型的准确性：

```python
# 评估GAN生成的ResNet-18模型
gan_resnet_accuracy = evaluate_model(resnet18, test_loader, device)

# 评估从零开始训练的ResNet-18模型（需要另外训练的代码）
zero_resnet_accuracy = evaluate_model(resnet18_zero, test_loader, device)

print(f'GAN ResNet Accuracy: {gan_resnet_accuracy:.2f}%')
print(f'Zero ResNet Accuracy: {zero_resnet_accuracy:.2f}%')
```

## 使用权重文件

### 1. 下载权重文件

如果您希望使用预训练的模型权重，请确保下载相关的权重文件，并将其放置在项目的根目录或指定路径下。

### 2. 加载权重文件

在代码中，您可以通过以下方式加载权重文件：

```python
# 初始化ResNet-18
resnet18 = models.resnet18(weights=None)  # 不使用默认权重
resnet18.fc = nn.Linear(resnet18.fc.in_features, 100)  # 修改输出层
resnet18.load_state_dict(torch.load('path/to/your/weights.pth'))  # 加载权重文件
resnet18 = resnet18.to(device)
```

请将 `path/to/your/weights.pth` 替换为您权重文件的实际路径。

### 3. 测试模型

加载权重后，您可以使用以下代码测试模型的准确性：

```python
# 评估使用预训练权重的ResNet-18模型
pretrained_accuracy = evaluate_model(resnet18, test_loader, device)
print(f'使用预训练权重的ResNet-18准确率: {pretrained_accuracy:.2f}%')
```

## 模型评估

最后，比较GAN生成的ResNet-18模型和从零开始训练的ResNet-18模型的准确率：

```python
print("\n模型比较结果:")
print(f"GAN生成的ResNet-18准确率: {gan_resnet_accuracy:.2f}%")
print(f"从零开始训练的ResNet-18准确率: {zero_resnet_accuracy:.2f}%")
print(f'使用预训练权重的ResNet-18准确率: {pretrained_accuracy:.2f}%')
```

