# Lunar_segmentation_weights

## 1. 团队与项目概述
本项目针对 MoonSeg AI Challenge 2026 竞赛任务，基于 **SegFormer (`nvidia/mit-b0`)** 架构构建了端到端的月面语义分割与推理系统，实现了对月壤（0）、岩石（1）、撞击坑（2）、山体（3）和天空（4）的精准像素级分类。

## 2. 目录结构说明
```text
.
├── train.py                    # 端到端模型训练脚本（含动态色彩映射、日志记录、ONNX导出及图表生成）
├── predict.py                  # 测试集推理脚本（支持ONNX权重、生成8位单通道灰度及RGB掩码、加权mIoU统计）
├── check_dimensions.py         # 预测掩码与训练集掩码尺寸一致性自动校验脚本
├── weights/                    # 模型权重目录
│   ├── segformer_luar_best.pth   # PyTorch 训练保存的最优权重
│   └── segformer_luar_best.onnx  # 自动导出的动态 Batch ONNX 推理权重
├── LuSNAR_train/               # 官方训练集目录（包含 Moon_1, Moon_3, Moon_5, Moon_6, Moon_7, Moon_8）
├── LuSNAR_test/                # 官方测试集目录（包含 Moon_2, Moon_4, Moon_9）
├── LuSNAR_Predictions/         # 测试集推理输出目录（自动生成 label_gray 与 label_rgb）
├── docker/
│   └── Dockerfile              # Docker 容器构建配置文件
├── requirements.txt            # Python 依赖环境包列表
├── train.log                   # 完整训练日志文件
├── check_dimensions.log        # 尺寸一致性校验日志文件
├── loss_curve.png              # 训练与验证 Loss 变化折线图
└── scene_miou_weighted_bar.png # 场景 mIoU 与加权总分柱状图
```

## 3. 模型训练关键代码配置说明


### (2) `train.py` 需要按需修改的代码位置
在运行训练前，请根据本地硬件与路径情况修改 `train.py` 顶部的**可调参数配置区**：
```python
# ==================== [可调参数] ====================
DATA_ROOT = "./LuSNAR_train"            # 指向本地原始训练集根目录
MODEL_NAME = "nvidia/mit-b0"            # 骨干网络预训练权重（离线环境可改为本地权重文件夹路径）
NUM_CLASSES = 5                         # 分割类别总数（固定为 5，无需修改）
IMAGE_SIZE = (256, 256)                 # 训练输入分辨率 (H, W)
BATCH_SIZE = 2                          # 批大小
EPOCHS = 2                              # 训练总轮数
LEARNING_RATE = 6e-5                    # 初始学习率
WEIGHT_DECAY = 0.01                     # AdamW 权重衰减系数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测 GPU/CPU
# ==============================================================================
```
* **离线环境**：若运行环境无法访问 Hugging Face 下载预训练模型，请提前下载 `nvidia/mit-b0` 权重至本地文件夹，并将 `MODEL_NAME` 修改为该本地文件夹路径。

### (3) `predict.py` 需要按需修改的代码位置
在运行推理前，请核对 `predict.py` 顶部（约第 9–15 行）的路径与尺寸参数：
```python
TEST_DATA_ROOT = "./LuSNAR_test"                        # 指向官方测试集根目录（包含 Moon_2, Moon_4, Moon_9）
ONNX_MODEL_PATH = "./weights/segformer_luar_best.onnx"  # 指向训练导出的 ONNX 权重路径
OUTPUT_DIR = "./LuSNAR_Predictions"                     # 预测结果输出根目录
MODEL_INPUT_SIZE = (256, 256)                           # 必须与 train.py 中的 IMAGE_SIZE 保持完全一致
```


---

## 4. 可执行 Python 指令说明

### (1) 安装环境依赖
```bash
pip install -r requirements.txt
```

### (2) 启动模型训练（自动生成权重、日志及训练曲线）
```bash
python train.py
```

### (3) 启动测试集推理（基于 ONNX 生成单通道灰度/RGB 掩码及加权柱状图）
```bash
python predict.py
```


## 5. Docker 容器构建与运行指令

### (1) 构建镜像
在项目根目录下执行：
```bash
docker build -f docker/Dockerfile -t lunar-seg-app:v1 .
```

### (2) 运行容器执行推理
* **Linux / macOS**：
  ```bash
  docker run --rm -v $(pwd)/LuSNAR_test:/workspace/LuSNAR_test -v$(pwd)/LuSNAR_Predictions:/workspace/LuSNAR_Predictions lunar-seg-app:v1
  ```
* **Windows PowerShell**：
  ```powershell
  docker run --rm -v ${PWD}/LuSNAR_test:/workspace/LuSNAR_test -v${PWD}/LuSNAR_Predictions:/workspace/LuSNAR_Predictions lunar-seg-app:v1
  ```
