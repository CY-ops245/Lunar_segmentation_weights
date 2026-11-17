# 选用轻量且稳定的 Python 3.8 官方镜像
FROM python:3.8-slim

# 设置容器内的工作目录
WORKDIR /workspace

# 设置非交互式安装
ENV DEBIAN_FRONTEND=noninteractive

# 安装 OpenCV 等必要的系统底层库
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖文件并安装 Python 库（利用 Docker 缓存加速）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 将项目根目录下的所有源码、脚本和权重复制到容器中
COPY . /workspace/

# 容器启动时默认执行的推理命令
CMD ["python", "predict.py"]