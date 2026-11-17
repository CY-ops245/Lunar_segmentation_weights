import os
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm

# ==================== [可调参数 / Tunable Parameters] ====================
TEST_DATA_ROOT = "./LuSNAR_test"  # 测试集根目录（包含各个测试场景，如 Moon_2, Moon_4, Moon_9）
ONNX_MODEL_PATH = "./weights/segformer_luar_best.onnx"  # ONNX 模型权重路径
OUTPUT_DIR = "./LuSNAR_Predictions"  # 推理结果输出根目录
MODEL_INPUT_SIZE = (256, 256)  # 训练时设定的输入模型尺寸 (H, W)
# =========================================================================

# 类别映射与十六进制颜色对应表
HEX_TO_ID = {
    "#BB469C": 0,  # 月壤
    "#E8FA50": 1,  # 岩石
    "#7800C8": 2,  # 撞击坑
    "#AD451F": 3,  # 山体
    "#22C9F8": 4,  # 天空
}


def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    # 返回 RGB 顺序的元组
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


# 建立 类别ID -> RGB 颜色 的映射
ID_TO_RGB = {id_val: hex_to_rgb(hex_code) for hex_code, id_val in HEX_TO_ID.items()}


def colorize_mask(pred_gray: np.ndarray) -> np.ndarray:
    """将 0~4 的单通道灰度掩码图转换为 RGB 彩色掩码图"""
    h, w = pred_gray.shape
    mask_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, rgb_color in ID_TO_RGB.items():
        mask_rgb[pred_gray == class_id] = rgb_color
    return mask_rgb


def main():
    if not os.path.exists(ONNX_MODEL_PATH):
        raise FileNotFoundError(f"未找到 ONNX 模型权重文件: {ONNX_MODEL_PATH}，请检查路径。")

    # 初始化 ONNX Runtime 推理会话（优先使用 CPU，若有 GPU 支持可改为 ['CUDAExecutionProvider']）
    print(f"正在加载 ONNX 模型: {ONNX_MODEL_PATH} ...")
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name

    test_path = Path(TEST_DATA_ROOT)
    if not test_path.exists():
        raise FileNotFoundError(f"测试集根目录不存在: {TEST_DATA_ROOT}")

    # 获取所有测试场景文件夹
    scenes = sorted([d for d in test_path.iterdir() if d.is_dir()])
    print(f"共检测到 {len(scenes)} 个测试场景: {[s.name for s in scenes]}")

    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    for scene_dir in scenes:
        scene_name = scene_dir.name
        color_dir = scene_dir / "color"
        if not color_dir.exists():
            color_dir = scene_dir  # 兼容直接存放在场景目录下的情况

        img_paths = sorted(list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg")))
        if not img_paths:
            continue

        # 为每个场景建立对应的输出子目录
        scene_out_dir = out_root / scene_name
        gray_out_dir = scene_out_dir / "label_gray"  # 8位单通道灰度掩码图保存目录
        rgb_out_dir = scene_out_dir / "label_rgb"  # RGB彩色掩码图保存目录

        gray_out_dir.mkdir(parents=True, exist_ok=True)
        rgb_out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> 正在对场景进行推理: {scene_name} (共 {len(img_paths)} 张图像)")

        for img_path in tqdm(img_paths, desc=scene_name):
            sample_id = img_path.stem

            # 1. 读取原图
            img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if img_bgr is None:
                print(f"    [警告] 无法读取图片: {img_path}")
                continue
            orig_h, orig_w = img_bgr.shape[:2]

            # 2. 图像预处理：缩放到模型输入尺寸
            img_resized = cv2.resize(img_bgr, (MODEL_INPUT_SIZE[1], MODEL_INPUT_SIZE[0]),
                                     interpolation=cv2.INTER_LINEAR)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

            # 归一化并转换为 (1, 3, H, W) 张量格式
            img_tensor = img_rgb.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_tensor, (2, 0, 1))  # (3, H, W)
            img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, 3, H, W)

            # 3. ONNX 模型前向推理
            ort_inputs = {input_name: img_tensor}
            ort_outs = ort_session.run(None, ort_inputs)
            logits = ort_outs[0]  # 输出形如 (1, Num_Classes, H_model, W_model)

            # 4. 后处理：利用 argmax 获取每个像素预测的类别索引 (0~4)
            pred_map = np.argmax(logits, axis=1)[0].astype(np.uint8)

            # 5. 尺寸还原：将推理出的掩码图通过最近邻插值恢复为测试原图的尺寸
            pred_orig_size = cv2.resize(pred_map, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

            # 6. 生成对应的两张图像
            # (1) 8位单通道灰度掩码图 (像素值为 0~4)
            gray_mask = pred_orig_size

            # (2) RGB 彩色掩码图
            rgb_mask = colorize_mask(pred_orig_size)
            rgb_mask_bgr = cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR)  # OpenCV 保存时转为 BGR

            # 7. 保存结果
            cv2.imwrite(str(gray_out_dir / f"{sample_id}.png"), gray_mask)
            cv2.imwrite(str(rgb_out_dir / f"{sample_id}.png"), rgb_mask_bgr)

    print(f"\n所有测试场景推理完毕！结果已保存在目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()