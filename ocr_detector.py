import cv2
import numpy as np
import easyocr

class PaddleOCRDetector:
    """
    OCR 检测类（基于 EasyOCR），接口与原 PaddleOCR 版本保持一致
    增强功能：异常处理、GPU自动检测、自适应参数、结果缓存
    """
    def __init__(self, lang='ch', use_gpu=None):
        """
        初始化 EasyOCR
        :param lang: 'ch' 表示中文（简繁+英文），'en' 表示英文
        :param use_gpu: 是否使用 GPU（None=自动检测，True/False=手动指定）
        """
        try:
            # 自动检测GPU
            if use_gpu is None:
                try:
                    import torch
                    use_gpu = torch.cuda.is_available()
                except ImportError:
                    use_gpu = False
                    print("警告：未安装 torch，无法检测 GPU，将使用 CPU")
            
            if lang == 'ch':
                lang_list = ['ch_sim', 'en']   # 简体中文 + 英文
            else:
                lang_list = ['en']
            
            self.reader = easyocr.Reader(
                lang_list, 
                gpu=use_gpu,
                # 性能优化参数
                model_storage_directory=None,  # 使用默认模型目录
                download_enabled=True,  # 允许自动下载模型
                detector=True,
                recognizer=True
            )
            
            self.lang = lang
            self.use_gpu = use_gpu
            self._cache = {}  # 结果缓存：{image_hash: results}
            self._recognition_params = self._get_default_params()
            
            print(f"✅ EasyOCR 初始化完成 - 语言: {lang}, GPU: {use_gpu}")
            
        except Exception as e:
            print(f"❌ EasyOCR 初始化失败: {e}")
            raise RuntimeError(f"OCR 检测器初始化失败: {e}")

    def _get_default_params(self):
        """获取默认识别参数"""
        return {
            'paragraph': False,      # 不合并文本段落
            'min_size': 10,          # 最小文本区域
            'text_threshold': 0.6,   # 文本检测阈值
            'low_text': 0.3,         # 低文本阈值
            'link_threshold': 0.3,   # 链接阈值
            'canvas_size': 2560,     # 画布大小
            'mag_ratio': 1.5,        # 放大比例
            'slope_ths': 0.1,        # 斜率阈值
            'ycenter_ths': 0.5,      # Y中心阈值
            'height_ths': 0.5,       # 高度阈值
            'width_ths': 0.5,        # 宽度阈值
            'add_margin': 0.1,       # 添加边距
        }

    def _adapt_params_to_image(self, img_shape):
        """根据图像尺寸自适应调整参数"""
        h, w = img_shape[:2]
        params = self._recognition_params.copy()
        
        # 小图像降低阈值以提高检测率
        if max(h, w) < 500:
            params['text_threshold'] = 0.5
            params['low_text'] = 0.25
            params['mag_ratio'] = 2.0
        # 大图像提高画布大小
        elif max(h, w) > 2000:
            params['canvas_size'] = 3840
            params['mag_ratio'] = 1.0
        
        return params

    def predict(self, image_input, save_path=None, show_result=False, 
                use_cache=True, return_plain_text=False):
        """
        对输入图片进行文字检测与识别
        :param image_input: 图片路径 (str) 或 numpy 数组 (np.ndarray)
        :param save_path: 标注结果图片保存路径，None 则不保存
        :param show_result: 是否打印识别结果
        :param use_cache: 是否使用缓存（默认 True）
        :param return_plain_text: 是否返回纯文本列表（默认 False）
        :return: 结果列表，格式 [[box, (text, confidence)], ...]
                 如果 return_plain_text=True，返回 (formatted_result, text_list)
        """
        try:
            # 读取图片
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    raise FileNotFoundError(f"无法读取图片 {image_input}")
                cache_key = image_input
            elif isinstance(image_input, np.ndarray):
                img = image_input.copy()
                cache_key = hash(img.tobytes())  # 使用图像内容的哈希作为缓存键
            else:
                raise ValueError("输入类型必须为图片路径 (str) 或 numpy 数组")

            # 检查缓存
            if use_cache and cache_key in self._cache:
                formatted_result = self._cache[cache_key]
                if show_result:
                    print("✅ 使用缓存结果")
            else:
                # 自适应参数
                params = self._adapt_params_to_image(img.shape)
                
                # 执行 OCR（返回 bbox, text, confidence）
                result_raw = self.reader.readtext(img, **params)

                # 转换为统一格式
                formatted_result = []
                for (bbox, text, confidence) in result_raw:
                    # bbox 是浮点数列表，转为整数
                    box = [list(map(int, point)) for point in bbox]
                    formatted_result.append([box, (text, confidence)])

                # 缓存结果
                if use_cache:
                    self._cache[cache_key] = formatted_result
                    if len(self._cache) > 100:  # 限制缓存大小
                        self._cache.pop(next(iter(self._cache)))

            # 打印结果
            if show_result:
                if formatted_result:
                    print("\n========== OCR 识别结果 ==========")
                    for i, item in enumerate(formatted_result):
                        box = item[0]
                        text = item[1][0]
                        confidence = item[1][1]
                        print(f"  区域 {i+1}: 文字='{text}', 置信度={confidence:.2f}")
                        print(f"      坐标: {box}")
                    print("====================================\n")
                else:
                    print("未识别到任何文字。")

            # 绘制并保存
            if formatted_result and save_path is not None:
                self._draw_with_opencv(img, formatted_result, save_path)
                print(f"标注结果已保存至: {save_path}")

            # 返回结果
            if return_plain_text:
                text_list = [item[1][0] for item in formatted_result]
                return formatted_result, text_list
            
            return formatted_result

        except Exception as e:
            print(f"❌ OCR 识别失败: {e}")
            return None if not return_plain_text else (None, [])

    def _draw_with_opencv(self, img, result, save_path):
        """使用 OpenCV 绘制多边形框和文字（增强版）"""
        try:
            img_copy = img.copy()
            for item in result:
                box = np.array(item[0], dtype=np.int32)
                text = item[1][0]
                confidence = item[1][1]

                # 根据置信度选择颜色（越高越绿，越低越黄）
                if confidence >= 0.8:
                    color = (0, 255, 0)  # 绿色 - 高置信度
                elif confidence >= 0.6:
                    color = (0, 255, 255)  # 黄色 - 中等置信度
                else:
                    color = (0, 165, 255)  # 橙色 - 低置信度

                # 绘制边框
                cv2.polylines(img_copy, [box], isClosed=True, color=color, thickness=2)

                # 添加背景框以增强文字可读性
                x, y = int(box[0][0]), int(box[0][1]) - 5
                font_scale = 0.5 if len(text) > 10 else 0.7
                (text_w, text_h), _ = cv2.getTextSize(
                    f"{text} ({confidence:.2f})", 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale, 2
                )
                
                # 绘制文字背景
                cv2.rectangle(
                    img_copy,
                    (x, y - text_h - 4),
                    (x + text_w + 4, y + 4),
                    (0, 0, 0),
                    -1
                )

                # 显示文字和置信度
                cv2.putText(
                    img_copy,
                    f"{text} ({confidence:.2f})",
                    (x + 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    2
                )

            cv2.imwrite(save_path, img_copy)
            
        except Exception as e:
            print(f"❌ 绘制结果失败: {e}")

    def get_plain_text(self, image_input):
        """便捷方法：直接获取识别文本列表"""
        results = self.predict(image_input, show_result=False, return_plain_text=False)
        if results:
            return [item[1][0] for item in results]
        return []

    def clear_cache(self):
        """清除缓存"""
        self._cache.clear()
        print("✅ OCR 缓存已清除")

    def get_stats(self):
        """获取统计信息"""
        return {
            'language': self.lang,
            'use_gpu': self.use_gpu,
            'cache_size': len(self._cache),
            'is_ready': hasattr(self, 'reader') and self.reader is not None
        }