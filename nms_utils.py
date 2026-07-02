import numpy as np

class NMSUtils:
    """NMS 非极大值抑制工具类"""
    
    @staticmethod
    def ocr_nms(ocr_results, iou_threshold=0.3, conf_threshold=0.5, merge_threshold=0.7):
        """
        对OCR结果进行NMS抑制
        :param ocr_results: OCR结果列表 [[box, (text, confidence)], ...]
        :param iou_threshold: IoU阈值，大于此值认为是重叠框
        :param conf_threshold: 置信度阈值，低于此值的框被过滤
        :param merge_threshold: IoU大于此值时，合并重叠框
        :return: 过滤后的OCR结果
        """
        if not ocr_results:
            return []
        
        # 过滤低置信度结果
        filtered = [item for item in ocr_results if item[1][1] >= conf_threshold]
        if not filtered:
            return []
        
        # 按置信度降序排序
        filtered.sort(key=lambda x: x[1][1], reverse=True)
        
        keep = []
        while filtered:
            # 取置信度最高的框
            best = filtered.pop(0)
            best_box = NMSUtils._box_to_rect(best[0])
            
            # 检查是否与已保留的框重叠
            merged = False
            for i, kept in enumerate(keep):
                kept_box = NMSUtils._box_to_rect(kept[0])
                iou = NMSUtils._calculate_iou(best_box, kept_box)
                
                if iou > merge_threshold:
                    # 合并重叠框
                    keep[i] = NMSUtils._merge_boxes(best, kept)
                    merged = True
                    break
                elif iou > iou_threshold:
                    # 重叠度高，保留置信度高的
                    merged = True
                    break
            
            if not merged:
                # 检查与剩余框的重叠
                i = 0
                while i < len(filtered):
                    filtered_box = NMSUtils._box_to_rect(filtered[i][0])
                    iou = NMSUtils._calculate_iou(best_box, filtered_box)
                    
                    if iou > iou_threshold:
                        # 移除重叠框
                        filtered.pop(i)
                    else:
                        i += 1
                
                keep.append(best)
        
        return keep
    
    @staticmethod
    def _box_to_rect(box):
        """将多边形框转换为矩形框 [x1, y1, x2, y2]"""
        points = np.array(box)
        x1 = np.min(points[:, 0])
        y1 = np.min(points[:, 1])
        x2 = np.max(points[:, 0])
        y2 = np.max(points[:, 1])
        return [x1, y1, x2, y2]
    
    @staticmethod
    def _calculate_iou(box1, box2):
        """计算两个矩形框的IoU"""
        # 计算交集区域
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # 无交集
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # 交集面积
        intersection = (x2 - x1) * (y2 - y1)
        
        # 并集面积
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    @staticmethod
    def _merge_boxes(box1_info, box2_info):
        """合并两个重叠的文本框"""
        points1 = np.array(box1_info[0])
        points2 = np.array(box2_info[0])
        
        # 合并多边形（取所有点的外接矩形）
        all_points = np.vstack([points1, points2])
        x1 = np.min(all_points[:, 0])
        y1 = np.min(all_points[:, 1])
        x2 = np.max(all_points[:, 0])
        y2 = np.max(all_points[:, 1])
        
        merged_box = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        
        # 选择置信度更高的文字
        if box1_info[1][1] >= box2_info[1][1]:
            merged_text = box1_info[1][0]
            merged_conf = box1_info[1][1]
        else:
            merged_text = box2_info[1][0]
            merged_conf = box2_info[1][1]
        
        return [merged_box, (merged_text, merged_conf)]