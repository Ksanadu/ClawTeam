"""肿瘤相关关键词，用于从通用医学 benchmark 中筛选肿瘤题。"""

from __future__ import annotations

# 中文肿瘤关键词
ZH_ONCOLOGY_KEYWORDS = [
    "肿瘤", "癌", "恶性", "良性肿瘤", "转移", "复发", "化疗", "放疗",
    "靶向", "免疫治疗", "新辅助", "辅助治疗",
    "腺癌", "鳞癌", "小细胞癌", "肉瘤", "淋巴瘤", "白血病",
    "肺癌", "胃癌", "肝癌", "结肠癌", "直肠癌", "乳腺癌", "卵巢癌",
    "前列腺癌", "胰腺癌", "食管癌", "甲状腺癌", "膀胱癌",
    "TNM", "分期", "EGFR", "ALK", "PD-L1", "HER2", "KRAS",
]

# 英文肿瘤关键词
EN_ONCOLOGY_KEYWORDS = [
    "cancer", "tumor", "tumour", "carcinoma", "sarcoma", "lymphoma", "leukemia",
    "malignant", "metastasis", "metastatic", "chemotherapy", "radiotherapy",
    "neoadjuvant", "adjuvant", "targeted therapy", "immunotherapy",
    "adenocarcinoma", "squamous cell", "small cell",
    "lung cancer", "gastric cancer", "liver cancer", "colon cancer",
    "breast cancer", "prostate cancer", "pancreatic cancer",
    "TNM", "staging", "EGFR", "ALK", "PD-L1", "HER2", "KRAS",
    "oncology", "oncologic",
]


def is_oncology_zh(text: str) -> bool:
    """判断中文文本是否与肿瘤相关。"""
    return any(kw in text for kw in ZH_ONCOLOGY_KEYWORDS)


def is_oncology_en(text: str) -> bool:
    """判断英文文本是否与肿瘤相关。"""
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in EN_ONCOLOGY_KEYWORDS)


def is_oncology(text: str) -> bool:
    """中英文都检查。"""
    return is_oncology_zh(text) or is_oncology_en(text)


# 角色相关关键词（v3.3 大幅扩展，提升真实数据召回率）
SURGEON_KEYWORDS = [
    # 基础手术
    "手术", "切除", "切口", "术中", "术后", "围手术期", "可切除", "不可切除",
    "淋巴清扫", "根治术", "姑息术", "腹腔镜", "胸腔镜", "微创", "开放手术",
    "肺叶切除", "全肺切除", "胃大部切除", "结肠切除", "肝切除", "胰十二指肠",
    "乳腺切除", "甲状腺切除", "前列腺根治", "肾切除", "膀胱切除",
    # 围手术期
    "麻醉", "出血", "并发症", "再发", "术前评估", "切缘", "术中冰冻", "新辅助",
    "Clavien", "ECOG", "ASA", "分期手术",
    # 病理切除
    "R0", "R1", "切缘阴性", "切缘阳性",
    # 英文
    "surgery", "surgical", "resection", "operation", "operative",
    "lymphadenectomy", "laparoscopic", "thoracoscopic", "lobectomy",
    "gastrectomy", "hepatectomy", "mastectomy", "perioperative",
]

MEDICAL_ONCOLOGIST_KEYWORDS = [
    # 化疗
    "化疗", "化学治疗", "联合化疗", "单药化疗", "维持化疗",
    "顺铂", "卡铂", "紫杉醇", "多西他赛", "培美曲塞", "吉西他滨", "氟尿嘧啶",
    "奥沙利铂", "伊立替康", "环磷酰胺", "阿霉素", "长春瑞滨", "依托泊苷",
    # 靶向
    "靶向", "靶向治疗", "EGFR", "ALK", "ROS1", "BRAF", "HER2", "VEGF",
    "吉非替尼", "厄洛替尼", "奥希替尼", "克唑替尼", "阿来替尼", "曲妥珠单抗",
    "贝伐珠单抗", "西妥昔单抗", "帕妥珠单抗", "T-DM1", "DS-8201",
    # 免疫
    "免疫", "免疫治疗", "免疫检查点", "PD-1", "PD-L1", "CTLA-4",
    "帕博利珠单抗", "纳武利尤单抗", "信迪利单抗", "替雷利珠单抗", "卡瑞利珠单抗",
    # 内分泌/激素
    "内分泌治疗", "激素治疗", "他莫昔芬", "芳香化酶抑制剂", "氟维司群",
    # 治疗策略
    "新辅助治疗", "辅助治疗", "维持治疗", "姑息治疗", "一线治疗", "二线治疗",
    "同步放化疗", "序贯", "ORR", "PFS", "OS", "DCR",
    # 不良反应
    "骨髓抑制", "粒细胞减少", "血小板减少", "肝肾功能", "心脏毒性", "周围神经病变",
    # 英文
    "chemotherapy", "targeted therapy", "immunotherapy", "cisplatin",
    "paclitaxel", "gefitinib", "erlotinib", "pembrolizumab", "trastuzumab",
    "bevacizumab", "neoadjuvant", "adjuvant",
]


def is_surgeon_topic(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in SURGEON_KEYWORDS)


def is_oncologist_topic(text: str) -> bool:
    text_lower = text.lower()
    return any(kw.lower() in text_lower for kw in MEDICAL_ONCOLOGIST_KEYWORDS)
