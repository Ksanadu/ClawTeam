"""ClawTeam Tumor Board 角色智能体模块（v3.1）。

4 个真专科角色：
- Pathologist (病理科)
- Surgeon (肿瘤外科)            ⭐ LoRA 训练
- MedicalOncologist (肿瘤内科)   ⭐ LoRA 训练
- RadiationOncologist (放疗科)
"""

from graph.roles.base_role import RoleAgent, RoleOpinion, RoleType
from graph.roles.pathologist import PathologistAgent
from graph.roles.surgeon import SurgeonAgent
from graph.roles.medical_oncologist import MedicalOncologistAgent
from graph.roles.radiation_oncologist import RadiationOncologistAgent

__all__ = [
    "RoleAgent",
    "RoleOpinion",
    "RoleType",
    "PathologistAgent",
    "SurgeonAgent",
    "MedicalOncologistAgent",
    "RadiationOncologistAgent",
]
