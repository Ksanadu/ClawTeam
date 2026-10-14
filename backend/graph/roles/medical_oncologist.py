"""肿瘤内科医生角色（Tumor Board）。

⭐ 训练角色：可加载 Qwen3-4B + LoRA adapter
"""

from graph.roles.base_role import RoleAgent, RoleType


class MedicalOncologistAgent(RoleAgent):
    role_type = RoleType.MEDICAL_ONCOLOGIST
    role_label = "肿瘤内科医生"
    prompt_file = "MEDICAL_ONCOLOGIST.md"
