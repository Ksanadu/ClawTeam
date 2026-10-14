"""肿瘤外科医生角色（Tumor Board）。

⭐ 训练角色：可加载 Qwen3-4B + LoRA adapter
"""

from graph.roles.base_role import RoleAgent, RoleType


class SurgeonAgent(RoleAgent):
    role_type = RoleType.SURGEON
    role_label = "肿瘤外科医生"
    prompt_file = "SURGEON.md"
