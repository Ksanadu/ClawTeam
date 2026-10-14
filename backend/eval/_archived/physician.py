"""主治医生角色智能体。"""

from graph.roles.base_role import RoleAgent, RoleType


class PhysicianAgent(RoleAgent):
    role_type = RoleType.PHYSICIAN
    role_label = "主治医生"
    prompt_file = "PHYSICIAN.md"
