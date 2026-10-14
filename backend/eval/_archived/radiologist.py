"""影像科医生角色智能体。"""

from graph.roles.base_role import RoleAgent, RoleType


class RadiologistAgent(RoleAgent):
    role_type = RoleType.RADIOLOGIST
    role_label = "影像科医生"
    prompt_file = "RADIOLOGIST.md"
