"""放疗科医生角色（Tumor Board）。"""

from graph.roles.base_role import RoleAgent, RoleType


class RadiationOncologistAgent(RoleAgent):
    role_type = RoleType.RADIATION_ONCOLOGIST
    role_label = "放疗科医生"
    prompt_file = "RADIATION_ONCOLOGIST.md"
