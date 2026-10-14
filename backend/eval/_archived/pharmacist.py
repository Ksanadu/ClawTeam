"""药师角色智能体。"""

from graph.roles.base_role import RoleAgent, RoleType


class PharmacistAgent(RoleAgent):
    role_type = RoleType.PHARMACIST
    role_label = "临床药师"
    prompt_file = "PHARMACIST.md"
