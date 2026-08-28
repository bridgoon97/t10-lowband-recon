"""Model registry — build any arm from a config dict."""
from .arm_a_ddsp import ArmA_DDSP
from .arm_b_crn import ArmB_CRN
from .arm_c_ftlstm import ArmC_FTLSTM

ARMS = {
    "arm_a_ddsp": ArmA_DDSP,
    "arm_b_crn": ArmB_CRN,
    "arm_c_ftlstm": ArmC_FTLSTM,
}


def build_model(cfg: dict):
    arm = cfg["arm"]
    if arm not in ARMS:
        raise KeyError(f"unknown arm '{arm}'; choose from {list(ARMS)}")
    return ARMS[arm](cfg)
