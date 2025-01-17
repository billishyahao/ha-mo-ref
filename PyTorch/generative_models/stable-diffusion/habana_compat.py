###############################################################################
# Copyright (C) 2023 Habana Labs, Ltd. an Intel Company
###############################################################################

import os

is_mark_step_enabled = False

def setup_hpu(args):
    global is_mark_step_enabled
    if args.device == 'hpu':
        if os.environ.get("PT_HPU_LAZY_MODE") != "2":
            is_mark_step_enabled = True
        import habana_frameworks.torch.core


def mark_step():
    if is_mark_step_enabled:
        import habana_frameworks.torch.core as htcore
        htcore.mark_step()
