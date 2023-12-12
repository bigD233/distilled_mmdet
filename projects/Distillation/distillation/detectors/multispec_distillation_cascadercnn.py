# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .multispec_three_stage_detector import MultiSpecDistillDetector
from .two_stage_fgd import TwoStageFGDDetector


@MODELS.register_module()
class MultiSpecDistillCascadeRCNN(MultiSpecDistillDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone: ConfigType,
                 enable_distilled ,
                 distilled_file_config,
                 distilled_checkpoint,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            enable_distilled = enable_distilled,
            distilled_file_config = distilled_file_config,
            distilled_checkpoint = distilled_checkpoint,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
