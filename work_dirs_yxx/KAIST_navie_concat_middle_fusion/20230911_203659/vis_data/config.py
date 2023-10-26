auto_scale_lr = dict(base_batch_size=16, enable=True)
backend_args = None
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.BAANet.baanet',
    ])
data_root = ''
dataset_type = 'KAISTDataset'
default_hooks = dict(
    checkpoint=dict(by_epoch=True, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = './projects\\BAANet\\checkpoints\\best_coco_bbox_mAP_epoch_9.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
model = dict(
    backbone=dict(
        depth=50,
        frozen_stages=1,
        init_cfg=dict(checkpoint='torchvision://resnet50', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        style='pytorch',
        type='MultiSpecResNets',
        weight_path=
        'D:/大四/实验室/mmdetection/projects/BAANet/checkpoints/resnet50.pth'),
    data_preprocessor=dict(
        mean=[
            103.53,
            116.28,
            123.675,
            135.438,
            135.438,
            135.438,
        ],
        pad_size_divisor=32,
        std=[
            57.375,
            57.12,
            58.395,
            1.0,
            1.0,
            1.0,
        ],
        type='BGR3TDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
        bbox_head=dict(
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=1.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=1,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared2FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.41,
                0.41,
                0.41,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=False,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=512,
                pos_fraction=0.25,
                type='RandomSampler')),
        rpn=dict(
            allowed_border=-1,
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='MultiSpecFasterRCNN')
optim_wrapper = dict(
    optimizer=dict(lr=0.001, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            6,
            10,
        ],
        type='MultiStepLR'),
]
plugin = True
plugin_dir = './projects/BAANet/baanet/'
resume = False
test_cfg = dict(type='ValLoop')
test_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        pipeline=[
            dict(backend_args=None, type='LoadBGR3TFromKAIST'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='KAISTDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        format_only=False,
        metric='bbox',
        outfile_prefix='./coco_detection/test',
        type='CocoMetric'),
    dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        format_only=False,
        metric='bbox',
        type='KAISTMissrateMetric'),
]
test_pipeline = [
    dict(backend_args=None, type='LoadBGR3TFromKAIST'),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=4,
    dataset=dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/train_anno/KAIST_train_RGB_annotation.json',
        backend_args=None,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadBGR3TFromKAIST'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='KAISTDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadBGR3TFromKAIST'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        512,
    ), type='Resize'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=4,
    dataset=dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        pipeline=[
            dict(backend_args=None, type='LoadBGR3TFromKAIST'),
            dict(keep_ratio=True, scale=(
                640,
                512,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='KAISTDataset'),
    drop_last=False,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        format_only=False,
        metric='bbox',
        outfile_prefix='./coco_detection/test',
        type='CocoMetric'),
    dict(
        ann_file=
        '/media/yons/1/yxx/grad_proj_data/KAIST/anno/test_anno/KAIST_test_RGB_annotation.json',
        backend_args=None,
        format_only=False,
        metric='bbox',
        type='KAISTMissrateMetric'),
]
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = 'D:\\大四\\实验室\\mmdetection\\work_dirs_yxx\\KAIST_navie_concat_middle_fusion'
