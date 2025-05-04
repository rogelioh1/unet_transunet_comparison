from easydict import EasyDict

cfg = EasyDict()

# General training settings
cfg.batch_size = 16
cfg.num_epochs = 5
cfg.learning_rate = 0.01
cfg.img_size = 256

# UNet configuration
cfg.unet = EasyDict()
cfg.unet.n_channels = 3
cfg.unet.n_classes = 1
cfg.unet.bilinear = True

# TransUNet configuration
cfg.transunet = EasyDict()
cfg.transunet.img_dim = 256  # Input image size
cfg.transunet.in_channels = 3  # Input image channels
cfg.transunet.out_channels = 128  # Base feature channels
cfg.transunet.head_num = 4  # Number of attention heads
cfg.transunet.mlp_dim = 512  # MLP dimension in transformer
cfg.transunet.block_num = 8  # Number of transformer blocks
cfg.transunet.patch_dim = 16  # Patch size
cfg.transunet.class_num = 1  # Number of output classes

# YOLOv8 configuration
cfg.yolov8 = EasyDict()
cfg.yolov8.in_channels = 3
cfg.yolov8.n_classes = 1

# Comparison settings
cfg.comparison = EasyDict()
cfg.comparison.metrics = ['iou', 'dice', 'boundary_f1', 'hausdorff']
cfg.comparison.img_sizes = [128, 256, 512]  # Different image sizes to test
cfg.comparison.batch_sizes = [8, 16, 32]  # Different batch sizes to test