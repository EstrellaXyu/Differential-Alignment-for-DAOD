from detectron2.data.datasets import register_coco_instances

# Cityscapes 
register_coco_instances("cityscapes_train", {},         "/home/hxy/datasets/cityscapes_mini/annotations/cityscapes_train_instances.json",                  "/home/hxy/datasets/cityscapes_mini/leftImg8bit/train/")
register_coco_instances("cityscapes_val",   {},         "/home/hxy/datasets/cityscapes_mini/annotations/cityscapes_val_instances.json",                    "/home/hxy/datasets/cityscapes_mini/leftImg8bit/val/")

# Foggy Cityscapes
register_coco_instances("cityscapes_foggy_train", {},   "/home/hxy/datasets/cityscapes_mini/annotations/cityscapes_foggy_train_instances.json",   "/home/hxy/datasets/cityscapes_mini/leftImg8bit_foggy/train/")
register_coco_instances("cityscapes_foggy_val", {},     "/home/hxy/datasets/cityscapes_mini/annotations/cityscapes_foggy_val_instances.json",     "/home/hxy/datasets/cityscapes_mini/leftImg8bit_foggy/val/")

# Sim10k and cityscapes_cars
register_coco_instances("sim10k_cars_train", {},             "/PATH/TO/DATASETS/sim10k/coco_car_annotations.json",                  "/PATH/TO/DATASETS/sim10k/VOC2012/JPEGImages/")
register_coco_instances("cityscapes_cars_val",   {},         "/PATH/TO/DATASETS/cityscapes/annotations/cityscapes_val_instances_cars.json",                    "/PATH/TO/DATASETS/cityscapes/leftImg8bit/val/")

# BDD100k
register_coco_instances("bdd100k_train", {},   "/PATH/TO/DATASETS/bdd100k/annotations/bdd100k_daytime_train_cocostyle.json",   "/PATH/TO/DATASETS/bdd100k/images/100k/train/")
register_coco_instances("bdd100k_val", {},     "/PATH/TO/DATASETS/bdd100k/annotations/bdd100k_daytime_val_cocostyle.json",     "/PATH/TO/DATASETS/bdd100k/images/100k/val/")