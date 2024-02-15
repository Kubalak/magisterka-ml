call "D:/Dokumenty/Dokumenty Kuba/Studia/magisterka/tf2_obj_api/Scripts/activate.bat"
call python model_main_tf2.py --pipeline_config_path="models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config" --model_dir="models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8" --checkpoint_dir="models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8" --alsologtostderr
pause