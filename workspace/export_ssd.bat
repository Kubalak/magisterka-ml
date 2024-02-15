call "D:/Dokumenty/Dokumenty Kuba/Studia/magisterka/tf2_obj_api/Scripts/activate.bat"
call python exporter_main_v2.py --pipeline_config_path="models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8/pipeline.config" --trained_checkpoint_dir="models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8" --output_directory="exported_models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8" --input_type=image_tensor
pause