call "D:/Dokumenty/Dokumenty Kuba/Studia/magisterka/tf2_obj_api/Scripts/activate.bat"
call python exporter_main_v2.py --pipeline_config_path="models/efficientdet_d0_coco17_tpu-32/pipeline.config" --trained_checkpoint_dir="models/efficientdet_d0_coco17_tpu-32" --output_directory="exported_models/efficientdet_d0_coco17_tpu-32" --input_type=image_tensor
pause