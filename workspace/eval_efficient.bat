call "D:/Dokumenty/Dokumenty Kuba/Studia/magisterka/tf2_obj_api/Scripts/activate.bat"
call python model_main_tf2.py --pipeline_config_path="models/efficientdet_d0_coco17_tpu-32/pipeline.config" --model_dir="models/efficientdet_d0_coco17_tpu-32" --checkpoint_dir="models/efficientdet_d0_coco17_tpu-32" --alsologtostderr
pause