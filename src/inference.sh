#!/bin/bash

model_path="/home/CenterPose/exp/object_pose/objectron_custom_box_dlav1_34_2024-10-23-06-05/custom_box_best.pth"

video_paths=($(find "../../custom_dataset/box" -type f -name "*.mp4"))

for demo_path in "${video_paths[@]}"
do
    echo "현재 처리 중인 파일: $demo_path"
    # 파일명에서 경로 제거
    file_name=$(basename "$demo_path")

    # 파일명에서 확장자 제거
    file_name_without_ext="${file_name%.*}"

    model_name="${model_path%/*}"

    model_name="${model_name##*/}"

    python demo_custom.py --demo $demo_path --load_model $model_path --output_dir ${model_name} --rep_mode 4

    # FFmpeg 명령어 실행
    ffmpeg -framerate 30 -i ./demo_output/${model_name}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p ./demo_output/${model_name}/${file_name_without_ext}.mp4
    rm  ./demo_output/${model_name}/frame_*.png
done

echo "처리가 완료되었습니다."