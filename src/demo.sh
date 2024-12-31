#! /bin/bash

# model_path="/mnt/CenterPose_exp/object_pose/objectron_test_box_dlav1_34_2024-12-12-01-05/test_box_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case3_filtered_dlav1_34_2024-12-19-10-08/case3_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_filtered_box_dlav1_34_2024-12-17-08-58/filtered_box_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case4_filtered_dlav1_34_2024-12-23-07-54/case4_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/pretrained_model/cereal_box_v1_140.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_parcel_dlav1_34_2024-11-28-07-08/parcel_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case1_filtered_dlav1_34_2024-12-19-05-26/case1_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case3_filtered_dlav1_34_2024-12-30-07-15/case3_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case6_filtered_dlav1_34_2024-12-26-00-56/case6_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case6_filtered_dlav1_34_2024-12-26-03-15/case6_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case6_filtered_dlav1_34_2024-12-27-04-47/case6_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case6_filtered_dlav1_34_2024-12-30-04-49/case6_filtered_last.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case7_filtered_dlav1_34_2024-12-26-09-17/case7_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case7_filtered_dlav1_34_2024-12-27-06-35/case7_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case8_filtered_dlav1_34_2024-12-27-08-49/case8_filtered_best.pth"
# model_path="/mnt/CenterPose_exp/object_pose/objectron_case9_filtered_dlav1_34_2024-12-31-00-18/case9_filtered_best.pth"
model_path="/mnt/CenterPose_exp/object_pose/objectron_case9_filtered_dlav1_34_2024-12-31-03-56/case9_filtered_best.pth"


model_name="${model_path%/*}"
# echo $model_name
model_name="${model_name##*/}"
# echo $model_name

weight_name="${model_path%.*}"                      # 확장자 제거
weight_name="${weight_name##*/object_pose/}"     # object_pose 이후만 추출
# echo $weight_name

# video_paths=($(find "../../custom_dataset/box" -type f -name "*.mp4"))
video_paths=(../../custom_dataset/box/test2_black_tea_1.mp4 ../../custom_dataset/box/test2_barcode_1.mp4 ../../custom_dataset/box/test_barcode_1.mp4 ../../custom_dataset/box/test2_tea_1.mp4 ../../custom_dataset/box/test_tea_2.mp4 ../../custom_dataset/box/test2_tea_2.mp4 ../../custom_dataset/box/test_black_tea_2.mp4 ../../custom_dataset/box/test2_black_tea_2.mp4 ../../custom_dataset/box/test2_barcode_2.mp4 ../../custom_dataset/box/test2_black_tea_3.mp4 ../../custom_dataset/box/test_barcode_2.mp4 ../../custom_dataset/box/test_tea_1.mp4 ../../custom_dataset/box/test_black_tea_1.mp4)


for demo_path in "${video_paths[@]}"
do
    # echo $demo_path
    echo "현재 처리 중인 파일: $demo_path"
    # 파일명에서 경로 제거
    file_name=$(basename "$demo_path")

    # 파일명에서 확장자 제거
    file_name_without_ext="${file_name%.*}"

    # echo "현재 처리 중인 파일명: $file_name_without_ext"

    # python demo_custom_homography.py --demo $demo_path --load_model $model_path
    python demo_custom_homography.py --demo $demo_path --load_model $model_path --labelling

    # FFmpeg 명령어 실행
    ffmpeg -framerate 30 -i ./demo_output/${weight_name}/${file_name_without_ext}/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -y ./demo_output/${weight_name}/${file_name_without_ext}/${file_name_without_ext}.mp4
    rm  ./demo_output/${weight_name}/${file_name_without_ext}/frame_*.png
    # break
done

echo "처리가 완료되었습니다."