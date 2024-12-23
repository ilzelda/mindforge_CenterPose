@echo off
chcp 65001 > nul

set model_path=C:\Users\mindforge\mindforge_CenterPose\exp\object_pose\objectron_custom_box_dlav1_34_2024-10-23-06-05\custom_box_best.pth

rem 모델 경로에서 모델 이름 추출
for %%C in ("%model_path%") do (
    set "model_dir=%%~dpC"
)
for %%D in ("%model_dir:~0,-1%") do (
    set "model_name=%%~nxD"
)

echo 모델 이름: %model_name%

rem 비디오 파일 경로 설정
setlocal enabledelayedexpansion
set video_paths=

for %%F in ("C:\Users\mindforge\custom_dataset\box\*.mp4") do (
    set video_paths=!video_paths! "%%F"
)

for %%A in (!video_paths!) do (
    echo 현재 처리 중인 파일: %%A

    rem 파일명에서 경로 제거
    set file_name=%%~nxA

    rem 파일명에서 확장자 제거
    set file_name_without_ext=%%~nA

    echo 비디오 파일 이름: !file_name_without_ext!

    python demo_custom.py --demo %%A --load_model %model_path% --output_dir %model_name% --rep_mode 1

    rem FFmpeg 명령어 실행
    ffmpeg -framerate 30 -i .\demo_output\%model_name%\frame_%%04d.png -c:v libx264 -pix_fmt yuv420p .\demo_output\%model_name%\!file_name_without_ext!.mp4
    del .\demo_output\%model_name%\frame_*.png
)

echo 처리가 완료되었습니다.
