echo "01"
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face ./inputs/result_change_speed_50per.mov --audio ./input_audio/01_kss_2.wav --outfile output/hr_01_kss_210618_2.mov
echo "2-1"
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face ./inputs/3-720p_fg_rct.mp4 --audio ./input_audio/01_kss_2.wav --outfile output/hr_01_kss_showhost_210618.mov
echo "2-2"
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face ./inputs/3-720p_fg_rct.mp4 --audio ./input_audio/02_kss.wav --outfile output/hr_02_kss_showhost_210618.mov
echo "2-3"
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face ./inputs/3-720p_fg_rct.mp4 --audio ./input_audio/03_kss.wav --outfile output/hr_03_kss_showhost_210618.mov
echo "2-4"
python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face ./inputs/3-720p_fg_rct.mp4 --audio ./input_audio/04_kss.wav --outfile output/hr_04_kss_showhost_210618.mov