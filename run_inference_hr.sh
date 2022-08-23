#echo "3-1"
#python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face /wl_data/cjhumanville_2022/cjhumanville/2_1.mp4 --audio /wl_data/cjhumanville_2022/audio_220709/1.wav --outfile output/temp.mp4
for entry_video in "${1}"/*
do
  video_file=$(basename -- "$entry_video")
  for entry_audio in "${2}"/*
  do
    audio_file=$(basename -- "$entry_audio")
    save_file="A${audio_file%.*}_V${video_file%.*}_NOSR.mp4"
    echo "Save video-${save_file} from audio-${audio_file} and video-${video_file}"
    python inference.py --checkpoint_path "checkpoints/wav2lip.pth" --segmentation_path "checkpoints/face_segmentation.pth" --face /wl_data/cjhumanville_2022/cjhumanville/${video_file} --audio /wl_data/cjhumanville_2022/audio_220709/${audio_file} --outfile output/${save_file} --wav2lip_batch_size 512 --no_sr
  done
done
