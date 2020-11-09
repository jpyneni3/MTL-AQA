for i in 1 2 3 4 5 6 7 9 10 13 14 17 18 22 26;
do
mkdir whole_videos_frames/$i
ffmpeg -i whole_videos/$i.mp4 whole_videos_frames/$i/%06d.jpg
done
