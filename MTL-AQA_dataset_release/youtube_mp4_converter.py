import pandas as pd
import os
from pytube import YouTube

def youtube_mp3_converter(youtube_link, download_path, video_num):
	yt = YouTube(youtube_link)
	video = yt.streams.first()
	out_file = video.download(download_path)
	new_file = download_path + '/' + str(video_num) + '.mp4'
	os.rename(out_file, new_file)
	print("finished youtube to mp4 download of video : ", video.title)

# parse video list
video_list_path = "MTL-AQA/MTL-AQA_dataset_release/Video_List.xlsx"
youtube_links_df = pd.read_excel(video_list_path, sheet_name=None)
sheet = youtube_links_df['Sheet1']
video_nums = sheet['Sr. No.']
video_links = sheet['Video']
indices = len(video_nums)
video_num_link_map = {}
for idx in range(len(video_nums)):
	print(str(video_nums[idx]))
	video_num_link_map[video_nums[idx]] = video_links[idx]

downloaded_videos_path = "whole_videos"

# download videos
for video_num in video_num_link_map:
	video_link = video_num_link_map[video_num]
	youtube_mp3_converter(video_link, downloaded_videos_path, video_num)

