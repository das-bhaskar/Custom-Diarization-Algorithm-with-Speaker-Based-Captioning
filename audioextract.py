from moviepy.editor import VideoFileClip

# Load the video file
video_path = "/path/to/filename.mp4"
video_clip = VideoFileClip(video_path)

# Extract the audio
audio_clip = video_clip.audio
audio_clip.write_audiofile("filename.wav")
