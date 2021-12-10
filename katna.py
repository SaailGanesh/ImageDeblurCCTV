import logging
logging.disable(logging.WARNING)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
import sys

# For windows, the below if condition is must.
if __name__ == "__main__":
  fn = sys.argv[1]
  vd = Video()
  no_of_frames_to_returned = 1
  diskwriter = KeyFrameDiskWriter(location="./static/resize_img")
  video_file_path = os.path.join(".", "static", "upload_video", fn)
  print(f"Input video file path = {video_file_path}")
  vd.extract_video_keyframes(
       no_of_frames=no_of_frames_to_returned, file_path=video_file_path,
       writer=diskwriter
  )