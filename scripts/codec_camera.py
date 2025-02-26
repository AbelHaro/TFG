# import the necessary packages
from deffcode import FFdecoder
import cv2
import os

# define suitable FFmpeg parameter
ffparams = {
    "-vcodec": "h264_nvmpi",  # use H.264 CUVID Video-decoder
    "-enforce_cv_patch": True # enable OpenCV patch for YUV(YUV420p) frames
}

video_path = "../datasets_labeled/videos/prueba_velocidad_07.mp4"

if not os.path.exists(video_path):
    print("Error: No se puede encontrar el archivo de video.")
    exit()

# initialize and formulate the decoder with `foo.mp4` source
decoder = FFdecoder(
    video_path,  # path to the video
    frame_format="yuv420p",  # use YUV420p frame pixel format
    verbose=True, # enable verbose output
    **ffparams # apply various params and custom filters
).formulate()

# grab the YUV420p frame from the decoder
for frame in decoder.generateFrame():

    # check if frame is None
    if frame is None:
        break

    # convert it to `BGR` pixel format,
    # since imshow() method only accepts `BGR` frames
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)

    # {do something with the BGR frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# close output window
cv2.destroyAllWindows()

# terminate the decoder
decoder.terminate()