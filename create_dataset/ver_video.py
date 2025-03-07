import cv2


def display_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video")
        return

    # Read and display video frames
    while cap.isOpened():
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            print("End of video")
            break

        # Display the frame
        cv2.imshow('Video', frame)

        # Press 'q' to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = "./videos_nuevo/video_original.avi"  # Replace with your video path
display_video(video_path)
