# What does this run.py do?
# 1. Loads first CLI(if provided) as background image else helps user to capture background at run-time.
# 2. We build the namedWindow called "Magic", add a track-bar, information-bar and mouse-event listener to it.
#       a. track-bar allowes user to set the threshold for color-matching
#       b. Info bar gives some instructions
#       c. mouse listener allows user to click on any point in image capture that colors. Mouse listener helps maintain a list of all "clicked" colurs
# 3. For each new frame(captured from INPUT_SOURCE = webcam), we remove the pixels that are within the threshold with any of the "clicked" colours. We replace these pixels with correspondong pixels from the background image! This is how the vanishing effect is generated
# 3a. We use some tricks here and there to get the best ranges for the threshold, some morphology to make the output more pretty. All this is documented in code.

# Let's import our dependencies!
import os, sys
import time
import numpy as np
import cv2

# Since Python doesn't have pointers like C++, we need to have global variables and pass copy of their references to functions who might change them.
# More details at:  https://stackoverflow.com/questions/986006/how-do-i-pass-a-variable-by-reference
colors_list_hsv = []
plot_data_list = [colors_list_hsv, None]

# Listener for mouse-event
def on_mouse(event, x, y, _, plot_data_list):
    # Do something on single-click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Ignore if the user clicks on the InfoBar.. seriously.. WHAT?
        if y > INFOBAR_HEIGHT:
            # We capture the pixel-colors at click position from the frame and add it to our global list; we save all colours in HSV format.
            frame = plot_data_list[1]
            color = frame[y][x]
            color_hsv = cv2.cvtColor(np.array([[color]], np.uint8), cv2.COLOR_BGR2HSV)
            plot_data_list[0].append(color_hsv)

if __name__ == "__main__":
    # we take path to BG image as CLI
    args = sys.argv
    if len(args) != 2:
        raise Exception("Proper call: $python run.py PATH/TO/BG/IMAGE or 'AUTO'")
    BG_IMAGE_PATH = args[1]
    
    # All these global constants can have main as their scope
    WARMUP_TIME = 5             # in seconds
    MIN_AREA_OBJECT = 500       # It is used for noise filtration, crank it up if you find more noise and less signal :)

    WINDOW_NAME = "Magic"       # Name of the output window
    TRACKBAR_NAME = "threshold" # no need to change, name of trackbar component
    thresh = 30                 # Initial value for threshold
    INFOBAR_HEIGHT = 100        #Height of the Information-bar, WIDTH = FRAME_WIDTH

    #Output file properties
    OUTFILE = "output.mp4"
    FRAME_FPS = 10.0    
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480          #Output file dimension = FRAME_HEIGHT + TRACKBAR_HEIGHT

    back_ground_image = None    #initialize the background variable
    OUT_BGFILE = "BG_image.png" #Path to save background file if you want to use it again
    BUILD_FRAMES = 60           #Number of frames used to build background at runtime

    INPUT_SOURCE = 0            # 0 = webcam,add file-path if you want to read from file
    
    kernel_33 = np.ones((3,3), dtype = np.uint8) # Kernel for morphology
    
    # First we check if input_source is accessible properly
    cap = cv2.VideoCapture(INPUT_SOURCE)
    print("Wait! We're heating up the camera for ", WARMUP_TIME, "seconds!")
    time.sleep(WARMUP_TIME)     # Give DELAY to warm-up in case you have silly computer
    ret_val, frame = cap.read()
    if not ret_val:
        raise Exception("Issue with input source. Does specified input source exist?")

    # Prompt the user
    print("Input being fetched from source:", str(INPUT_SOURCE))
    print("Input Frame size:", str(frame.shape[1]), str(frame.shape[0]))
    print("Output being written to:", str(OUTFILE))
    print("Output Frame size:", str(FRAME_WIDTH), str(FRAME_HEIGHT))
    
    #Now we make sure that we have a kick-ass background for magic!
    if os.path.isfile(BG_IMAGE_PATH):
        back_ground_image = cv2.imread(BG_IMAGE_PATH)
        back_ground_image = cv2.resize(back_ground_image, (FRAME_WIDTH, FRAME_HEIGHT))
    else:
        print("We've to build a kick-ass background.")
        print("Move away from the camera-view!")
        _ = input("Enter 'c' to continue!")
        print("Building the afore-mentioned kick-ass background ;)")
        collector = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.int64)
        for temp in range(BUILD_FRAMES):
            _, frame = cap.read()
            collector = np.add(collector, frame)
        back_ground_image = (collector / BUILD_FRAMES).astype(np.uint8)
        cv2.imwrite(OUT_BGFILE, back_ground_image)
        print("generated back-ground image stored at", OUT_BGFILE, "; you can use it next time!")

    #Writing to video file in openCV is highly machine dependents, Below given codes work with windows10 and ubuntu 16.04 LTS(mostly); do raise an issue if this doesn't work for you. If you dont have proper codec, mp4 file will be generated but nothing will be writen into it.
    if sys.platform == "win32":
        codec = cv2.VideoWriter_fourcc(*'DIVX')
    elif sys.platform in ["linux", "linux2"]:
        codec = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(OUTFILE, codec, FRAME_FPS, (FRAME_WIDTH, FRAME_HEIGHT+INFOBAR_HEIGHT))

    # Let's create some GUI objects
    window = cv2.namedWindow(WINDOW_NAME)
    cv2.createTrackbar(TRACKBAR_NAME, WINDOW_NAME, thresh, 100,  lambda x:None)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse, plot_data_list)
    
    frame_number = 0
    while True:
        # Read the frame, resize it and smoothen it a little
        _, frame = cap.read()
        thresh = cv2.getTrackbarPos(TRACKBAR_NAME, WINDOW_NAME)
        result = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        result = cv2.medianBlur(result, 5)

        # we create the info-bar, we'll patch it on last though
        info_bar = np.zeros((INFOBAR_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.putText(img = info_bar, text = "--Information Bar--",
        org = (25, 20), fontFace = 1,
        fontScale = 1, color = (255, 255, 255)) 
        cv2.putText(img = info_bar, text = "Click on you 'invisiblity cloak' below!",
        org = (25, 40), fontFace = 1,
        fontScale = 1, color = (255, 255, 255)) 
        cv2.putText(img = info_bar, text = "Click all different shades for best results, multiple clicks are allowed. Tune the threshold to get best result.",
        org = (25, 60), fontFace = 1,
        fontScale = 1, color = (255, 255, 255)) 
        cv2.putText(img = info_bar, text = "Press 'q' to exit.",
        org = (25, 80), fontFace = 1,
        fontScale = 1, color = (255, 255, 255)) 
        
        # Initialize the mask for this frame
        curr_mask = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 1), dtype=np.uint8)
        for hsv_color in colors_list_hsv:
            # See how we find the lower and upper bounds for each colors, that's where the magic is!
            hsv_color = hsv_color[0, 0]
            lower = np.copy(hsv_color)
            upper = np.copy(hsv_color)
            
            lower[0] = max(lower[0] - thresh//2, 0)
            lower[1] = max(lower[1] - thresh, 0)
            lower[2] = max(lower[2] - thresh, 0)
            
            upper[0] = min(upper[0] + thresh//2, 179)
            upper[1] = min(upper[1] + thresh,255)
            upper[2] = min(upper[2] + thresh,255)

            #using HSV colorspace and inRange function to do the magic.  
            result_hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(result_hsv, lower, upper)

            # accumulating all colors into one mask
            curr_mask = cv2.bitwise_or(mask, curr_mask)

        # Morphological Opening to remove the small noise
        curr_mask = cv2.dilate(curr_mask, kernel_33, iterations=5)
        _, curr_mask = cv2.threshold(curr_mask, 127, 255, cv2.THRESH_BINARY)
        curr_mask = cv2.erode(curr_mask, kernel_33, iterations=3)

        # Then we find the contours of all "invisible objects"
        contours, hierarchy = cv2.findContours(curr_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
        for contour in contours:
            if cv2.contourArea(contour) < MIN_AREA_OBJECT:
                cv2.drawContours(curr_mask, [contour], 0, (0, 0, 0), thickness=cv2.FILLED)        

        # Morphological Closinging to remove the small noise
        curr_mask = cv2.erode(curr_mask, kernel_33, iterations=3)
        curr_mask = cv2.dilate(curr_mask, kernel_33, iterations=5)
        _, curr_mask = cv2.threshold(curr_mask, 127, 255, cv2.THRESH_BINARY)
                
        # If there are any objects that we must remove from the scene, we remove them
        # This technique is taken from wonder OpenCV tutorial available at: https://docs.opencv.org/4.1.2/d0/d86/tutorial_py_image_arithmetics.html 
        if len(colors_list_hsv) != 0:
            bg = cv2.bitwise_and(back_ground_image,back_ground_image, mask = curr_mask)
            curr_mask_inv = cv2.bitwise_not(curr_mask)
            result = cv2.bitwise_and(result,result, mask = curr_mask_inv)
            result = cv2.add(result, bg)

        # we addd the info-bar and make some variable updates for the mouse listener
        result = np.vstack((info_bar, result))
        result_unmodified = np.vstack((info_bar, frame))
        plot_data_list[1] = result_unmodified[:]

        # we show the video, write it to file,update the counter and check if 'q' is pressed
        cv2.imshow(WINDOW_NAME, result)
        out.write(result)
        frame_number+=1
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            #If q pressed, we break-dance B)
            break

    # releasing resources before we end execution
    cv2.destroyAllWindows()
    cap.release()
    out.release()

## _/|\_ May the Patronous charm ward of all evil that comes your way. _/|\_