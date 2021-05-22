Blogpost referencing the code present in this repository: https://medium.com/analytics-vidhya/building-hari-puttars-missing-invisibility-cloak-with-opencv-1d9ebbb73395 

## Very Brief Introduction:

* This repository has code that creates an "Invisibility Cloak" using openCV's Python wrapper.
* The basic idea is to have a reference-background image as well as the color of the supposed invisibility cloak
* Then, we read the video from the webcam and replace the pixels near the set-invisibility cloak color & replace those pixels with the corresponding pixels from the background image.
* I've written some GUI code for better usability.

## Usage:

$python run.py PATH\TO\BG\IMAGE.jpg

-> there is a "green.png" which can be used as a sample background

Program will provide some statistics and prompts before opening the display window.

On the display window, click on the colors you want to remove; try to select all shades of the color you want to remove. (Too many clicks will decrease the speed!)

If you see the mask is being over-estimated, use the threshold track-bar. Decrease the threshold to reduce the span of the invisibility.


Hope this helps you!
_/|\_ 
