## A classifier to identify people in Instagram images for Looping Power Room

Hello, Matthias! 

The task is done. I already wrote you the short version of the strategy I was using, but here is the extended version:

### Approach

- I wrotea simple function that scrapes images from Instagram, using the links you provided me with. The function names is "load_images_on_disk" and it's inside the looping_instagram_task.Rmd file.

I found 1886 images using that function and downloaded them from Instagram. The link to the images is in "data/downloaded_images" directory. You can download them, but I use only 100 images from there to test the model. The links to the test images and the README document with the instructions are in the "data/test" directory.

- I used a state-of-the-art object detection algorithm called YOLO (You Only Look Once) operated on a convolutional deep neural network. YOLO has 80 classes of objects it can detect. It divides an image into 19x19 grid and using 5 anchor boxes of different size tries to detect objects. It has a high accuracy for of people and cars detection, since it's used in autonomous driving applications. The algorithm has a non-trivial output that I processed with the help of 6 python scripts stored in "python_scripts" directory. You can run them, using _reticulate_ package available for R. The tech requirements are written below.

- The images were classified and sorted into different output directories "output/with_people" and "output/no_people". You can either load the sorted images using the links in the output folder or run the classification yourself with the function "identify_people". It took me one hour to process 100 pictures on a CPU.

- The test accuracy of the classifier is ...

### Technical requirements.

- Firsty, you should install Keras for R.

```r
install.packages("keras")




