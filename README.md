## A classifier to identify people in Instagram images for Looping Power Room

Hello, Matthias! 

The task is done. I already wrote you the short version of the strategy I was using, but here is the extended version:

### Approach

- I wrotea simple function that scrapes images from Instagram, using the links you provided me with. The function names is "load_images_on_disk" and it's inside the looping_instagram_task.Rmd file.

I found 1886 images using that function and downloaded them from Instagram. The link to the images is in "data/downloaded_images" directory. You can download them, but I use only 100 images from there to test the model. The links to the test images and the README document with the instructions are in the "data/test" directory.

- I used a state-of-the-art object detection algorithm called YOLO (you only look once) operated on a convolutional deep neural network. YOLO has 80 classes of objects it can detect. It divides an image into 19x19 grid and using 5 anchor boxes of different size tries to detect objects. It has a high accuracy for of people and cars detection, since it's used in autonomous driving applications. The algorithm has a non-trivial output that I processed with the help of 6 python scripts stored in "python_scr




