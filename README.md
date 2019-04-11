# railway_video_detection

- part.1 image detection:
        detect the railway in the video:
          railway_detect.py:
            use opencv and matplotlib to process the origin frame.
            add gussianblur filter, canny filter.
            then use houghline transform to detect the railway track.
            
            use matplotlib to draw railway and merge with the original image.
            
            
            
            
- part.2 object detection and tracking
       detect and track the objects in the video
          tracking.py:
             imageAI has trained classifier to do object detection.
             trained model are saved in h5 file. 
             using rcnn structure.
