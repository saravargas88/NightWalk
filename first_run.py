


'''
Feasibility test for first run of the application. 
1. Run the images through Grounding DINO and identify the 3 proxies for lightness 
2. Maybe attempt some other indicators: 



- two stage models: detecting ligthing relevant visual cues from dayytime images 
    to regress the predicted brightness using.small set o paired day and night images.
    
- So we basically predict lightness from the daytime image proxies 
 and then use that to predict the brightness of the nightime images.
 
 
 - traning a model to extrat features from daytime SVI
 - map them to nighttime brightness 
 => produce city wide lighting estimates solving lack of nighttime SVI data.
 
 
 For every daytime image in the dataset:
 - predict the brightness of the nighttime image using the proxies for lightness (this would be running every picture with DINO?)
      -counting frequencies 
      
 
 For every nighttime image in the dataset:
 - do we have another image in that location for daytime? If so pair it 
 - if not, fetch daytime image from Google Street View and pair it 
    - crop the google street view PANORAMA to just be the direction our nighttime image is 
 
 Ground truth is drawn from the nighttime images that we do have and is determined by pixel brightness.  
 

'''
