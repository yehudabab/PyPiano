# PyPiano
Piano keys detector
===
Given an input image of piano keys, the script identifies the piano keys and labels them


![Alt Text](https://github.com/yehudabab/PyPiano/blob/master/readme_resources/s2.gif)
![Alt Text](https://github.com/yehudabab/PyPiano/blob/master/readme_resources/s3.gif)
![Alt Text](https://github.com/yehudabab/PyPiano/blob/master/readme_resources/s1.gif)

The process uses no learning models, only classical image processing techniques. 

The process is pretty straightforward: detect the edges in the image and close the holes. 
Find connected components and filter the white components, which will hopefully be the piano keys. 
Filter out too large white components. 
Find the center of mass of the white components, and again filter out components which do not conform to a RANSAC linear fit. 
Find the borders between each pair of adjacent keys and probe along the top of the border to see if it's a black region. 
If so, there's a black key there. Identify the F note by a series of 3 black notes. 
Label the notes accordingly.
