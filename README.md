This is codebase is designed to analyse the crowd of Indian temple. Such crowd is very Heavy and Disordered.
There is also a problem of Perspective depth because some people appear in near regions of camera view and appear distinctly. When we see far off regions of camera view people become very obscure and traditional methods fail.
This project uses Parallel pipeline. Here is the description

The proposed system is built on three distinct but complementary components, each chosen to handle a specific layer of the occupancy estimation pipeline. Together they cover real-time object detection, persistent identity tracking and density-aware counting for regions where individual detection becomes unreliable.

We used YOLO 11x + DeepSORT + DM count and Python as base.

Motivation:

Since YOLO 11x is a state of art of model we thought to use it firstly. It was also backed by many papers. We required tracking of people so we employed DeepSORT to help combat occlusion in crowd. Now for far off regions we tried to employ Bayesian Loss (BL) but then we came across State of art DM count more robust framework so we decided to go for it.  We used pre trained model trained on "Shahghai\_A" which resembles Indian temple crowd. 
Implementation:

We cropped our video frame in 60:40 ratio area-wise. Then we implemented our parallel pipeline. As for results it works best if there is no NOISE (jitter) of far off people that is people should at least be visible. 
If you employ this code base Install all dependencies of temple\_crowd\_project folder and of DM count. 
This codebase is an attempt to digitise crowd control especially in temples. 




