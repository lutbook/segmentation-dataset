# segmentation-dataset

Combining CamVid[1] and Cityscapes[2] segmentation datasets.  
New classes are common objects appear on both datasets.  
Image of the camvid dataset will be resized to that of cityscapes dataset.  
No need to move files in each datasets.

Before exectution, data directory must be prepared same as below.    
New dataset dir tree.
<pre>
data__  
    |__test___
    |        |__images__ ... 
    |        |__labels__ ... 
    |  
    |__train__
    |        |__images__ ...  
    |        |__labels__ ... 
    |  
    |__val____
             |___images__ ...  
             |__labels__ ...  
</pre>

----------------------------------------------------------------------------------------------------------
[1]. The Cambridge-driving Labeled Video Database (CamVid):
Segmentation and Recognition Using Structure from Motion Point Clouds, ECCV 2008 ([pdf](http://www.inf.ethz.ch/personal/gbrostow/ext/MotionSegRecECCV08.pdf))
Brostow, Shotton, Fauqueur, Cipolla ([bibtex](http://www.cs.ucl.ac.uk/staff/G.Brostow/bibs/RecognitionFromMotion_bib.html))   
(from https://course.fast.ai/datasets)

[2]. Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, Bernt Schiele, "The Cityscapes Dataset for Semantic Urban Scene Understanding", (2016), 	arXiv:1604.01685 [cs.CV].
https://arxiv.org/pdf/1604.01685.pdf   
https://www.cityscapes-dataset.com/
