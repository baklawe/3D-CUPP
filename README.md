# 3D-CUPP
This will be the best project ever

we also have boxes:
- First task
  - [X] implement a good version of PointNet
  - [X] implement implement the supplementary code ( need to expand)
- Second task
  - [X] create code for creating a binary image from PC
  - [ ] create documentation for creating a binary image from PC
  - [ ] add depth to the 2d image 
  - [X] decide and create a filter for the image and decide the scaling
  - [X] create a network that gets M projected images (28x28) create M feature or class vectors\
 then perform a symmetric function over the M dimension and then classify.
  - [X] Try version with Batch-norm instead of dropout in the conv layer. Conclusion: got the same result but faster\
   and larger overfit gap - trying with the jitter
  - [X] Try Pic-net with rotation and jitter. Conclusion: Don't rotate for 28x28. 
  - [ ] Try more pixels: 32, 64, etc. Conclusion: 32 makes improvement. 
  - [ ] Try more projection angles: pi/4, random, etc. Conclusion: pi/4 addition (10 instead of 6) makes deterioration.
  - [ ] Try FPS for projection angles (FPP).
  - [ ] Try sum views in PicNet.  Conclusion: Dos'nt shoe improvement.
  
- Third task
  - [X] Implement a full version of PointNet.
  - [ ] Get to PointNet ~89 accuracy.
  
- Forth task 
  - [X] Combining things.
  - [ ] Take max (and sum) from the features of pc and proj.
  - [ ] Take max (and sum) from the classes pred of pc and proj.
  - [ ] Combine features in different stages.
  - [ ] Train the networks separately first (keep features weights) and then join them.

Current SOTA = 89.9 

- [ ] we also need to order some good pizza.\
Zanzara : http://www.2eat.co.il/zanzara/menu.aspx?pid=6113\
Crosta : https://he-il.facebook.com/CROSTAHAIFA/
