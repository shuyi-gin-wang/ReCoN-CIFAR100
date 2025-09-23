# Implementation of [Request Confirmation Network](https://ceur-ws.org/Vol-1583/CoCoNIPS_2015_paper_6.pdf)

A ReCoN is implemented in recon.py. It's used to build a learned representation of CIFAR100's hiearchical classes with self organizing hypothesis of CIFAR's classes and superclasses.

![recon_sequence_2](https://github.com/user-attachments/assets/597e117f-320d-4b79-9508-2acdf9653330)

### Details
* A small ResNet acts as the vision module of the ReCoN (pre-trained on CIFAR100); Given an image, it outputs both superclass and class labels
* A learned representation of the hiearchical prediction (superclass -> class) is built with ReCoN. A superclass hypothesis is confirmed before the subclass hypothesis, creating por/ret relationships.
* If the ReCoN representation (with visual module attached) sees a new class, a class hypothesis gets added if it's able to confirm its superclass. Otherwise, create a new superclass hypothesis with the nested class hypothesis.
  
<img width="1304" height="897" alt="image" src="https://github.com/user-attachments/assets/9c0c1fef-277b-42c8-a9b0-1d244dcee588" />

* As the ReCoN representation sees more and more images from the CIFAR test set, less hypotheses is formed.

![recon_sequence_5](https://github.com/user-attachments/assets/d5be8454-de55-4af1-9ba1-f27467821119)

### To Step through the ReCoN Building process
* pip install -r requirements.txt
* run the file `run_recon.py`
