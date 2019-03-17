# TFG - Bird Audio Detection

Based on the work of Michael T. Johnson,Narjes Bozorg, Sidrah Liaqat, Neenu Jose, Patrick Conrey, Anthony Tamasi
Original work here: https://github.com/UKYSpeechLab/ukybirddet

**Required Software:**

- Python version 3.6
- Tensorflow-gpu (tested on v1.8)
- Keras
- other packages and dependencies listed in [requirements.txt](https://github.com/UKYSpeechLab/ukybirddet/blob/master/requirements.txt)

**Project directory structure**

For running this project on the DCASE challenge 2018 data, follow this directory structure:
(unzip the provided compressed files in the main project directory)

- < project directory >/adaptation_files
- < project directory >/labels
- < project directory >/prediction
- < project directory >/trained_model
- < project directory >/workingfiles

The directory 'workingfiles' has four subdirectories that contain feature files. 

- < project directory >/workingfiles/features_baseline ([download link](https://drive.google.com/drive/folders/1Zf8LQxZF9KISByGmmxx-dbtHLc5dk9Ib?usp=sharing))

In order to reproduce the results of this submission, place the python files in the main project directory and run in the following order:
- birddet_baseline.py



