# Thesis-work on Detection of underground burried networks using Gpr Images by Faster RCNN

**INTRODUCTION**
This work was done as an Internship at CEREMA Laboratory, Angers.
This work was supervised and guided by David GUILBERT, Researcher And JAUFER Rakeeb, Researcher.

**MODEL EXECUTION**

First and for most step is to install anaconda software in your system.
If it is done, Install two github repositories inorder to work with the model.

·	Download the full tensorflow models from the mentioned repository

    https://github.com/tensorflow/models
    
·   In order to execute the model in colab, I have used below mentioned reference.
    
    https://medium.com/analytics-vidhya/training-an-object-detection-model-with-tensorflow-api-using-google-colab-4f9a688d5e8b
 
·	Download another repository by edjeelectronics

    https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10 

After downloading these repositories, we need to copy the models-master folder to the local folder in c-drive by naming the folder as tensorflow1. Rename the models-master as models.

Then after copy the folders and files extracted from edjeelectronics to object detection folder in models.

Download the model you would like to work (in my case it is faster rcnn) from the below mentioned repository.

        https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf1_detection_zoo.md
        
Extract the tar file and copy it to the object detection folder.


**Training the model**

Training of Faster rcnn can be performed in two stages

·	Using Anaconda command prompt (for installing packages)

·	Using google colab(for training)

The steps that i followed for installing packages using Anaconda prompt is mainly from the above edjeelectronics github repository.


**Execution through Anaconda prompt**

·	open anaconda prompt in your system and then follow the order of following commands to install the packages.

·	You need to create a new virtual enviroment tensorflow1 by following command

    C:\> conda create -n tensorflow1 pip

·	If it is already existing press "y" to create a new environment.

·	Activate tensorflow1 by the following command.

    Activate tensorflow1

·	Activate tensorflow gpu by the following command.

    pip install --ignore-installed --upgrade tensorflow-gpu

·	Inorder to install the protobuff compiler follow the below command.

    C:\> conda install -c anaconda protobuf
 
·	Then install the series of packages mentioned below one by one.

    pip install pillow
    pip install lxml
    pip install jupyter
    pip install matplotlib
    pip install pandas
    pip install opencv-python

·	The ‘pandas’ and ‘opencv-python’ packages are not needed by TensorFlow, but they are used in the Python scripts to generate TFRecords and to work with images, videos, and webcam feeds.

·	A PYTHONPATH variable must be created that points to the \models, \models\research, and \models\research\slim directories. 
Do this by issuing the following commands (from any directory):

        set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim
        
·	Now we add pythonpath variable to path variable by typing the below command.

    set PATH= %PATH%;PYTHONPATH

·	You can use "echo %PATH% to see if it has been set or not.

·	Next, compile the Protobuf files, which are used by TensorFlow to configure model and training parameters. Unfortunately, the short protoc compilation command posted on TensorFlow’s Object Detection API installation page does not work on Windows. Every .proto file in the \object_detection\protos directory must be called out individually by the command.

·	 In the Anaconda Command Prompt, change directories to the \models\research directory:

    cd C:\tensorflow1\models\research

·	Then copy and paste the following command into the command line and press Enter:


    protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto .\object_detection\protos\calibration.proto .\object_detection\protos\flexible_grid_anchor_generator.proto


·	Then you can see protobuff files created in protos folder in object detection in the form of .pb files.

·	Execute the following commands to run setup.py file.

·	 Note: Before running we need to change the setup.py from C:\tensorflow1\models-master\research\object_detection\packages\tf1 to C:\tensorflow1\models-master\research\ then run the below command.

     python setup.py build
     python setup.py install

·	Note: if you cannot process the above install command try executing below command.
           
     python setup.py install --user

·   If you have not labled the images then you should do it by the Labelimg. It is a tool used to label the images. The links to access labelImg is given below.
·   For Downloading you can of the link given below.
    
    https://www.dropbox.com/s/tq7zfrcwl44vxan/windows_v1.6.0.zip?dl=1
    
 ·  If you want to access the github follow the below link.

    https://github.com/tzutalin/labelImg

·	If you have labeled the images into xml file then you can divide the train images and validation images in train the test folders in the images folder within object_detection.

·	Then run the following two commands one after another.

    cd C:\tensorflow1\models\research\object_detection
 
    python xml_to_csv.py

.   If the above command don't work then try to execute xml_to_csv.py file in pycharm or jupyter notebook by opening it from the folder location.

·	open the generate_tfrecord.py using text editor and the change the label(). In my case i am detecting only pipe so i have give pipe. If you need to detect multiple things then you can gives all the labels you want to detect

    # TO-DO replace this with label map
    def class_text_to_int(row_label):
        if row_label == 'pipe':
            return 1
        else:
            None
                                         
•	With this above operation the installations of different packages is done through command prompt and our model is ready to train through colab.




**Using google colab**

I have followed the below mentioned blog for execution of the model through Google colab.

    https://medium.com/analytics-vidhya/training-an-object-detection-model-with-tensorflow-api-using-google-colab-4f9a688d5e8b

•	First step is to upload the Tensoflow1 into google drive and then rename the folder as tensorflow.

•	Second step is to open the tensorflow folder in google drive and create the colab file by right click / more/ Colaboratory inside the tensorflow folder.

•	Third step is to open colab file and follow the below steps to execute the model and train it.

•   If you have doubts in executing the colab part, you can access to my .ipynb files uploaded.

•	Firstly we need to select tensorflow version 1. Because the model works fine with the tensorflow 1. Follow the below command.

    %tensorflow_version 1.x

•	We need to change the runtime as gpu so that the model execution will be faster. In order to check the runtime you are using type the below command.

• You should see ‘Found GPU’ and tf version 1.x 

    import tensorflow as tf
    device_name = tf.test.gpu_device_name()
    if device_name != '/device:GPU:0':
       raise SystemError('GPU device not found')
    print('Found GPU at: {}'.format(device_name))
    print(tf.__version__)


•	Mount Google Drive with the code below and click on the link. Then sign in with your google drive account, and grant it access. You will be redirected to a page, copy the code on that page and paste it in the text-box of the Colab session you are running then hit the ENTER key.

    from google.colab import drive
    drive.mount('/content/gdrive')

•	Change directory to the folder you created initially on your google drive. In our case, we named the folder as tensorflow by following command.

     %cd '/content/gdrive/My Drive/tensorflow/'
     
•	Install some needed tools and dependencies by following command.

     !apt-get install protobuf-compiler python-pil python-lxml python-tk
     !pip install Cython
     
•	execute the following command for protoc files.

    %cd /content/gdrive/My Drive/Desktop/models/research/
    !protoc object_detection/protos/*.proto --python_out=.
    
•	set the environment by following below command.

    import os
    os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/tensorflow/models/research/:/content/gdrive/My Drive/tensorflow/models/research/slim'

•	Install the setup.py from the following command.

    !python setup.py build
    !python setup.py install
    
•	If you wish to know the remaining hours you have for your colab session, run the copy and run the code below 

    import time, psutil
    Start = time.time()- psutil.boot_time()
    Left= 12*3600 - Start
    print('Time remaining for this session is: ', Left/3600)
•	Test with the code in the snippet below to see if all we need for the training has been installed. 

    %cd /content/gdrive/My Drive/tensorflow/models/research/object_detection/builders/
    !python model_builder_test.py
    
•   After running model builder test change the directory to 

    %cd /content/gdrive/My Drive/tensorflow/models/research/object_detection

    
• We need to create train.record and test.record files

• For creating train.record copy the code mentioned below.

     !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

• For creating train.record copy the code mentioned below.

    !python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record
    
•	To background track your training checkpoints, run the code cell below. This is the latest way to get your Tensorboard running on colab.

    %load_ext tensorboard
    %tensorboard --logdir training/
    
•	Before training the model here are the few changes we need to do in the existing files.

•	Open training folder and you can see labelmap.pbtxt file. Open it with text editor and then replace all the labels and just include pipe label alone in the folder.

•	In the same folder we need to replace the existing config file with our model config file. You can find all the models config files in the below directory.

    tensorflow\models\research\object_detection\samples\configs

•	After replacing the config file then open it with text editor. Change the number of classes to 1 and next changes needed to be done are explained in the step below.

•	Line 106. Change fine_tune_checkpoint to:

    fine_tune_checkpoint : "faster_rcnn_resnet50_coco_2018_01_28/model.ckpt" 

•	Replace with faster_rcnn_inception_v2_coco_2018_01_28 or faster_rcnn_resnet101_coco_2018_01_28 based on the model you choose.

•	Lines 123 and 125. In the train_input_reader section, change input_path and label_map_path to:
    
    input_path : "train.record" 
    label_map_path: "training/labelmap.pbtxt" 


•	Lines 135 and 137. In the eval_input_reader section, change input_path and label_map_path to:

    input_path : "test.record" 
    label_map_path: "training/labelmap.pbtxt" 

•	Save the file after the changes have been made. That’s it! The training job is all configured and ready to go!

•	Now paste the following command to run the model.

    !python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_resnet50_coco.config

•	In the above command replace the config file with your respective config file you choose faster_rcnn_inception_v2_coco.config/ faster_rcnn_resnet101_coco.config if you like to work on them. 

•	Now the model gets trained and run it upto 60000 steps to get better precision and recall.

•	After reaching to the required step count interupt the cell running.

•	Now that training is complete, the last step is to generate the frozen inference graph (.pb file). From the \object_detection folder, issue the following command, where “XXXX” in “model.ckpt-XXXX” should be replaced with the highest-numbered .ckpt file in the training folder by giving the number of last trained ckpt at XXXX place in the below code.

    !python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/faster_rcnn_resnet50_coco.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory inference_graph

•	In the above command replace the config file with your respective config file you choose faster_rcnn_inception_v2_coco.config/ faster_rcnn_resnet101_coco.config if you like to work on them.

•	Then open the object detection folder and open the respective python file to test your model. In my case i have used Object_detection_image.py and Object_detection_video.py

• open the respective file using the text editor and copy it to the colab.

•	After that you should upload a test image or video in the object detection folder and do the necessary changes accordingly.

•	Include this line at importing packages

    from google.colab.patches import cv2_imshow
    
•	video name or image name is changed with respective to your operation. 

    # Name of the directory containing the object detection module we're using
     MODEL_NAME = 'inference_graph'
     VIDEO_NAME = 'test.mov'

•	If it is image it should be in jpg format and if it is video it should be in .mov. At the end we need to change code of specific line 

    cv2.imshow('Object detector', frame) to cv2_imshow(frame)
    
•	If it is image it is done as below.

    cv2.imshow('Object detector', image) to cv2_imshow(image)
    
•	Then you can execute the cell and see the output.

•	Final step is to check the validation results of the model performance through the metrics. It can be checked by following command.

    !python /content/gdrive/MyDrive/tensorflow/models/research/object_detection/model_main.py --model_dir = /content/gdrive/MyDrive/tensorflow/models/research/object_detection/faster_rcnn_resnet50_coco_2018_01_28 --pipeline_config_path=/content/gdrive/MyDrive/tensorflow/models/research/object_detection/training/pipeline.config --checkpoint_dir=/content/gdrive/MyDrive/tensorflow/models/research/object_detection/training

•	Change the model name in command with respect to the model you use like faster_rcnn_inception_v2_coco_2018_01_28 or faster_rcnn_resnet101_coco_2018_01_28. 

•   If you follow the above mentioned steps, You can easily execute Faster RCNN model.
