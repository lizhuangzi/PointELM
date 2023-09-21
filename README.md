# PointELM
Demo code for PointELM for point cloud classification

The testKELMDGCNN.py is training and testing KELM for ModelNet40 features extracted by randomly initilized DGCNN.
The testComplexKELMDGCNN.py is training and testing KELM for ModelNet40 features (using complex mapping) extracted by randomly initilized DGCNN.
There are some packages in our settings:

Version:

numpy==1.19.2

scipy==1.2.0

scikit-learn==0.24.2


If you want to extract feature by yourself, you can run saveOutPutDGCNN.py. The DGCNN model is used PyTorch version.
