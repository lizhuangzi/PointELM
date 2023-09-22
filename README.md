# PointELM
Demo code for PointELM for point cloud classification

Abstract:
Designing an influential and informative deep network has re- cently become hot research for point cloud analysis. How- ever, deep networks inevitably need to be trained for a long time and are sensitive to variational data distribution. In this paper, we provide special thinking that shows point cloud classification can be realized without training deep networks. Specifically, we propose a Point Extreme Learning Machine (PointELM), which first extracts infant features from point clouds using a randomly initialized network. Then, a complex mapping is designed to convert the infant features from real- valued to complex-valued domains. Lastly, a kernel ELM is adopted to discriminate the complex features and outputs the prediction results. We evaluate PointELM on three classical networks and indicate it has three advantages: (1) PointELM doesn’t need the back-propagation algorithm, so it can be trained extremely fast. Yet PointELM still achieves promising performance compared with corresponding well-trained net- works, e.g., its classification accuracy only lowers a trained DGCNN about 2.5% on ModelNet40. (2) PointELM can conveniently benefit a trained network to transfer to new un- seen datasets with less performance loss. (3) PointELM can quickly adjust the performance degradation of trained net- works on sampled point clouds, showing strong potential for adapting different data distributions.

# Useage
The testKELMDGCNN.py is training and testing KELM for ModelNet40 features extracted by randomly initilized DGCNN.
The testComplexKELMDGCNN.py is training and testing KELM for ModelNet40 features (using complex mapping) extracted by randomly initilized DGCNN.
There are some packages in our settings:

Version:

numpy=1.19.2

scipy=1.2.0

scikit-learn=0.24.2

If you want to extract feature by yourself, you can run saveOutPutDGCNN.py. The DGCNN model is used PyTorch version.
