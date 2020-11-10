# Image-Recognition-_Alexnet_Tensorflow
Implementthe classic Alexnet for image categorization of CIFAR-10 dataset with Tensorflow

# tensorflow-cifar-10
###this is the Alexnet network working in cifar-10 dataset

## Requirement
**Library** | **Version**
--- | ---
**Python** | **^3.6.5**
**Tensorflow** | **^1.6.0**
**Numpy** | **^1.14.2** 
**Pickle** | **^4.0**  
**opencv-python**|**3.4.6**

## Accuracy
Best accurancy what I receive was ```81.12%``` on test data set. 


### Train network
By default network will be run 60 epoch (60 times on all training data set).  
_BATCH_SIZE =128

python3 train.py
```

### Run network on test data set
```sh
python3 predict.py
```

###result files
some result files are saved in the file named checkpoint_dir

