# Build Neural Network from scratch - Dog Breed Classification

###### This is derived from the project of CSCI3230 . This project is using the data set from the project of CSCI3230.

## Program description:
/main/nn.py : the main for training, modify the #Configurable to adjust the result  
/main/util.py : util library for the preprocess and training phase  
/main/preprocess.py : program that generate the training data and labels, which can also be downloaded from the link below  
/main/test.py : just my playground program to test some function, useless to training and preprocess  

## Link to data set:
https://mycuhk-my.sharepoint.com/:f:/g/personal/1155063445_link_cuhk_edu_hk/Eg7YyOwbJxpHp6USzQjgINoB2DdoaTYfEd0cTNL9SaP8Mw?e=ot6sT5
## How to start:
**Step 1:**  
Download the above data set to the root directory of the repository and extract it. It should contain data/testing.npy, data/train_data_array.npy and data/train_data_onehot.npy.  
**Step 2:**  
Remove the data/trained_theta.npy and data/training_info.txt as your will.  
(If the theta is reloaded, the validation accuracy should be around 0.28, with nodes number in four layers=[30000,3000,800,25])  
These files will be regenerated in the training process.  
**Step 3:**  
cd to /main and run nn.py. Some message will then show up.  
**Step 4:**  
Base on the rough result, do step 2 again and try to adjust and config to see the effect on training and testing.  
![console output](https://i.imgur.com/B7Y3WDG.png)  
**Step 5:**  
Enjoy!  

## Dataset description:  
**Training set**: 4729 images of total 25 classes of dog  
/train_original/ : original images of training set  
/train_reshaped/ : reshaped images of training set, each with size (100, 100, 3)  
/train_label.txt : the labels of the training set  

**Validation set**: 250 images of total 25 classes of dog  
/testing.npy : dictionary of the validation set, included keys: (['shapes', 'file_name', 'original', 'reshaped', 'label'])  

### Supplementary file:
/tips*.txt : CSCI3230 tutors tips on the original project  
/specification.pdf : specification of the original project  
