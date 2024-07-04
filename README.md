![](https://github.com/pragyy/datascience-readme-template/blob/main/Headerheader.jpg)

# Project Zaku

> A Guid to use Zaku.

Program that can assess more than 10 different classification and regression models to determine the best fit for your CSV data file. This program should be capable of evaluating the performance of these models and selecting the most suitable one based on specific metrics. 

# Project Overview

This has 3 modules which you have to use which are classifier , regressor  , and preproccesor 
(Note: You can define a custom regressor as well)
lets discuss it in details futher

# Installation and Setup

## Codes and Resources Used
In this section I give user the necessary information about the software requirements.
- **Editor Used:**  Pycharm
- **Python Version:** 3.11.9

##Install Dependencies: 
Use pip to install all the packages listed in the file by running the following command in your terminal or command prompt:
```bash
pip install -r requirements.txt
```

# Code structure
The code is divided into 3 files mainly 
src: This is the source file for the codes
Model: The best model will be saved here
logs: These are the logs for debugging if needed

```bash
├── Data
│   ├── Add your csv file here
├── Model
│   ├── best_model.joblib
├── logs
│   ├── preporcessing.log
├── src
│   ├── MLC_Classifier.py
│   ├── Regressor.py
│   ├── data_preprocessing.py
└── Main.py
```

# Running Code
## Running in compiler
if you want to run code as a script open main.py and run the file both sample code for classifcation and regression are given.
notes parameters which you can use for both are:

### Classification
- **Default:** : Acc
- **f1 score** : f1
### Regression
- **Default:** : r2_score
- **mean_squared_error** : mse

# Future work
Outline potential future work that can be done to extend the project or improve its functionality. This will help others understand the scope of your project and identify areas where they can contribute.

# Acknowledgments/References
Acknowledge any contributors, data sources, or other relevant parties who have contributed to the project. This is an excellent way to show your appreciation for those who have helped you along the way.

For instance, I am referencing the image that I used for my readme header - 
- Image by [rashadashurov](https://www.vectorstock.com/royalty-free-vector/data-science-cartoon-template-with-flat-elements-vector-27984292)

# License
Specify the license under which your code is released. Moreover, provide the licenses associated with the dataset you are using. This is important for others to know if they want to use or contribute to your project. 

For this github repository, the License used is [MIT License](https://opensource.org/license/mit/).
