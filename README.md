# Classification of Political Party Leaning Through NLP Analysis on US Presidential Election Tweets
## Project Description
Social media plays a significant role in shaping public opinions in politics, and it remains unclear how the pattern of political party preference has evolved between the recent US elections. To find out how social media posts reflect user’s political leaning, this research focuses on the posts from X (Twitter). This project applies natural language processing (NLP) techniques to classify political party leaning in Twitter posts related to the 2020 and 2024 U.S. Presidential elections using textual and semantic features. We will clean the tweet texts, transform the texts through tokenization and embeddings conversion, and train classification models to infer party preference. Specifically, the ML models used for classification are logistic regression, DistilBERT, SVM, and Naive Bayes. The reason why multiple models are used is because of the uncertainty in the form of relationship between tweet texts and party preference. By comparing model outputs across both election datasets, we aim to examine shifts in political discourse and alignment over time. This analysis may also reflect which ML model is better and more suitable for social media texts classification, as well as showing the broader changes in social media usage and political communication between the two election cycles.

## Classification Results
Logistic Regression Result:
- 2020 Classification

![2020_logistic_regression](https://github.com/user-attachments/assets/c51c8ea2-5de7-4775-b458-159b024d5e78)
- 2024 Classification

![2024_logistic_regression](https://github.com/user-attachments/assets/48ef1bb0-b0d5-4199-b07f-86182c94cd24)

DistilBERT Result:
- 2020 Classification

![2020_DistilBERT](https://github.com/user-attachments/assets/8cd6ce02-38e7-43c3-a65a-5daa668f0628)
- 2024 Classification

![2024_DistilBERT](https://github.com/user-attachments/assets/ba87379e-1496-4ded-a480-168d432f8ef1)

SVM Result:
- 2020 Classification

![2020_SVM](https://github.com/user-attachments/assets/83b24a4f-05e0-47f0-b57f-f3ebc4dcf4f2)
- 2024 Classification

![2024_SVM](https://github.com/user-attachments/assets/0826d13b-692e-47ca-b8cb-6eeb5725283b)

Naive Bayes Result:
- 2020 Classification

![2020_Naive_Bayes](https://github.com/user-attachments/assets/b54a92da-de2e-42f0-9ae2-b7365f1bf1ca)
- 2024 Classification

![2024_Naive_Bayes](https://github.com/user-attachments/assets/004ba97c-dbc2-4742-97a8-7ae25b4f4acd)

## How to Install and Run the Program
### Dependencies:
- transformers >= 5.0.0
- scikit-learn >= 1.6.1
- langdetect >= 1.0.9
- matplotlib >= 3.10.0
- pandas >= 2.2.2
- numpy >= 2.0.2
- torch >= 2.11.0
- re >= 2.2.1

### Instructions:
  1. Make sure all libraries in the above dependency list are installed to your current working environment
  2. Download the Data folder, which contains all data files needed for this program. To see how the data is preprocessed and cleaned, check the clean_data.py file
  3. Download and run the Experiments.ipynb file to see the preliminary results and statistics of the data files.
  4. Download and run the Project.ipynb file, which should load the 2020 and 2024 datasets, train the 4 models, and yield the classification results for both datasets.

### Troubleshooting:
  - If there's error reading the data files, make sure they're in the Data folder and their names aren't modified after downloaded
  - If there's error with the libraries used, make sure they're all properly installed to your current environment, and the version are the same or newer than the versions in the dependency list
  - If there's error running the Project.iypnb file or the runtime is unexpectedly long, make sure your editor (ex: VS Code, Jupyter Notebook...) supports GPU runtype so your models can be trained and tested on GPU, or else simply relying on CPU will take hours to train or even load the models.

## How to Use the Program
### Uploading data files:


### Running the models


## How to Interpret the Results


## Analysis You Can Do Using This Program
- 
- 
- 
- 
- 

## Contributors
1. Kuan-Chun Chiu (Myself) - beagledirk1@gmail.com
2. Jae Na Wray - wray.ja@northeastern.edu
3. Meroska Gouhar - gouhar.m@northeastern.edu
4. Yiyang Zhang - zhang.yiyang4@northeastern.edu
