# Classification of Political Party Leaning Through NLP Analysis on US Presidential Election Tweets
## Project Description
Social media plays a significant role in shaping public opinions in politics, and it remains unclear how the pattern of political party preference has evolved between the recent US elections. To find out how social media posts reflect user’s political leaning, this research focuses on the posts from X (Twitter). This project applies natural language processing (NLP) techniques to classify political party leaning in Twitter posts related to the 2020 and 2024 U.S. Presidential elections using textual and semantic features. We will clean the tweet texts, transform the texts through tokenization and embeddings conversion, and train classification models to infer party preference. Specifically, the ML models used for classification are logistic regression, DistilBERT, SVM, and Naive Bayes. The reason why multiple models are used is because of the uncertainty in the form of relationship between tweet texts and party preference. By comparing model outputs across both election datasets, we aim to examine shifts in political discourse and alignment over time. This analysis may also reflect which ML model is better and more suitable for social media texts classification, as well as showing the broader changes in social media usage and political communication between the two election cycles.

## Classification Results
Logistic Regression Result:
- 2020 Classification

![2020_logistic_regression](https://github.com/user-attachments/assets/c51c8ea2-5de7-4775-b458-159b024d5e78)
![2020_logistic_report](https://github.com/user-attachments/assets/dc18d8a4-e20c-4a0f-97cb-5f37a712731b)
- 2024 Classification

![2024_logistic_regression](https://github.com/user-attachments/assets/48ef1bb0-b0d5-4199-b07f-86182c94cd24)
![2024_logistic_report](https://github.com/user-attachments/assets/f53fae2b-1ad1-4157-96da-4e505ee30459)

DistilBERT Result:
- 2020 Classification

![2020_DistilBERT](https://github.com/user-attachments/assets/8cd6ce02-38e7-43c3-a65a-5daa668f0628)
![2020_DistilBERT_report](https://github.com/user-attachments/assets/bfe60c15-cc89-4c49-bdcf-8c56658775d2)
- 2024 Classification

![2024_DistilBERT](https://github.com/user-attachments/assets/ba87379e-1496-4ded-a480-168d432f8ef1)
![2024_DistilBERT_report](https://github.com/user-attachments/assets/8fead2d7-ac6d-44ec-9556-d6b96300a317)

SVM Result (with hyperparameter fine-tuning):
- 2020 Classification

![2020_SVM_fine_tuned](https://github.com/user-attachments/assets/4cd5dc8b-ac67-4988-bc63-1d065ccdf2c3)
![2020_SVM_fine_tuned_report](https://github.com/user-attachments/assets/20235eec-55f1-4354-bebd-c4d7c74e818d)
- 2024 Classification

![2024_SVM_fine_tuned](https://github.com/user-attachments/assets/d68bce0d-1b49-46e1-9b8d-f98608ebf261)
![2020_SVM_fine_tuned_report](https://github.com/user-attachments/assets/60a9b1d4-da31-4e58-88cb-b52212cef31c)

Naive Bayes Result:
- 2020 Classification

![2020_Naive_Bayes](https://github.com/user-attachments/assets/b54a92da-de2e-42f0-9ae2-b7365f1bf1ca)
![2020_Naive_Bayes_Report](https://github.com/user-attachments/assets/b13242d3-c989-44f6-bd20-0e597d3ba657)
- 2024 Classification

![2024_Naive_Bayes](https://github.com/user-attachments/assets/004ba97c-dbc2-4742-97a8-7ae25b4f4acd)
![2024_Naive_Bayes_report](https://github.com/user-attachments/assets/dd6d4600-b266-4c2a-85ff-3a7fa640ea6b)

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
First of all, run the very first cell in Project.iypnb as it imports all neccessary libraries for this program. Then, run the second cell where a "Choose Files" button should pop up, and you can click on it to browse files from your device. When browsing, go into the Data folder downloaded earlier, then go to the "cleaned" subfolder where you'll find 2020_cleaned_tweets.csv and 2024_cleaned_tweets.csv. Upload these two files as they contain the tweets related to the 2020 and 2024 US presidential election. Since these files are uploaded to Google collab's VM, they only temporarily stay within the current collab session, so once you close the collab window or restart the kernel, these files will be gone and you have to upload them again.

### Running the models
Make sure the very first cell is already executed before running any of the models, because they rely on many of the libraries imported at the beginning. 
- Logistic regression: To run the logistic regression model, go to the header of "TF-IDF embedding + logistic regression approach", then run all the cells below it until the next header. This should correctly create the train/test set, vectorize the dataset with TF-IDF, load and train the model, and finally make the classification on both datasets. The last few cells will generate two classification report and confusion matrices, which shows the prediction performance in terms of precision, recall, F1-score, and accuracy.

- DistilBERT: To run the DistilBERT model, go to the header of "DistilBERT approach", then run all the cells below it until the next header. This should correctly create the train/validate/test set, vectorize the dataset with DistilBERT's own embedding layer, load and train the model, and finally make the classification on both datasets. Similarly, the last few cells will generate two classification report and confusion matrices, which shows the prediction performance.

- SVM: To run the SVM model, go to the header of "SVM", then run all the cells below it until the next header. This should correctly load and train the model, and make the classification on both datasets. There's no independent embedding process as SVM uses TF-IDF as its vectorizer, so it simply uses the vectorized train/test set created earlier in the logistic regression approach. As for a note, the training and testing process happens twice for SVM as the first one is the raw model while the second one has hyperparameter fine-tuning, which can be compared to see how much improvement can be done after fine-tuning the model. Similarly, the last few cells will generate two classification report and confusion matrices, which shows the prediction performance.

- Naive Bayes: To run the Naive Bayes model, go to the header of "Naive Bayes Classifier", then run all the cells below it until the next header. This should correctly load and train the model, and make the classification on both datasets. There's no independent embedding process as Naive Bayes also uses TF-IDF as its vectorizer, so it simply uses the vectorized train/test set created earlier in the logistic regression approach. Similarly, the last few cells will generate two classification report and confusion matrices, which shows the prediction performance.

## How to Interpret the Results
- Analyzing Confusion Matrix

When looking at the confusion matrix, the horizontal axis is the predicted label while the vertical axis is the actual label. Next, the order of parties is (Democrat, Republican) for both axes. Hence, the top left square indicates the number of test cases that's actually democrat and predicted to be democrat (true positive), while top right means the cases that are actually democrat but predicted to be republican (false negative). With the same logic, the bottom left means the cases that are actually republican but predicted as democrat (false positive), while the bottom right means the cases that are actually republican and predicted to be republican (true negative). For example, for the prediction result of 2020 election by logistic regression, it correctly classified 1099 democrat tweets and 1148 republican tweets, but falsely classfied 162 democrat tweets and 173 republican tweets, resulting in an overall accuracy of 87%.

- Analyzing Classification Report

When looking at the classification report, the key metrics to focus on is precision, recall, f1-score, and accuracy. Precision means out of all the labels predicted to be correct/positive, how many of them are actually correct. Recall means how much of the total correct labels are identified and predicted as correct. F1-score means whether there's a good balance between the precision and recall score, in which a high precision and high recall results in a high F1-score, while a high + low combination of precision and recall results in a low F1-score. Lastly, accuracy means how many labels are correctly classified, which is the combination of true positive and true negative on the confusion matrix.

## Analysis You Can Do Using This Program
- Which model shows the highest balance between its recall and precision? What does this infer about its classification performance?
- Which model shows the highest accuracy? Given the distribution of tweets supporting Democrat and Republican, how significant is accuracy when measuring model performance?
- Assuming the model classification results are correct, how has the political party leaning changed between the 2020 and 2024 US presidential election?
- Does the use of embedding method with semantic understanding improves the classification performance? By how much?
- Which model is the best option for social media texts classification?

## Contributors
1. Kuan-Chun Chiu (Myself) - beagledirk1@gmail.com
2. Jae Na Wray - wray.ja@northeastern.edu
3. Meroska Gouhar - gouhar.m@northeastern.edu
4. Yiyang Zhang - zhang.yiyang4@northeastern.edu
