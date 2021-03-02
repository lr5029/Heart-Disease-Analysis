# Instructions 
## Read data
* Dataset can be downloaded directly to a csv file from the link on the dataset section of the pdf named **CSE163 Final Project Proposal**. I renamed the file to **datasets_heart.csv** and saved it to the **data** folder, while the original file name is much longer, remember to change the file name in the path if you don't want to rename the file. I also submitted the renamed csv file if desired.
## Run functions
* Go to **main.py** modules to run every functions I implemented for this project. 
    * **final_project_lauren.py** includes all the functions implemented targetting the research questions, while **final_project_testing.py** includes all the testing functions to check the correctness of the machine learning model and any data visualizations. 
        * Specifically, in the **final_project_lauren.py**, **model** function targetting machine learning research question, **target_age_gender** function examining relationshp between rate of heart attack over ages by gender targetting the second research question, **target_angina** function examining relationshp between rate of heart attack by exercise induced angina targetting the first part of the third research question, and **target_blood_pressure** function examining relationshp between rate of heart attack over resting blood pressure targetting the second part of the third research question.
        * In the **final_project_testing.py**, **model_tune** function helps identifying the hyperparameter for machine learning model, **target_gender_test** function examining relationshp between rate of heart attack by gender helps confirming the result of the second research question, and **target_chest_pain** function examining relationshp between rate of heart attack by chest pain type helps confirming the result of the first part of the third research question.
## Output graphs
* All the plots will be saved to the folder named **result**, so go to result folder to check any data visualization you expect to see.
## Extra Libraries to import
* When I used plotly for data visualization, an error **"No module named Plotly"** came up. If that ever happen again, try run **"pip install plotly"** or **"conda install -c plotly"** from the terminal window. Once it works, follow the instruction (like install packages) after you run main.py in the terminal couple times as it may ask you to download two more libraries/packages for Plotly to work depending on the computer. 



