"""
Lauren Liao
CSE 163 AB

The file that implements multiple testing functions including
checking data visualization and hyperparameter setting
to verify the correctness of functions for Final Project
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def model_tune(data):
    """
    Takes the data as a parameter and make a decision tree classifier
    model to take in age, gender, etc to predict the chance of
    getting a heart attack and prints the train and test accuracy score
    with models at max depth from 1 to 13.
    """
    data = data.dropna()
    features = data.loc[:, data.columns != "target"]
    labels = data["target"]
    features = pd.get_dummies(features)
    labels = pd.get_dummies(labels)
    # print(labels)
    # print(features)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
    for i in range(1, 14):
        print("model with hyperparameter of " + str(i))
        model = DecisionTreeClassifier(max_depth=i)
        model.fit(features_train, labels_train)
        train_predictions = model.predict(features_train)
        train_accuracy = accuracy_score(labels_train, train_predictions)
        test_predictions = model.predict(features_test)
        test_accuracy = accuracy_score(labels_test, test_predictions)
        print(train_accuracy, test_accuracy)


def target_gender_test(data):
    """
    Takes the data as a parameter and make a bar chart of the rate of heart
    attack by gender. You should group the dataset by sex and calculate the
    rate of heart attacks for each. Label x-axis as Gender, y-axis as Rate of
    Heart Attack and title as Rate of Heart Attacks by Gender
    (0=Female, 1=Male) Save it in the file named target_gender_test.png.
    Remember you need to pass that extra parameter to make the layout tight.
    """
    revised_data = data.loc[:, ["sex", "target"]]
    revised_data = revised_data.dropna()
    revised_data = revised_data.groupby(["sex"])["target"].mean()
    revised_data = revised_data.reset_index()
    sns.catplot(x="sex", y="target", data=revised_data, kind="bar")
    plt.ylabel("Rate of Heart Attack")
    plt.xlabel("Gender")
    plt.title("Rate of Heart Attacks by Gender (0=Female, 1=Male)")
    plt.savefig("result/target_gender_test.png", bbox_inches="tight")


def target_chest_pain(data):
    """
    Takes the data as a parameter and make a bar chart of the rate of heart
    attack by chest pain type. You should group the dataset by chest pain type
    and calculate the rate of heart attacks for each type. Label x-axis as
    Chest Pain Type, y-axis as Rate of Heart Attack and title as Rate of Heart
    Attacks by Chest Pain Type and include the legend of pain type.
    Save it in the file named target_chest_pain.png.
    """
    revised_data = data.loc[:, ["cp", "target"]]
    revised_data = revised_data.dropna()
    revised_data = revised_data.groupby(["cp"])["target"].mean()
    revised_data = revised_data.reset_index()
    sns.catplot(x="cp", y="target", data=revised_data, kind="bar")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Rate of Heart Attack")
    plt.legend(labels=["0: typical angina", "1: atypical angina",
                       "2: non-anginal pain", "3: asymptomatic"])
    plt.title("Rate of Heart Attacks by Chest Pain Type")
    plt.savefig("result/target_chest_pain.png", bbox_inches="tight")
