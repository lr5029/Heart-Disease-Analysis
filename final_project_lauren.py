"""
Lauren Liao
CSE 163 AB

The file that implement functions aiming to answer three
research question on heart attacks rate for Final Project.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sns.set()


def model(data):
    """
    Takes the data as a parameter and make a decision tree classifier
    model to take in age, gender, resting blood pressure, etc to predict
    the chance of getting a heart attack (target) and returns the train
    and test accuracy score.
    """
    data = data.dropna()
    features = data.loc[:, data.columns != "target"]
    labels = data["target"]
    features = pd.get_dummies(features)
    labels = pd.get_dummies(labels)
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3)
    model = DecisionTreeClassifier(max_depth=6)
    model.fit(features_train, labels_train)
    train_predictions = model.predict(features_train)
    train_accuracy = accuracy_score(labels_train, train_predictions)
    test_predictions = model.predict(features_test)
    test_accuracy = accuracy_score(labels_test, test_predictions)
    return train_accuracy, test_accuracy


def target_age_gender(data):
    """
    Takes the data as a parameter and make a line chart of the rate of heart
    attack over age. You should group the dataset by ages and sex and
    calculate the rate of heart attacks for each age, and color the point by
    their gender. Label x-axis as Age, y-axis as Rate of Heart Attack and title
    as Rate of Heart Attacks vs Ages by Gender and include the legend of sex.
    Save it in the file named target_age_gender.png. Remember you need to pass
    that extra parameter to make the layout tight.
    """
    revised_data = data.loc[:, ["sex", "age", "target"]]
    revised_data = revised_data.dropna()
    revised_data = revised_data.groupby(["age", "sex"])["target"].mean()
    revised_data = revised_data.reset_index()
    sns.relplot(x="age", y="target", data=revised_data, kind="line",
                hue="sex", legend=False)
    plt.legend(labels=['Female', 'Male'])
    plt.xlabel("Age")
    plt.ylabel("Rate of Heart Attacks")
    plt.title("Rate of Heart Attacks vs Ages by Gender")
    plt.savefig("result/target_age_gender.png", bbox_inches="tight")


def target_angina(data):
    """
    Take the data as a parameter and make a bar chart with the
    exercise induced angina(1 = yes, 0 = no) on the x-axis and the
    rate of heart attack on the y-axis. You should group the dataset
    by exercise induced angina(exang) and calculate the rate of heart attack
    for people who have it and who don’t. Label the y-axis as Rate of
    Heart Attack, the x-axis as Exercise Induced Angina: 0 = without angina,
    1 = with angina and title as the Rate of Heart Attack vs Exercise
    Induced Angina. Save your image to target_angina.png.
    """
    revised_data = data.loc[:, ["exang", "target"]]
    revised_data = revised_data.dropna()
    revised_data = revised_data.groupby(["exang"])["target"].mean()
    revised_data = revised_data.reset_index()
    fig = px.bar(revised_data, x="exang", y="target",
                 labels={"target": "Rate of Heart Attack", "exang":
                         "Exercise Induced Angina: 0 = without angina, " +
                         "1 = with angina"},
                 title="Rate of Heart Attack vs Exercise Induced Angina")
    fig.write_image("result/target_angina.png")


def target_blood_pressure(data):
    """
    Takes the data as a parameter and make a line plot of the rate of heart
    attack over the resting blood pressure values. You should group the dataset
    by resting blood pressure and calculate the rate of heart attack for each
    value (the mean of target). Label x-axis as Resting Blood Pressure(mm Hg),
    y-axis as Rate of Heart Attack and title as the Rate of Heart Attacks vs
    Resting Blood Pressure and save it in the file named
    target_blood_pressure.png.
    """
    revised_data = data.loc[:, ["trestbps", "target"]]
    revised_data = revised_data.dropna()
    revised_data = revised_data.groupby(["trestbps"])["target"].mean()
    revised_data = revised_data.reset_index()
    fig = px.line(revised_data, x="trestbps", y="target",
                  labels={"target": "Rate of Heart Attack",
                          "trestbps": "Resting Blood Pressure(mm Hg)"},
                  title="Rate of Heart Attacks vs Resting Blood Pressure")
    fig.write_image("result/target_blood_pressure.png")
