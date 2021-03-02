"""
Lauren Liao
CSE 163 AB

The file that reads dataset needed and calls
functions implemented for Final Project.
"""
import sys
import pandas as pd
import final_project_lauren as lr
import final_project_testing as ts

EXPECTED_MAJOR = 3
EXPECTED_MINOR = 7


def main():
    print('Main runs')

    version = sys.version_info
    if version.major != EXPECTED_MAJOR or version.minor != EXPECTED_MINOR:
        print('⚠️  Warning! Detected Python version '
              f'{version.major}.{version.minor} but expected version '
              f'{EXPECTED_MAJOR}.{EXPECTED_MINOR}')

    heart_data = pd.read_csv("data/datasets_heart.csv")
    model = lr.model(heart_data)
    print(model)
    ts.model_tune(heart_data)
    lr.target_age_gender(heart_data)
    ts.target_gender_test(heart_data)
    lr.target_angina(heart_data)
    ts.target_chest_pain(heart_data)
    lr.target_blood_pressure(heart_data)


if __name__ == '__main__':
    main()
