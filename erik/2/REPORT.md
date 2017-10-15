# Assignment 2 Delivery

Please follow the provided structure carefully and complete all of the sections described below. The parts that you need to complete are marked with *TODO*.

## Personal information

**Name**: Erik J. Sandal

List the GitHub usernames of your team here. If you are working alone, then write your GitHub username as the team leader and leave the list of additional team members empty.

**Team leader**: slippers1

**Additional team members**: vundere, chrystallic

If you are working as a part of a team, only the a single person, the *team leader* needs to submit the report. If that is not you, then do not edit anything below this point.

----

## Part A: Data exploration

It must be possible to click **[here](Exploring_data.ipynb)** and reach the notebook corresponding to Part A.  Running all its cells should fully proceed without errors.

Complete the table with the attribute name(s) chosen for the given requirements.

| **Requirement** | **Chosen attribute(s)** |
| -- | -- |
| *For one of those 3 attributes, plot in a single figure 2 boxplots, one per each of the 2 classes.* | 'SHOT_DIST' |
| *Choose 2 categorical attributes (different from the target) and plot each distribution in a histogram.* | 'SHOT_DIST', 'W' |
| *Choose 2 numeric attributes and compare them in a scatter plot, with different colors for each class.* | 'SHOT_DIST', 'CLOSE_DEF_DIST' |


## Part B: Decision tree implementation

1 - Report on the classification accuracy, as appearing for your team in the mandatory Kaggle leaderboard, of your predictions output. This submitted output must be the same file as produced by your decision tree.

Write the name of the corresponding output file in the table, as well as the name of the file with the source code that was used for producing that output file. These files should be pushed to your repository. (We use some example names; replace any of them by the actual name you gave it if necessary.)


| **Output file** | **Code** | **Accuracy** |
| -- | -- | -- |
| `data/dt_basketball.pred.csv` | `Decision_tree_basketball.ipynb` | '0.57887' |

2 - Identify the top attribute that is used by your decision tree classifier.

**Top attribute:** 'SHOT_DIST'

3 - Only if your final code implements any of the mentioned enhancements, and you observed an improvement by using it, report in a few words what is the enhancement. (Otherwise leave it as it is.)

**Possible enhancement:** 'Early stopping and checking for majority class when len(records) < 1600 resulted in a significant improvement.'


## Part C: Advanced classifiers

1 - Report on the classification accuracy, as appearing for your team in the optional Kaggle leaderboard, of your predictions output using an advanced classifier. This submitted output must be the same file as produced by the classifier you chose with some given parameter values.

Write the name of the corresponding output file in the table, as well as the name of the file with the source code that was used for producing that output file. These files should be pushed to your repository. (We use some example names; replace any of them by the actual name you gave it if necessary.)


| **Output file** | **Code** | **Accuracy** |
| -- | -- | -- |
| `data/classifier_basketball.pred.csv` | `Alternative_classifiers.ipynb` | *TODO* |


2 - Report the classifier and its parameter(s) settings for the submitted predictions (add more line pairs of *Parameter* and *Parameter value* if needed):
  - Classifier: *TODO*
  - Parameter: *TODO*
    - Parameter value = *TODO*
