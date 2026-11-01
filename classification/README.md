\# GitHub Issue Classification and Label Recommendation



\## Member 2 Module



This module is responsible for classifying GitHub issues into predefined categories and recommending suitable GitHub labels.



\## Categories



The classifier predicts six categories:



1\. Bug

2\. Feature

3\. Documentation

4\. Security

5\. Performance

6\. Question



\## Label Mapping



| Category | GitHub Label |

|---|---|

| Bug | bug |

| Feature | enhancement |

| Documentation | documentation |

| Security | security |

| Performance | performance |

| Question | question |



\## Approach



The original GitHub dataset did not contain clean target labels matching the six required categories.



For example, the dataset contained labels such as `rejected AI`, while many issues did not have labels corresponding to the six required categories.



Therefore, weak supervision was used to create initial category labels from issue titles and descriptions using interpretable keyword and pattern-based rules.



The generated categories were then used to train a machine learning classifier.



\## Machine Learning Pipeline



The classification pipeline uses:



\- Python

\- Pandas

\- Scikit-learn

\- TF-IDF Vectorization

\- Logistic Regression



TF-IDF converts issue text into numerical features, while Logistic Regression performs multi-class classification.



\### Pipeline Steps



1\. Load the cleaned GitHub issue dataset.

2\. Combine issue title and description into a text field.

3\. Generate weak category labels using keyword and pattern-based rules.

4\. Split the labelled data into training and testing sets.

5\. Convert text into numerical features using TF-IDF.

6\. Train a Logistic Regression classifier.

7\. Evaluate the model using classification metrics.

8\. Save the trained model and TF-IDF vectorizer.

9\. Use the trained model for new issue predictions.



\## Hybrid Classification



The final classifier combines two approaches.



\### 1. Rule-Based Classification



Strong keywords and patterns are used for obvious issues.



Examples:



\- "crash", "exception", "error" → Bug

\- "add support", "new feature" → Feature

\- "documentation", "docs", "README" → Documentation

\- "security", "vulnerability", "CVE" → Security

\- "slow", "latency", "memory leak" → Performance

\- question-style titles → Question



The rule-based layer is checked first because strong keywords can provide a clear indication of the issue category.



\### 2. Machine Learning Classification



If a strong rule does not match, the issue is passed to the TF-IDF + Logistic Regression model.



The machine learning model predicts the most likely category and also provides a prediction confidence.



This creates a hybrid classification system combining interpretable rules with machine learning.



\## Model Evaluation



The current model achieved:



\*\*79.47% accuracy on the weakly labelled evaluation set.\*\*



The model evaluation also includes precision, recall and F1-score for each category.



This accuracy should not be interpreted as accuracy against manually verified ground-truth labels because the training and evaluation labels were generated through weak supervision.



\### Current Dataset



The original dataset contained approximately:



\- 2,769 GitHub issues

\- 1,287 weakly labelled issues used for classification



The weakly labelled categories were:



\- Bug: 615

\- Documentation: 283

\- Feature: 191

\- Question: 158

\- Security: 27

\- Performance: 13



The Security and Performance categories contain fewer examples, which can affect their machine learning performance.



\## Files



classification/

│

├── issue\_classification.py

├── label\_recommender.py

├── README.md

│

├── models/

│   ├── issue\_classifier.pkl

│   └── tfidf\_vectorizer.pkl

│

└── results/

&#x20;   ├── classification\_report.txt

&#x20;   ├── classified\_issues.csv

&#x20;   └── confusion\_matrix.csv



\## How to Run



From the `classification` directory:



\### Train the Classifier



python issue\_classification.py



This loads the cleaned GitHub issue dataset, generates weak labels, trains the classifier, evaluates the model, and saves the trained model.



\### Run the Label Recommender



python label\_recommender.py



The recommender provides sample predictions and allows an issue title and description to be entered interactively.



\## Example



\### Input



Issue title:



Flask crashes when uploading large files



Issue description:



The application throws an exception when the uploaded file is very large.



\### Output



Predicted Category: Bug

Suggested Label: bug

Method: Rule-based

Matched Keyword: crashes

Confidence Level: High



\## Supported Label Recommendations



The system maps the predicted category to a GitHub label.



| Predicted Category | Suggested GitHub Label |

|---|---|

| Bug | bug |

| Feature | enhancement |

| Documentation | documentation |

| Security | security |

| Performance | performance |

| Question | question |



\## Limitations



The original dataset does not contain manually verified labels for the six target categories.



Therefore, the current system relies on weak supervision to create the initial training labels.



The model's reported accuracy is therefore an evaluation against weakly generated labels rather than manually verified ground-truth labels.



The Performance and Security categories also have fewer examples than the other categories, so their machine learning metrics are less stable.



The keyword-based rules may also fail when an issue describes a category without using the expected keywords.



\## Future Improvements



\- Create a manually verified labelled dataset.

\- Increase the number of examples for minority categories.

\- Improve the weak-supervision rules using more patterns and GitHub issue metadata.

\- Experiment with transformer-based text classifiers.

\- Improve label recommendation using existing GitHub labels and issue history.

\- Add confidence thresholds for uncertain predictions.

\- Integrate the classifier with the project backend and dashboard.

\- Evaluate the system using manually verified test data.



\## Project Contribution



This module represents the Issue Classification and Label Recommendation component of the GitHub Issue Intelligence project.



The main contribution includes:



\- Six-category issue classification.

\- Weak-supervision based training data generation.

\- TF-IDF feature extraction.

\- Logistic Regression multi-class classification.

\- Hybrid rule-based and machine-learning prediction.

\- GitHub label recommendation.

\- Classification evaluation and result generation.

\- Saved trained model and vectorizer for future integration.

