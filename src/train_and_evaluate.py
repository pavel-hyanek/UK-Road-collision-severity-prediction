import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import average_precision_score, precision_recall_curve, brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import joblib
from scipy.stats import bootstrap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

train = pd.read_parquet(PROJECT_ROOT / 'data/preprocessed/train_final.parquet')
test = pd.read_parquet(PROJECT_ROOT / 'data/preprocessed/test_final.parquet')

columns = train.columns.tolist()
vehicle_columns = [x for x in columns if x.startswith("VEHICLE_")]
collision_columns = [x for x in columns if x not in vehicle_columns+["severity", "collision_index"]]

X_train_A = train[collision_columns]
X_train_B = train[collision_columns+vehicle_columns]
X_test_A = test[collision_columns]
X_test_B = test[collision_columns+vehicle_columns]

y_train = train["severity"]
y_test = test["severity"]


def evaluate_models(model_A, model_B, X_test_A=X_test_A,
                   X_test_B=X_test_B, y_test=y_test, project_root=PROJECT_ROOT):

       model_A_pred_proba = model_A.predict_proba(X_test_A)[:,1]
       model_B_pred_proba = model_B.predict_proba(X_test_B)[:,1]

       model_A_avg_prec_sc = average_precision_score(y_test, model_A_pred_proba)
       model_B_avg_prec_sc = average_precision_score(y_test, model_B_pred_proba)

       model_A_brier_score = brier_score_loss(y_test, model_A_pred_proba)
       model_B_brier_score = brier_score_loss(y_test, model_B_pred_proba)

       #print(f"Average precision scores of random forest classifier models: \n Trained on feature set A: {round(model_A_avg_prec_sc, 4)}" +
       #       f"\n Trained on feature set B: {round(model_B_avg_prec_sc, 4)}"+
       #       f"Brier scores of random forest classifier models: \n Trained on feature set A: {round(model_A_brier_score, 4)}" +
       #       f"\n Trained on feature set B: {round(model_B_brier_score, 4)}")

       precisions_model_A, recalls_model_A, thresholds_model_A = precision_recall_curve(y_test, model_A_pred_proba)
       precisions_model_B, recalls_model_B, thresholds_model_B = precision_recall_curve(y_test, model_B_pred_proba)

       plt.plot(recalls_model_A, precisions_model_A, label="Trained on feature set A")
       plt.plot(recalls_model_B, precisions_model_B, label="Trained on feature set B")

       plt.axhline(y=y_test.mean(), label=f"Dummy baseline ", linestyle="--", color="grey")
       plt.xlabel("Recall")
       plt.ylabel("Precision")
       plt.legend()
       plt.title("Precision-recall curve of final random forest classifier models")
       plt.grid()
       plt.xlim((0,1))
       plt.savefig(project_root / 'outputs/precision_recall_curve_final.png')

       boot = bootstrap(data=(y_test, model_A_pred_proba, model_B_pred_proba), 
                        statistic = (lambda y_test, model_A_pred_proba, model_B_pred_proba: 
                                  average_precision_score(y_test, model_B_pred_proba) - average_precision_score(y_test, model_A_pred_proba)), 
                        confidence_level=0.95, paired=True, n_resamples=1000, method="percentile", random_state=42)

       results = pd.DataFrame({"AP score" : [round(model_A_avg_prec_sc,4), round(model_B_avg_prec_sc,4)], 
                               "Brier score" : [round(model_A_brier_score,4), round(model_B_brier_score,4)]},
                               index=["Trained on feature set A", "Trained on feature set B"])
       results.to_csv(project_root / 'outputs/evaluation_metrics.csv')

       A_B_comparison = pd.DataFrame({"AP on B - AP on A" : [round(model_B_avg_prec_sc - model_A_avg_prec_sc,4)],
                                  "95% CI lower bound for AP on B - AP on A" : [round(float(boot.confidence_interval[0]),4)],
                                  "95% CI upper bound for AP on B - AP on A" : [round(float(boot.confidence_interval[1]),4)]},
                                  index=[""])
       A_B_comparison.to_csv(project_root / 'outputs/comparison_of_A_B_models.csv')

       plt.clf()
       plt.hist(boot.bootstrap_distribution, bins=30, edgecolor="grey")
       plt.xlabel("AP on B - AP on A")
       plt.ylabel("Number of bootstrapped instances out of 1000")
       plt.title("Histogram of the diference between AP scores on feature \nsets B and A paired-bootstrap resampled 1000 times")
       plt.savefig(project_root / 'outputs/AP_difference_bootstrap_histogram.png')


rfc_A = RandomForestClassifier(random_state=42, max_depth=40, max_features=7, 
                               min_samples_leaf=15, n_estimators=70)
rfc_A.fit(X_train_A, y_train)

rfc_B = RandomForestClassifier(random_state=42, n_estimators=150, 
                               min_samples_leaf=3, max_features=14, max_depth=30)
rfc_B.fit(X_train_B, y_train)

evaluate_models(rfc_A, rfc_B)

joblib.dump(rfc_A, PROJECT_ROOT / "models/final_rfc_model_A.joblib")
joblib.dump(rfc_B, PROJECT_ROOT / "models/final_rfc_model_B.joblib")