"""Hyperparameter tuning for CreditGuard-AI"""
# (See chat for explanation.)
import json, joblib, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from config import MODEL_DIR,ARTIFACT_DIR,RANDOM_STATE,TARGET_COLUMN,TEST_SIZE,TRAIN_DATA
from preprocessing import CreditRiskPreprocessor
from feature_engineering import FeatureEngineer
from threshold_optimizer import ThresholdOptimizer
from evaluate import ModelEvaluator

def main():
    print("Loading dataset...")
    df=pd.read_csv(TRAIN_DATA)
    print("Dataset Path:", TRAIN_DATA)
    print("Dataset Shape:", df.shape)
    X=df.drop(columns=[TARGET_COLUMN]); y=df[TARGET_COLUMN]
    X=FeatureEngineer().transform(X)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=TEST_SIZE,stratify=y,random_state=RANDOM_STATE)
    pre=CreditRiskPreprocessor()
    X_train=pre.fit_transform(X_train)
    X_test=pre.transform(X_test)
    model=XGBClassifier(objective="binary:logistic",eval_metric="auc",tree_method="hist",random_state=RANDOM_STATE,n_jobs=-1)
    params={
      "max_depth": [4,5,6],
      "learning_rate": [0.02,0.03,0.04,0.05],
      "n_estimators": [400,500,600,700],
      "subsample":[0.8,0.9,1.0],
      "colsample_bytree":[0.6,0.7,0.8],
      "min_child_weight":[2,3,4],
      "gamma":[0.3,0.5,0.7],
      "reg_alpha":[0,0.05,0.1],
      "reg_lambda":[3,5,7],
      "scale_pos_weight":[10,11.3,12,13]
    }
    search=RandomizedSearchCV(model,param_distributions=params,n_iter=100,cv=3,
        scoring="roc_auc",verbose=3,random_state=RANDOM_STATE,n_jobs=-1,return_train_score=True)
    search.fit(X_train,y_train)
    pd.DataFrame(search.cv_results_).to_csv(ARTIFACT_DIR/"xgboost_tuning_results.csv",index=False)
    with open(ARTIFACT_DIR/"best_params.json","w") as f: json.dump(search.best_params_,f,indent=4)
    best=search.best_estimator_
    probs=best.predict_proba(X_test)[:,1]
    th,table=ThresholdOptimizer().find_best_threshold(y_test,probs)
    preds=(probs>=th).astype(int)
    table.to_csv(ARTIFACT_DIR/"threshold_results.csv",index=False)
    ev=ModelEvaluator(ARTIFACT_DIR)
    ev.save_confusion_matrix(y_test,preds)
    ev.save_roc_curve(y_test,probs)
    ev.save_precision_recall_curve(y_test,probs)
    print(search.best_params_)
    print("Threshold:",th)
    print("Accuracy",accuracy_score(y_test,preds))
    print("Precision",precision_score(y_test,preds))
    print("Recall",recall_score(y_test,preds))
    print("F1",f1_score(y_test,preds))
    print("ROC",roc_auc_score(y_test,probs))
    joblib.dump(best,MODEL_DIR/"best_xgboost.pkl")
    joblib.dump(pre,MODEL_DIR/"preprocessor.pkl")
if __name__=="__main__":
    main()
