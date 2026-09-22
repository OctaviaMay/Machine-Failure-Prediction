import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models" / "binary"
MODELS_DIR_MULTI = PROJECT_ROOT / "models" / "multi"

TEST_DATA_PATH =PROJECT_ROOT / "data" / "split" / "binary"
TEST_DATA_PATH_MULTI =PROJECT_ROOT / "data" / "split" / "multi"

FEAT_IMP_PATH = PROJECT_ROOT / "assets" / "binary"
FEAT_IMP_PATH_MULTI = PROJECT_ROOT / "assets" / "multi"

PREDICTED_DATA_PATH = PROJECT_ROOT / "data" / "predict" / "binary"
PREDICTED_DATA_PATH_MULTI = PROJECT_ROOT / "data" / "predict" / "multi"

MODELS_DIR.mkdir(exist_ok=True)
MODELS_DIR_MULTI.mkdir(exist_ok=True)

TEST_DATA_PATH.mkdir(exist_ok=True)
TEST_DATA_PATH_MULTI.mkdir(exist_ok=True)

FEAT_IMP_PATH.mkdir(exist_ok=True)
FEAT_IMP_PATH_MULTI.mkdir(exist_ok=True)

PREDICTED_DATA_PATH.mkdir(exist_ok=True)
PREDICTED_DATA_PATH_MULTI.mkdir(exist_ok=True)

def run_prediction(model:type,
                    X_test:pd.DataFrame)->dict:

    y_pred = model.predict(X_test)
    y_predict_prob = model.predict_proba(X_test)[:,1]

    result = { 
                'y_predict_prob' : y_predict_prob
                }

    return result


def run_multiclass_prediction(model:type,
                    X_test:pd.DataFrame)->dict:

    y_prob = model.predict_proba(X_test)

    return { 'y_prob' : y_prob }

    

def get_feature_importance(pipeline:type,
                           modeling_mode = '')->dict:
  
    classifier = pipeline.named_steps['classifier']
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    # Extract the actual feature name from the split list, making it a string
    feature_names = [f.split('__')[-1] for f in feature_names]

    impts = classifier.feature_importances_

    fea_imp = pd.DataFrame({
        "feature": feature_names,
        "Importance": impts
    }).sort_values("Importance", ascending=False)

    print(fea_imp.sort_values("Importance", ascending=False).head(10))
    plt.figure(figsize=(10,6))
    sns.barplot(data = fea_imp,x=fea_imp['feature'],y=fea_imp['Importance'], alpha=0.5,color='gray')
    plt.xticks(rotation='vertical')
    plt.title(f"Features Importance in {classifier.__class__.__name__}")

    # save feature importance chart
    if modeling_mode == '':
        FILE_PATH = f"{FEAT_IMP_PATH}\\feature_importance_{classifier.__class__.__name__}.png"
    else:
        FILE_PATH = f"{FEAT_IMP_PATH_MULTI}\\feature_importance_{classifier.__class__.__name__}.png"

    plt.savefig(FILE_PATH, dpi=150, bbox_inches ='tight')

    plt.show()
    return fea_imp

def load_finalist_model(modeling_mode = '')->type:

    # Check if path exists
    if modeling_mode == '':
        model_path = MODELS_DIR
    else:
        model_path = MODELS_DIR_MULTI
    
    if os.path.exists(model_path):
        print(f"✅ Success: Model found at {os.path.abspath(model_path)}")
        loaded_models = {}

        for file in model_path.glob("*.pkl"):
            try:
                loaded_models[file.stem] = joblib.load(file)
                print(f"Loaded: {file.name}")
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")

        print(f"Total models loaded: {len(loaded_models)}")

    print(f"✅ {len(loaded_models)} model(s) successfully loaded.")

    return loaded_models

def load_test_data(modeling_mode = '') -> dict:

    if modeling_mode == '':
        X_test = pd.read_csv(TEST_DATA_PATH / "X_test.csv")
        X_train = pd.read_csv(TEST_DATA_PATH / "X_train.csv")
        y_test = pd.read_csv(TEST_DATA_PATH / "y_test.csv")
        y_train = pd.read_csv(TEST_DATA_PATH / "y_train.csv")
    else:
        X_test = pd.read_csv(TEST_DATA_PATH_MULTI / "X_test.csv")
        X_train = pd.read_csv(TEST_DATA_PATH_MULTI / "X_train.csv")
        y_test = pd.read_csv(TEST_DATA_PATH_MULTI / "y_test.csv")
        y_train = pd.read_csv(TEST_DATA_PATH_MULTI / "y_train.csv")
   
    print('test data successfully loaded.')

    return {'X_test' : X_test,
            'X_train' : X_train,
            'y_test' : y_test,
            'y_train' : y_train}

def load_predicted_value(modeling_mode = '') -> dict:
    # Check if path exists
    if modeling_mode == '':
        pvalue_path = PREDICTED_DATA_PATH
    else:
        pvalue_path = PREDICTED_DATA_PATH_MULTI
    
    if os.path.exists(pvalue_path):
        print(f"✅ Success: Predicted data found at {os.path.abspath(pvalue_path)}")

        loaded_data = {}

        for file in pvalue_path.glob("*.csv"):
            try:
                loaded_data[file.stem]=pd.read_csv(file)
                print(f"Loaded: {file.name}")
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")

        print(f"Total predicted data files loaded: {len(loaded_data)}")

    print(f"✅ {len(loaded_data)} predicted data files successfully loaded.")

    return loaded_data

