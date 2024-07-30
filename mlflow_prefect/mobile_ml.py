import argparse
from copy import deepcopy
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

import mlflow
import numpy as np
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from hyperopt import hp, space_eval

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner
import configparser
import shutil
from sklearn.pipeline import make_pipeline


@task
def load_datasets(train_file: str, test_file: str):
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    return train_df, test_df

@task
def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame, columns_to_scale: list):
    scaler = StandardScaler()
    train_df[columns_to_scale] = scaler.fit_transform(train_df[columns_to_scale])
    test_df[columns_to_scale] = scaler.transform(test_df[columns_to_scale])
    return train_df, test_df

@task
def split_train_validation(train_df: pd.DataFrame, target_column: str, test_size: float, random_state: int):
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

@task
def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        pickle.dump(obj, f_out)

@task
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)
    
@task
def preprocess(dicts, dv: DictVectorizer, fit_dv: bool = False):  
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

@task
def train_and_log_model(dict_train, y_train, dict_val, y_val, dict_test, y_test, params):

    SPACE = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    with mlflow.start_run():
        params2 = deepcopy(params)
        params = space_eval(SPACE, params)
        mlflow.log_params(params)

        pipeline = make_pipeline(
            DictVectorizer(),
            RandomForestRegressor(**params)
        )

        pipeline.fit(dict_train, y_train)
        y_pred_val = pipeline.predict(dict_val)

        params2 = space_eval(SPACE, params2)
        pipeline2 = make_pipeline(
            DictVectorizer(),
            RandomForestRegressor(**params)
        )        

        pipeline2.fit(dict_train, y_train)
        y_pred_test = pipeline2.predict(dict_test)

        # evaluate model on the validation and test sets
        valid_rmse = mean_squared_error(y_val, y_pred_val, squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, y_pred_test, squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        # log the pipeline  
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

@task
def promote_best_model(stage):  
    
    mlflow.set_experiment("mobile-price-experiment")
    client = MlflowClient()
    mlflow_model = client.get_latest_versions("mobile-price-regressor", stages=["None"])[0]
    run_id_best_model = mlflow_model.run_id
    print(run_id_best_model)

    model_version = mlflow_model.version
    new_stage = stage
    client.transition_model_version_stage(
        name="mobile-price-regressor",
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )

    from datetime import datetime

    date = datetime.today().date()
    client.update_model_version(
        name="mobile-price-regressor",
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    )

@task
def hpo(X_train, y_train, X_valid, y_valid, num_trials):
    mlflow.set_experiment("random-forest-hyperopt")
    def objective(params):        
        with mlflow.start_run():
            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)  # for reproducible results
    best_result = fmin(
                    fn=objective,
                    space=search_space,
                    algo=tpe.suggest,
                    max_evals=num_trials,
                    trials=Trials(),
                    rstate=rstate
                )

def write_config(best_run_id, dv_full_path, artifact_uri):
    config = configparser.ConfigParser()
    config.read('../config.config')
    d1 = dict(config['DEFAULT'])
    d2 = {'best_run_id': best_run_id, 'dv_full_path': dv_full_path, 'S3_BUCKET': os.getenv('S3_BUCKET', ''), 'ARTIFACT_URI': artifact_uri}
    config['DEFAULT'] = {**d1, **d2}

    with open('../config.config', 'w') as configfile:
        config.write(configfile)
    shutil.copy('../config.config', '../web-service')

def read_config():
    config = configparser.ConfigParser()
    config.read('../config.config')
    TRACKING_SERVER_HOST = config['DEFAULT']['TRACKING_SERVER_HOST']
    AWS_PROFILE = config['DEFAULT']['AWS_PROFILE']
    return TRACKING_SERVER_HOST, AWS_PROFILE


@flow(task_runner=SequentialTaskRunner())
def main_flow(train_file: str, test_file: str, columns_to_scale: list, target_column: str, test_size: float, random_state: int, dest_path: str, num_trials_hpo=50, log_top_best_models=5):

    # 加载数据集
    train_df, test_df = load_datasets(train_file, test_file).result()

    # 规范化特征
    train_df, test_df = normalize_features(train_df, test_df, columns_to_scale).result()

    # 划分训练和验证集
    X_train, X_val, y_train, y_val = split_train_validation(train_df, target_column, test_size, random_state).result()

    # 保存数据
    os.makedirs(dest_path, exist_ok=True)
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_val, y_val), os.path.join(dest_path, "valid.pkl"))
    dump_pickle(test_df, os.path.join(dest_path, "test.pkl"))

    # ***Train with hyperparameter optimization***
    TRACKING_SERVER_HOST, AWS_PROFILE = read_config()
    if TRACKING_SERVER_HOST != '':    
        mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
        os.environ["AWS_PROFILE"] = AWS_PROFILE
    else:
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

    X_train, y_train = load_pickle(os.path.join(dest_path, "train.pkl")).result()
    X_valid, y_valid = load_pickle(os.path.join(dest_path, "valid.pkl")).result()

    best_result = hpo(X_train, y_train, X_valid, y_valid, num_trials=num_trials_hpo).result()

    # ***Train and register best model***
    HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
    EXPERIMENT_NAME = "random-forest-best-models"

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog()
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top_best_models,
        order_by=["metrics.rmse ASC"]
    )
  
    for run in runs:
        train_and_log_model(X_train, y_train, X_valid, y_valid, test_df, test_df[target_column], params=run.data.params)
   
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=2,
        order_by=["metrics.rmse ASC"]
    )[0]

    # register the best model
    model_uri = f"runs:/{best_run.info.run_id}/models"
    mlflow.register_model(model_uri=model_uri, name="mobile-price-regressor")
    os.environ["BEST_RUN_ID"] = best_run.info.run_id
    run_id = os.getenv('BEST_RUN_ID', '')
    print(f'Best run id: {run_id}')

    path_art = mlflow.artifacts.download_artifacts(run_id=best_run.info.run_id, dst_path="./mlflow_artifacts")
    print(path_art)
    shutil.copy('./mlflow_artifacts/model/model.pkl', '../web-service')


    # save the id of the best run and the path to the dictionary vectorizer
    dv_abs_path = os.path.abspath(os.path.join(dest_path, "dv.pkl"))
    os.environ["DV_FULL_PATH"] = dv_abs_path
    artifact_uri = best_run.info.artifact_uri
    write_config(str(best_run.info.run_id), str(dv_abs_path), str(artifact_uri))

    # Promote the best model    
    promote_best_model("Staging")

# 设置参数
train_file = '../data/train.csv'
test_file = '../data/test.csv'
columns_to_scale = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w']
target_column = 'price_range'
test_size = 0.2
random_state = 42
dest_path = './output'

# 运行主流程
main_flow(train_file, test_file, columns_to_scale, target_column, test_size, random_state, dest_path)
