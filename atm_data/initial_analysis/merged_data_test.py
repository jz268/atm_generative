import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
from pathlib import Path
import pickle



dtype = {
    'Year': 'Int16', 
    'Quarter': 'Int8', 
    'Month': 'Int8', 
    'DayofMonth': 'Int8', 
    'DayOfWeek': 'Int8', 
    'FlightDate': 'str',

    'Reporting_Airline': 'str', 

    'CRSDepTime': 'Int16', 
    'DepTime': 'Int16', 
    'DepDelay': 'Int16', 
    'DepDelayMinutes': 'Int16',
    'CRSDepTimeMinutes': 'Int16',
    'DepTimeMinutes': 'Int16',

    'CRSArrTime': 'Int16', 
    'ArrTime': 'Int16', 
    'ArrDelay': 'Int16', 
    'ArrDelayMinutes': 'Int16',
    'CRSArrTimeMinutes': 'Int16',
    'ArrTimeMinutes': 'Int16',

    'CarrierDelay': 'Int16', 
    'WeatherDelay': 'Int16', 
    'NASDelay': 'Int16', 
    'SecurityDelay': 'Int16', 
    'LateAircraftDelay': 'Int16',

    'Diverted': 'Int8',
    'DivArrDelay': 'Int16', 
    'DivDistance': 'Int16',
}

abs_time_cols = ['CRSDepTimeAbsolute', 'CRSArrTimeAbsolute', 
                 'DepTimeAbsolute', 'ArrTimeAbsolute']

cols = list(dtype.keys()) + abs_time_cols


if not Path('tmp.parquet').is_file():
    print("merging schedule and weather data...")

    schedule = pd.read_csv('data/schedule/lax_to_jfk_full_cleaned.csv', 
                    dtype=dtype, usecols=cols, parse_dates=abs_time_cols)
    merged = schedule

    # idk
    merged['ArrDelay'] = merged['ArrTimeMinutes'] - merged['CRSArrTimeMinutes']
    merged['DepDelay'] = merged['DepTimeMinutes'] - merged['CRSDepTimeMinutes']
    merged['NewDelay'] = merged['ArrDelay'] - merged['DepDelay']
    merged['NewDelayMinutes'] = merged['NewDelay'].clip(lower=0)

    weather_lax = pd.read_csv('data/noaa_lcdv2/LCD_LAX_1987-2023_CLEANED.csv', 
                            parse_dates=['DATE']).rename(columns=lambda x:f'LAX_{x}')
    weather_jfk = pd.read_csv('data/noaa_lcdv2/LCD_JFK_1987-2023_CLEANED.csv', 
                            parse_dates=['DATE']).rename(columns=lambda x:f'JFK_{x}')

    merged = pd.merge_asof(
        merged.sort_values('DepTimeAbsolute'), 
        weather_lax.sort_values('LAX_DATE'), 
        left_on='DepTimeAbsolute',
        right_on='LAX_DATE',
        allow_exact_matches=True, 
        direction='nearest')

    # merged.to_csv('test_0.csv')

    merged = pd.merge_asof(
        merged.sort_values('ArrTimeAbsolute'), 
        weather_jfk.sort_values('JFK_DATE'), 
        left_on='ArrTimeAbsolute',
        right_on='JFK_DATE',
        allow_exact_matches=True, 
        direction='nearest')

    # merged.to_csv('test_1.csv')

    merged = merged.fillna(0)

    merged.to_parquet('tmp.parquet')

else:
    print("reading saved merged schedule and weather data...")
    merged = pd.read_parquet('tmp.parquet')

merged_cols = merged.columns.values.tolist()
# print(merged_cols)

date_cols = [
    'CRSArrTimeAbsolute', 'ArrTimeAbsolute', 
    'CRSDepTimeAbsolute', 'DepTimeAbsolute',
    'LAX_DATE', 'JFK_DATE', 'FlightDate', 
    
]

response_cols = [
    'ArrDelay', 'ArrDelayMinutes', 'DepDelay', 'DepDelayMinutes',
    'CarrierDelay', 'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay', 
    'Diverted', 'DivDistance', 'DivArrDelay',
    'NewDelay', 'NewDelayMinutes'
]

misc_ignore = [
    'ArrTime', 'CRSArrTime', 'DepTime', 'CRSDepTime', 
    'LAX_HourlyWindDirectionCos', 'LAX_HourlyWindDirectionSin', #'LAX_HourlyWindDirection',
    'JFK_HourlyWindDirectionCos', 'JFK_HourlyWindDirectionSin', #'JFK_HourlyWindDirection',
    'LAX_HourlyPressureTendencyIncr', 'LAX_HourlyPressureTendencyDecr', 'LAX_HourlyPressureTendencyCons',
    'JFK_HourlyPressureTendencyIncr', 'JFK_HourlyPressureTendencyDecr', 'JFK_HourlyPressureTendencyCons',
    'DayofMonth', 'Quarter', 'Year', #'Month', 'DayOfWeek',
    #'Reporting_Airline', 
    'ArrTimeMinutes', #'DepTimeMinutes'
]

explanatory_cols = [col for col in merged_cols 
                    if (col not in response_cols 
                    and col not in date_cols 
                    and col not in misc_ignore)]

make_plots = False
if make_plots:
    df = merged
    # plotting some stuff

    p = Path("media/hist/")
    if not p.is_dir():
        p.mkdir(parents=True, exist_ok=True)

    plt.figure()
    for col in tqdm(merged_cols):
        plt.clf()
        plt.hist(x=df[col], bins='auto')
        plt.title(f'Histogram for {col} (N = {len(df.index)})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'media/hist/Histogram_{col}.png')
    plt.close()

    plt.figure()
    for col in tqdm(response_cols):
        plt.clf()
        tmp = df.loc[df[col]>0, col]
        plt.hist(x=tmp, bins='auto')
        plt.title(f'Histogram for Positive {col} (N = {len(tmp.index)})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'media/hist/Positive_Histogram_{col}.png')
    plt.close()

    plt.figure()
    for col in tqdm(response_cols):
        plt.clf()
        tmp = df.loc[df[col]>=15, col]
        plt.hist(x=tmp, bins='auto')
        plt.title(f'Histogram for Late {col} (N = {len(tmp.index)})')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'media/hist/Late_Histogram_{col}.png')
    plt.close()

    plt.figure()
    for rcol in tqdm(response_cols, position=0, desc="Y", leave=False):
        p = Path(f"media/scatter/{rcol}/")
        if not p.is_dir():
            p.mkdir(parents=True, exist_ok=True)
        # for ecol in tqdm(explanatory_cols, position=1, desc="X", leave=False):
        for ecol in tqdm(merged_cols, position=1, desc="X", leave=False):
            # print(rcol, ecol)
            plt.clf()
            # corr = df[rcol].corr(df[ecol])
            # plt.title(f'{rcol} vs. {ecol}, ρ={corr:.4f}')
            plt.title(f'{rcol} vs. {ecol}')
            plt.xlabel(ecol)
            plt.ylabel(rcol)
            plt.scatter(merged[ecol], merged[rcol])
            plt.savefig(f'media/scatter/{rcol}/Y={rcol}_X={ecol}.png')
    plt.close()







from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb
from pprint import pprint
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe


# df = merged.loc[merged['ArrDelay']>=15]
df = merged

# # idk
# df['ArrDelay'] = df['ArrTimeMinutes'] - df['CRSArrTimeMinutes']
# df['DepDelay'] = df['DepTimeMinutes'] - df['CRSDepTimeMinutes']
# df['NewDelay'] = df['ArrDelay'] - df['DepDelay']

# plt.figure()
# plt.title('ArrDelay vs. DepDelay')
# plt.xlabel('DepDelay')
# plt.ylabel('ArrDelay')
# plt.scatter(df['ArrDelay'], df['DepDelay'])
# plt.savefig('fig.png')
# plt.clf()

df['YearCat'] = df['Year']
# explanatory_cols.append('YearCat')

categorical_vars = [
    'Reporting_Airline',  
    'LAX_HourlyPressureTendency', 'JFK_HourlyPressureTendency',
    'Quarter', 'Month', 'DayOfWeek', 'YearCat'
]
for col in categorical_vars:
    df[col] = df[col].astype("category")

responses = [
    'ArrDelay', 
    'DepDelay', 
    'WeatherDelay', 
    'NASDelay', 
    'NewDelay'
]

df_full = df

for response in responses:

    print(f"\n=============  {response}  =============\n")

    # df = df_full.loc[merged[response]>0]

    # X_train = df.loc[df['Year'].between(2012,2015,inclusive='both'), explanatory_cols]
    # X_test = df.loc[df['Year'].between(2016,2016,inclusive='both'), explanatory_cols]
    # y_train = df.loc[df['Year'].between(2012,2015,inclusive='both'), response]
    # y_test = df.loc[df['Year'].between(2016,2016,inclusive='both'), response]

    # X = df.loc[df['Year'].between(2000,2019,inclusive='both'), explanatory_cols]
    # y = df.loc[df['Year'].between(2000,2019,inclusive='both'), response]
    X = df.loc[df['Year'] >= 2003, explanatory_cols]
    y = df.loc[df['Year'] >= 2003, response]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
    eval_set = [(X_train, y_train), (X_test, y_test)]

    def reg_from_autotuned_hyperparams():
        space = {
            'n_estimators': hp.quniform('n_estimators', 10, 500, 10),
            'max_depth': hp.quniform("max_depth", 3, 10, 1),
            'gamma': hp.uniform ('gamma', 0,2),
            'min_child_weight' : hp.quniform('min_child_weight', 0, 1, 1),
            'reg_alpha' : hp.quniform('reg_alpha', 0,5,1),
            # 'reg_lambda' : hp.uniform('reg_lambda', 0,1),
            'colsample_bytree' : hp.uniform('colsample_bytree', 0.7,1),
        }

        def reg_from_hyperparams(space):
            reg=xgb.XGBRegressor(
                n_estimators = int(space['n_estimators']), 
                max_depth = int(space['max_depth']), 
                gamma = space['gamma'],
                min_child_weight=int(space['min_child_weight']),
                reg_alpha = int(space['reg_alpha']),
                # reg_lambda = int(space['reg_alpha']),
                colsample_bytree=int(space['colsample_bytree']),
                early_stopping_rounds=10,
                seed = 0,
                enable_categorical=True)
            return reg

        def objective(space):
            reg=reg_from_hyperparams(space)

            reg.fit(X_train, y_train,
                    eval_set=eval_set,
                    verbose=False)
            
            pred = reg.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            print (f"rmse: {rmse}")
            return {'loss': rmse, 'status': STATUS_OK }

        trials = Trials()

        best_hyperparams = fmin(fn = objective,
                            space = space,
                            algo = tpe.suggest,
                            max_evals = 50,
                            trials = trials)
        # best_hyperparams['n_estimators'] = space['n_estimators']
        
        print(f"best hyperparams are:\n{best_hyperparams}")

        reg = reg_from_hyperparams(best_hyperparams)
        return reg
    
    # reg = reg_from_autotuned_hyperparams()

    reg=xgb.XGBRegressor(
        n_estimators = 5000, 
        max_depth = 5, 
        gamma = .9,
        min_child_weight = 1,
        reg_alpha = 5,
        colsample_bytree = .75,
        early_stopping_rounds = 10,
        learning_rate = .1,
        seed = 0,
        enable_categorical=True)

    fit = reg.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    plt.figure()
    results = reg.evals_result()
    train_rmse = results['validation_0']['rmse']
    test_rmse = results['validation_1']['rmse']
    epochs = range(0, len(train_rmse))
    plt.plot(epochs, train_rmse, label='train')
    plt.plot(epochs, test_rmse, label='test')
    plt.title(f'rmse vs. epoch ({response})')
    plt.legend(loc="upper right")
    plt.xlabel("epoch")
    plt.ylabel("rmse")
    plt.savefig(f'rmse_results_{response}')
    plt.clf()

    rmse = np.sqrt(mean_squared_error(y_train, reg.predict(X_train)))
    score = reg.score(X_train,y_train)
    print(f"   train -- score: {score:.4f}; rmse: {rmse:.2f}")
    rmse = np.sqrt(mean_squared_error(y_test, reg.predict(X_test)))
    score = reg.score(X_test,y_test)
    print(f"    test -- score: {score:.4f}; rmse: {rmse:.2f}\n")

    for year in range(1987, 2021):
        # if year <= 2003 or year == 2017: continue
        X = df.loc[df['Year']==year, explanatory_cols]
        y = df.loc[df['Year']==year, response]
        if len(X.index) == 0: 
            score = 'n/a'
            rmse = 'n/a'
        else:
            score = f'{reg.score(X,y):.4f}'
            rmse = f'{np.sqrt(mean_squared_error(y, reg.predict(X))):.2f}'
        print(f'year: {year}; score: {score}; rmse: {rmse}'
            # + (' [train]' if (2012 <= year <= 2015) else ' [test]' if (2016 <= year <= 2016) else '')
            # + (' [incomplete?]' if year == 2003 or year == 2017 else '')
            # + (' [train/test]' if 1999 < year < 2021 else '')
        )

    # pprint(reg.get_booster().get_score(importance_type='gain'))
    # pprint(reg.get_booster().get_score(importance_type='weight'))


    plt.figure()
    feature_importance = reg.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    fig = plt.figure(figsize=(10, 10))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(X_test.columns)[sorted_idx])
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(f'features_{response}.png')
    plt.clf()

    # save
    with open(f'reg_{response}.pkl','wb') as f:
        pickle.dump(reg,f)