import numpy as np
import pandas as pd
import pickle as pkl

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer

import xgboost as xgb

##########################
### DATA PREPROCESSING ###
##########################


def custom_to_numeric(value):
    if isinstance(value, str):
        if value[-1] == '_':
            return float(value[:-1])
    return float(value)


def month_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    months_mapping = {
        'January': 1,
        'February': 2,
        'March': 3,
        'April': 4,
        'May': 5,
        'June': 6,
        'July': 7,
        'August': 8,
        'September': 9,
        'October': 10,
        'November': 11,
        'December': 12
    }

    df['month_number'] = df.month.map(months_mapping)
    return df


def name_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['name'] = df[['customer_id', 'name']].groupby('customer_id').bfill()
    df['name'] = df[['customer_id', 'name']].groupby('customer_id').ffill()
    return df


def num_of_loan_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['num_of_loan'] = df['num_of_loan'].apply(custom_to_numeric)
    df.loc[df['num_of_loan'] == -100, 'num_of_loan'] = np.nan
    df['num_of_loan'] = \
        df[['customer_id', 'num_of_loan', 'month_number']].sort_values('month_number').groupby('customer_id').\
        ffill().bfill().sort_index()['num_of_loan']
    return df


def type_of_loan_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    all_loans_strings = list(df['type_of_loan'].dropna().unique())
    unique_types = set()
    for string_ in all_loans_strings:
        for type_ in string_.split(','):
            n_type = type_.replace(" and ", "").strip()
            unique_types.add(n_type)

    no_loans_index = df[df['num_of_loan'] == 0.0].index
    df.loc[no_loans_index, ['type_of_loan']] = "No loans"
    df.type_of_loan = df.type_of_loan.fillna('Not Specified')

    for utype in unique_types:
        new_column_name = f"loan_type_{utype.lower().replace(' ', '_')}"
        df[new_column_name] = df.type_of_loan.str.contains(utype).astype(int)
    return df


def credit_history_age_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    def credit_history_age_to_int(value: str):
        if pd.isnull(value):
            return value
        years_str, months_str = value.split(' and ')
        years = int(years_str.split(' ')[0])
        months = int(months_str.split(' ')[0])
        return 12 * years + months

    def fill_credit_history_age(group):
        group = group.sort_values(by='month_number').reset_index(drop=True)

        start_index = group['credit_history_age'].first_valid_index()
        if start_index is not None:
            start_age = group.loc[start_index, 'credit_history_age']
            start_month = group.loc[start_index, 'month_number']

            group['credit_history_age'] = group.apply(
                lambda row: start_age + row['month_number'] - start_month if pd.isna(row['credit_history_age']) else
                row['credit_history_age'],
                axis=1
            )
        return group

    df['credit_history_age'] = df['credit_history_age'].apply(credit_history_age_to_int)
    df = df.groupby('customer_id', group_keys=False).apply(fill_credit_history_age)
    return df


def num_credit_inquiries_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['num_credit_inquiries'] = df.groupby('customer_id')['num_credit_inquiries'].transform(
        lambda x: x.fillna(x.median()))

    df['num_credit_inquiries'] = np.log1p(df['num_credit_inquiries'])
    return df


def num_of_delayed_payment_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['num_of_delayed_payment'] = df['num_of_delayed_payment'].apply(custom_to_numeric)
    df.loc[df['num_of_delayed_payment'] > 28, 'num_of_delayed_payment'] = np.nan
    df['num_of_delayed_payment'] = df.groupby('customer_id')['num_of_delayed_payment'].transform(
        lambda x: x.fillna(int(x.mean())))
    return df


def amount_invested_monthly_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['amount_invested_monthly'] == '__10000__', 'amount_invested_monthly'] = np.nan
    df['amount_invested_monthly'] = np.float64(df['amount_invested_monthly'])
    df['amount_invested_monthly'] = df.groupby('customer_id')['amount_invested_monthly'].transform(
        lambda x: x.fillna(x.mean()))
    return df


def monthly_balance_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df['monthly_balance'] == '__-333333333333333333333333333__', 'monthly_balance'] = np.nan
    df['monthly_balance'] = np.float64(df['monthly_balance'])
    df['monthly_balance'] = df.groupby('customer_id')['monthly_balance'].transform(lambda x: x.fillna(x.mean()))
    return df


def age_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df['age'] = df['age'].apply(custom_to_numeric)
    df.loc[(df.age < 14) | (df.age > 56), 'age'] = np.nan
    df['age'] = df.groupby('customer_id')['age'].transform(lambda x: x.fillna(x.median()))
    return df


def ssn_area_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.ssn == '#F%$D@*&8', 'ssn'] = np.nan
    df.ssn = df.groupby('customer_id')['ssn'].transform(lambda x: x.fillna(x.mode().iloc[0]))
    df['ssn_area'] = df.ssn.apply(lambda x: x.split('-')[0])
    return df


def occupation_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.occupation == '_______', 'occupation'] = np.nan
    df.occupation = df.groupby('customer_id')['occupation'].transform(lambda x: x.fillna(x.mode().iloc[0]))
    return df


def changed_credit_limit_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.changed_credit_limit == '_', 'changed_credit_limit'] = np.nan
    df.changed_credit_limit = np.float64(df.changed_credit_limit)
    df.changed_credit_limit = df.groupby('customer_id')['changed_credit_limit'].transform(
        lambda x: x.fillna(x.mode().iloc[0]))
    return df


def payment_behaviour_column_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df.loc[df.payment_behaviour == '!@9#%8', 'payment_behaviour'] = np.nan
    df.payment_behaviour = df.groupby('customer_id')['payment_behaviour'].transform(
        lambda x: x.fillna(x.mode().iloc[0]))
    return df


def preprocess_types_and_nulls(df: pd.DataFrame) -> pd.DataFrame:
    df = month_column_preprocessing(df)
    df = name_column_preprocessing(df)
    df['annual_income'] = df['annual_income'].apply(custom_to_numeric)
    df['monthly_inhand_salary'] = df['monthly_inhand_salary'].fillna(df['annual_income']/12)
    df = num_of_loan_column_preprocessing(df)
    df = type_of_loan_column_preprocessing(df)
    df = credit_history_age_column_preprocessing(df)
    df = num_credit_inquiries_column_preprocessing(df)
    df = num_of_delayed_payment_column_preprocessing(df)
    df = amount_invested_monthly_column_preprocessing(df)
    df = monthly_balance_column_preprocessing(df)
    df = age_column_preprocessing(df)
    df = ssn_area_column_preprocessing(df)
    df = occupation_column_preprocessing(df)
    df = changed_credit_limit_column_preprocessing(df)
    df['outstanding_debt'] = df['outstanding_debt'].apply(custom_to_numeric)
    df = payment_behaviour_column_preprocessing(df)
    return df


def delete_redundant_сolumns(df: pd.DataFrame) -> pd.DataFrame:
    # general
    del df['id']
    del df['customer_id']
    del df['ssn']
    del df['name']
    del df['type_of_loan']

    # categorical
    del df['ssn_area']
    del df['payment_behaviour']
    del df['month']
    del df['occupation']
    return df


def preprocess_data(initial_dataset_path: str):
    df = pd.read_csv(initial_dataset_path)
    df.columns = df.columns.str.lower()
    df = preprocess_types_and_nulls(df)
    df = delete_redundant_сolumns(df)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    df_train, df_test = train_test_split(df, test_size=test_size, random_state=179)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.credit_score
    y_test = df_test.credit_score

    del df_train['credit_score']
    del df_test['credit_score']

    dv = DictVectorizer(sparse=False)

    X_train = dv.fit_transform(df_train.to_dict(orient='records'))
    X_test = dv.transform(df_test.to_dict(orient='records'))
    features_names = list(dv.get_feature_names_out())

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train.map({'Good': 0, 'Poor': 1, 'Standard': 2}),
        feature_names=features_names
    )

    dtest = xgb.DMatrix(
        X_test,
        label=y_test.map({'Good': 0, 'Poor': 1, 'Standard': 2}),
        feature_names=features_names
    )

    return dtrain, dtest, dv


def train_the_model(dtrain):
    xgb_params = {
        'eta': 0.3,
        'max_depth': 15,
        'min_child_weight': 1,
        'num_class': 3,
        'objective': 'multi:softmax',
        'seed': 1,
        'verbosity': 1,
        "random_state": 179
    }
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=200
    )
    return model


def save_model(model, dv, model_path: str):
    print(f"Model stored in {model_path} file.")
    with open(model_path, 'wb') as file:
        pkl.dump((model, dv), file)


def test_model(dtest):
    y_pred = model.predict(dtest)
    return accuracy_score(y_pred, dtest.get_label())

if __name__ == '__main__':
    initial_dataset_path = 'data/initial_dataset.csv'
    model_path = "classifier_model.pkl"
    test_proportion = 0.2

    # Preprocess data::
    print(f"Starting the data preprocessing")
    df = preprocess_data(initial_dataset_path=initial_dataset_path)
    print(f"Data is preprocessed")

    # Split data:
    print(f"Splitting data to train and test. Test proportion: {test_proportion * 100}%")
    dtrain, dtest, dv = split_data(df, test_size=test_proportion)

    # Training model and saving it
    print(f"Starting model training")
    model = train_the_model(dtrain)
    save_model(model, dv, model_path)
    print(f"Model is trained and saved to {model_path}")

    # Save trained model and test it:
    accuracy = test_model(dtest)
    print(f"Model accuracy on test set: {accuracy*100}%")