import pandas as pd 
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_val_score


def optimize_df(df):
    """
    downcasts datatypes in a dataframe where possible
    """
    
    df_fl = df.select_dtypes(include=['float'])
    converted_fl = df_fl.apply(pd.to_numeric, downcast='float')
    df_int = df.select_dtypes(include=['int'])
    converted_int = df_int.apply(pd.to_numeric, downcast='unsigned')

    df[converted_int.columns] = converted_int
    df[converted_fl.columns] = converted_fl

    return df


def check_df_columns(df1, df2):
  """
  checks if two dataframe has the same columns
  """
  if set(df1.columns) == set(df2.columns):
    print("Same columns in both df")
  else:
    df1_extra = set(df1.columns) - set(df2.columns)
    print("df1 has these extra columns: ", df1_extra)
    df2_extra = set(df2.columns) - set(df1.columns)
    print("df2 has these extra columns: ", df2_extra)


def dropna_df(df_1, df_2):
    """
    preliminary cleaning and dropping rows/columns with a lot of missing values from dataframes
    Two dfs are passed in as arguments to ensure consistency in cleaning

    Args:
    df1: dataframe in which cleaning should be based on (general population dataset/ train dataset)
    df2: another dataframe which needs to be cleaned (customer dataset/ test dataset)

    Returns:
    df1_drop: cleaned version of df1
    df2_drop: cleaned version of df2
    """
    df1 = df_1.copy()
    df2 = df_2.copy()

    # cols with mixed datatypes
    cols_to_clean = df1.columns[18:20]

    # convert 'X', 'XX' to np.nan, string values to numerical
    print("cleaning columns: ", cols_to_clean)
    df1[cols_to_clean] = df1[cols_to_clean].replace({'X': np.nan, 'XX': np.nan}).astype(float)
    df2[cols_to_clean] = df2[cols_to_clean].replace({'X': np.nan, 'XX': np.nan}).astype(float)

    print("cleaning column: CAMEO_DEU_2015")
    df1['CAMEO_DEU_2015'] = df1['CAMEO_DEU_2015'].replace({'XX': np.nan})
    df2['CAMEO_DEU_2015'] = df2['CAMEO_DEU_2015'].replace({'XX': np.nan})

    # convert -1 to np.nan
    cols_with_negative_one = ['AGER_TYP', 'HEALTH_TYP', 'SHOPPER_TYP', 'VERS_TYP']
    print("cleaning columns: ", cols_with_negative_one)
    df1[cols_with_negative_one] = df1[cols_with_negative_one].replace({-1: np.nan})
    df2[cols_with_negative_one] = df2[cols_with_negative_one].replace({-1: np.nan})

    # drop columns where more than 30% of data is missing
    col_to_drop = df1.isna().sum()[df1.isna().sum()/len(df1) > 0.5].index.to_list()
    col_to_drop = col_to_drop + ['KBA13_HERST_SONST'] # has correlation value 1 with KBA13_FAB_SONSTIGE so drop

    print("dropping columns: ", col_to_drop)
    df1_drop = df1.drop(col_to_drop, axis='columns')
    # drop columns in df2 based on the columns we dropped in df1 (to ensure consistency)
    df2_drop = df2.drop(col_to_drop, axis='columns')

    # drop rows in customer where more than 50 columns of data is missing
    row_to_drop_1 = df1_drop.isna().sum(axis=1)[df1_drop.isna().sum(axis=1) > 50].index.to_list()
    df1_drop = df1_drop.drop(row_to_drop_1)
    #row_to_drop_2 = df2_drop.isna().sum(axis=1)[df2_drop.isna().sum(axis=1) > 50].index.to_list()
    #df2_drop = df2_drop.drop(row_to_drop_2)

    return df1_drop, df2_drop

def check_distributions(ori_df, clean_df, col_name):
    """
    plots the distribution of a column found in two dataframes
    to check if their distribution look similar

    Args:
    ori_df: first dataframe (original dataframe)
    clean_df: cleaned dataframe (after dropping rows/ columns)
    col_name: str, name of column that should be plotted
    """

    plt.rcParams["figure.figsize"] = [7,5]
    
    #plot distributions before
    ax1 = plt.subplot(1,2,1)
    sns.countplot(ori_df[col_name], ax=ax1)
    #plot distributions after
    ax2 = plt.subplot(1,2,2)
    sns.countplot(clean_df[col_name], ax=ax2)
    
    plt.show()


def feat_engr(df):
    """
    encodes categorical variables and does feature engineering
    """

    # dict to map praegende_jugendjahre to decade of movement and avantgarde/ mainstream
    # mainstream: 1, avantgarde: 0
    decade_dict = {0: np.nan, 1: 40, 2: 40, 3: 50, 4: 50, 5: 60,
               6: 60, 7: 60, 8: 70, 9: 70, 10: 80, 11: 80,
               12: 80, 13: 80, 14: 90, 15: 90}

    mvt_dict = {0: np.nan, 1: 0, 2: 1, 3: 0, 4: 1, 5: 0,
                 6: 1, 7: 1, 8: 0, 9: 1, 10: 0, 11: 1, 
                 12: 0, 13: 1, 14: 0, 15: 1}

    df['PRAEGENDE_JUGENDJAHRE_DECADES'] = df['PRAEGENDE_JUGENDJAHRE'].map(decade_dict)
    df['PRAEGENDE_JUGENDJAHRE_MVT'] = df['PRAEGENDE_JUGENDJAHRE'].map(mvt_dict)

    # separate wealth stage and family type
    df['CAMEO_INTL_2015_0'] = df['CAMEO_INTL_2015'].apply(lambda x: x//10 if float(x) else np.nan)
    df['CAMEO_INTL_2015_1'] = df['CAMEO_INTL_2015'].apply(lambda x: x%10 if float(x) else np.nan)

    # get year for eingefuegt_am
    df['EINGEFUEGT_AM_YEAR'] = pd.to_datetime(df['EINGEFUEGT_AM']).dt.year

    # replace ost_west_kz values with 0,1
    df['OST_WEST_KZ'] = df['OST_WEST_KZ'].replace({"W": 0, "O": 1})

    #lp_familie_grob have different categories for same values
    family_dict = {1: 1, 2: 2, 
                3: 3, 4: 3, 5: 3,
                6: 4, 7: 4, 8: 4,
                9: 5, 10: 5, 11:5}
    df['LP_FAMILIE_GROB'] = df['LP_FAMILIE_GROB'].map(family_dict)

    # get age group from lp_lebensphase_fein, income type can be found through lp_status_grob
    # then drop lp_lebensphase_fein and lp_lebensphase_grob
    # too many categoris to one hot encode, and similar values
    # younger: 0, middle: 1, advanced/higher: 2, retirement: 3
    # no age -- middle
    age_dict = {1: 0, 2: 1, 3: 0, 4: 2, 5: 2,
                6: 3, 7: 2, 8: 3, 9: 1, 10: 1,
                11: 2, 12: 3, 13: 2, 14: 0, 15: 2,
                16: 2, 17: 1, 18: 0, 19: 2, 20: 2,
                21: 1, 22: 1, 23: 1, 24: 1, 25: 1,
                26: 1, 27: 1, 28: 1, 29: 0, 30: 0,
                31: 2, 32: 2, 33: 0, 34: 0, 35: 0,
                36: 2, 37: 2, 38: 3, 39: 1, 40: 3}
    df['LP_LEBENSPHASE_FEIN_AGE'] = df['LP_LEBENSPHASE_FEIN'].map(family_dict)

    #lp_status_grob has diff categories for same values
    earnings_dict = {1: 0, 2: 0, 
                3: 1, 4: 1, 5: 1,
                6: 2, 7: 2, 
                8: 3, 9: 3, 
                10: 4}
    df['LP_STATUS_GROB'] = df['LP_STATUS_GROB'].map(earnings_dict)

    # # one hot encode categorical columns
    cat_cols = ['AGER_TYP', 'ANREDE_KZ', 'CAMEO_DEU_2015', 'CJT_GESAMTTYP', 'D19_KONSUMTYP',
                'GEBAEUDETYP_RASTER', 'GFK_URLAUBERTYP', 'NATIONALITAET_KZ', 'ZABEOTYP']

    df = pd.get_dummies(df, columns=cat_cols, prefix=cat_cols)

    # drop feature engineered columns and 'D19_LETZTER_KAUF_BRANCHE'
    feat_engr_cols_drop = ['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015', 'EINGEFUEGT_AM', 'LP_LEBENSPHASE_FEIN', 
                           'D19_LETZTER_KAUF_BRANCHE']

    df = df.drop(columns=feat_engr_cols_drop)

    return df


def pca_plots(pca):
    """
    plots the cumulative percentage variance explained against number of components
    """
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    plt.grid(b=True)
    plt.xlabel("number of components")
    plt.ylabel("percentage of variance explained")
    plt.show


def get_principal_component(pca, df, index):
    """
    get important features for each PCA component

    Args:
    pca: the PCA object used for fitting
    df: dataframe where the PCA was fit on
    index: PCA component that we are interested in

    """
    pca_df = pd.DataFrame(pca.components_, columns=df.columns)
    row = pca_df.iloc[index]
    print(row.sort_values(ascending=False).head())
    print(row.sort_values(ascending=False).tail())


def plot_kmeans(customer_pred, azdias_pred):
    """
    plots the percentage of general population and customer
    in each cluster side by side

    Args:
    customer_pred: list of clusters customers are grouped into
    azdias_pred: list of clusters the general population are grouped into
    """

    # get number of people for each cluster
    customer_count = Counter(customer_pred)
    # sort accoding to cluster number
    customer_count = dict(sorted(customer_count.items(),key = lambda i: i[0]))
    # get percentage of people in each cluster
    num_customer = len(customer_pred)
    customer_count = {k: v / num_customer for k, v in customer_count.items()}

    azdias_count = Counter(azdias_pred)
    azdias_count = dict(sorted(azdias_count.items(),key = lambda i: i[0]))
    num_pop = len(azdias_pred)
    azdias_count = {k: v / num_pop for k, v in azdias_count.items()}

    indices = list(azdias_count.keys())

    # Calculate optimal width
    width = np.min(np.diff(indices))/3.

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(indices-width/2, azdias_count.values(), width, color='b',label='general population')
    ax.bar(indices+width/2, customer_count.values(), width, color='r',label='customer')
    ax.set_xlabel('cluster')
    ax.set_ylabel('percentage of people in each cluster')
    plt.legend()
    plt.show()


def get_important_features(pca, kmeans_model, cluster_num, df):
    """
    plots a horizontal bar chart of the important featur for a specified KMeans cluster

    Args:
    pca: the PCA object used for fitting
    kmeans_model: The Kmeans model used for fitting
    cluster_num: int, the numbering of the cluster we are interested in
    df: dataframe before PCA is applied
    """

    # inverse pca for cluster we are interested in
    weight = pca.inverse_transform(kmeans_model.cluster_centers_[cluster_num])
    # convert to df to get name of important features
    weight_df = pd.DataFrame(weight, index=df.columns, columns=['feature_weights'])
    weight_df = weight_df.sort_values(by='feature_weights', ascending=False)

    top_weight_df = pd.concat([weight_df.iloc[:5], weight_df.iloc[-5:]])
    top_weight_df.plot.barh()


def get_classification(model, train_data, train_response, test_data, test_clean, csv_name):
    """
    trains and predicts individuals' responses to the campaign
    provides the cross validation test score of the model
    and saves prediction to a csv

    Args:
    model: model to be used
    train_data: data for training
    train_response: label for training data
    test_data: test data for which prediction is to be made on
    test_clean: test data with the 'LNR' column
    csv_name: string, name of csv file to save the predictions
    """

    # use cross validation
    cv_results = cross_val_score(model, train_data, train_response, scoring='roc_auc', cv=3)
    print("cv scores: ", cv_results)
    print("mean cv score: ", cv_results.mean())

    # get predictions
    model.fit(train_data, train_response)
    preds = model.predict_proba(test_data)[:, 1]
    # save predictions to csv
    sub = pd.DataFrame(preds, columns=['RESPONSE'], index=test_clean['LNR'].astype(int))
    sub.to_csv(csv_name)


def plot_importance_features(model, train_data, model_name):
    """
    Plots the 10 most important features of a model

    Args: 
    model: model used
    train_data: data used for training
    model_name: string, name of model used (for labelling)
    """

    df = pd.DataFrame({'feat': train_data.columns, 'importance': model.feature_importances_}).sort_values(by='importance', ascending=False).head(10)
    df.plot.barh(x='feat', y='importance')
    plt.gca().invert_yaxis()
    plt.title('Top important features in {} classifier'.format(model_name))