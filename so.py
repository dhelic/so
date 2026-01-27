import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import patsy
from bs4 import BeautifulSoup
from pathlib import Path
from pandas.tseries.offsets import DateOffset


LAUNCH = '2022-11-30'


def load_data(filename='data/csv/final/posts_selected.csv'):
    """
    Loads the data from the specified csv file and converts the CreationDate column to datetime.
    """
    
    df = pd.read_csv(filename)
    df['CreationDate'] = pd.to_datetime(df['CreationDate'])
    return df


def select_data(df, control_start='2021-05-31', control_end=None, treated_start=None, treated_end='2023-05-29', 
                control_start_1=None, control_end_1=None, treated_start_1=None, treated_end_1=None):
    """
    Selects the data for the control and treated groups based on the specified dates. The default control group starts year and a half before the ChatGPT launch 
    and contains one year of data. The default treated group starts six months before the ChatGPT launch and ends six months after the launch also
    containing one year of data. The end dates have to be +1 date because the date is interpreted as the begining of the day at 00:00:00.
    """

    # filter the data
    if control_end is None:
        df_filtered = df[(df['CreationDate'] >= pd.to_datetime(control_start)) & (df['CreationDate'] <= pd.to_datetime(treated_end))]
    elif control_start_1 is None:
        df_filtered = df[((df['CreationDate'] >= pd.to_datetime(control_start)) & (df['CreationDate'] <= pd.to_datetime(control_end))) |
                         ((df['CreationDate'] >= pd.to_datetime(treated_start)) & (df['CreationDate'] <= pd.to_datetime(treated_end)))]
    else:
        df_filtered = df[((df['CreationDate'] >= pd.to_datetime(control_start)) & (df['CreationDate'] <= pd.to_datetime(control_end))) |
                         ((df['CreationDate'] >= pd.to_datetime(treated_start)) & (df['CreationDate'] <= pd.to_datetime(treated_end))) |
                         ((df['CreationDate'] >= pd.to_datetime(control_start_1)) & (df['CreationDate'] <= pd.to_datetime(control_end_1))) |
                         ((df['CreationDate'] >= pd.to_datetime(treated_start_1)) & (df['CreationDate'] <= pd.to_datetime(treated_end_1)))]
    return df_filtered


def prepare_data(df):
    """
    Adds new columns to the dataframe including the number of lines in the Body column and an indicator for treated and control groups,
    an indicator for periods before and after the ChatGPT launch, and a week counter. The end dates have to be +1 date because the date
    is interpreted as the begining of the day at 00:00:00.
    """
    
    # set the date of the ChatGPT launch
    launch = pd.to_datetime(LAUNCH)

    # add line count column
    df['line_count'] = df['Body'].str.count('\n') + 1

    # add treated and control group indicators
    df['T'] = ((df['CreationDate'] >= pd.to_datetime('2022-05-30')) & (df['CreationDate'] <= pd.to_datetime('2023-05-29'))).astype(int)
    
    df['P'] = (((df['CreationDate'] >= launch) & (df['CreationDate'] <= pd.to_datetime('2023-05-29'))) |
               ((df['CreationDate'] >= pd.to_datetime('2021-11-30')) & (df['CreationDate'] <= pd.to_datetime('2022-05-30')))).astype(int)
    
    # add week counter
    df['W'] = df['CreationDate'].dt.isocalendar().week


def extract_texts(body):
    """
    Extracts the code and other text from the HTML content of the Body column.
    It uses BeautifulSoup to parse the HTML and extract the text from <pre><code> blocks and other text.
    """

    try:
        # parse the HTML content
        soup = BeautifulSoup(body, 'html.parser')
    except:
        # if the parsing fails, return empty strings
        return pd.Series({'code': 'Could not parse the html', 'text': 'Could not parse the html'})

    # extract all <code> blocks
    code = [tag.get_text() for tag in soup.select('code')]

    # remove <code> elements
    for tag in soup.select('code'):
        tag.decompose()

    # get remaining text
    text = soup.get_text(separator='\n', strip=True)

    return pd.Series({'code': '\n'.join(code), 'text': text})


def questions(df):
    """
    Filters the data to only include questions.
    """
    
    questions = df[df['PostTypeId'] == 1]
    return questions


def answers(df):
    """
    Filters the data to only include answers.
    """
    
    answers = df[df['PostTypeId'] == 2]
    return answers


def extract_tags(df):
    tag_df = (df.assign(parsed_tags=df['Tags'].str.strip('|').str.split('|')).explode('parsed_tags')[['CreationDate', 'T', 'parsed_tags']].reset_index(drop=True))
    return tag_df


def ts(df):
    """
    Extracts and resample the time series of the treated and control groups per week.
    """

    df_T = df[df['T'] == 1]
    df_C = df[df['T'] == 0]
    df_T.index = df_T['CreationDate']
    df_C.index = df_C['CreationDate']
    df_T_resampled = df_T.resample('W')
    df_C_resampled = df_C.resample('W')

    return df_T_resampled, df_C_resampled


def plot_ts(treated, control, y, title, filename, ylabel='Count'):
    """
    Plots the time series of the treated and control groups with the specified y variable. The time series are aligned for a better comparison.
    The plot is saved in the figures folder.
    """

    # set the date of the ChatGPT launch
    launch = pd.to_datetime(LAUNCH)

    # date offset for the control group
    control['CreationDate_offset'] = control.index + pd.DateOffset(days=364)    

    plt.figure()

    # plot treated and control time series
    sns.lineplot(data=treated, x='CreationDate', y=y, marker='o', label='22/23')
    sns.lineplot(data=control, x='CreationDate_offset', y=y, marker='o', label='21/22')

    # add vertical line for ChatGPT launch
    plt.axvline(x=launch, color='black', linestyle='--', linewidth=1, label='ChatGPT Launch')

    # set x-axis to show months
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    plt.gca().tick_params(axis='x', rotation=45)

    # set grid and labels
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.xlabel('Month')
    plt.ylabel(ylabel)

    # set title and legend
    plt.title(title)
    plt.legend()

    # set tight layout
    plt.tight_layout()

    # save the plot
    filepath = Path('figures/weekly/' + filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)

    plt.show()
    plt.close()


def did_prepare_data(df, type=None):
    df['log_lc'] = np.log(df['line_count'])
    df['D'] = df['CreationDate'].dt.dayofweek

    if type == 'tags':
        df['loc'] = df['code'].str.count('\n') + 1
        df['log_loc'] = np.log(df['loc'])


def did(outcome, data):
    """
    Estimates the difference-in-differences model for the specified outcome variable.
    """
    
    # copy the df here and then add the columns to the copy
    # this surpresses the SettingWithCopyWarning
    df = data[[outcome, 'T', 'P', 'W']].copy()

    # standardize the outcome variable
    df[f'{outcome}_std'] = (df[outcome] - df[outcome].mean()) / df[outcome].std()
    
    lr = smf.ols(formula = f'{outcome}_std ~ T * P + W', data=df)
    results = lr.fit()
    return results


def did_design_matrix(outcome, data):
    """
    Creates the design matrix for the difference-in-differences model for the specified outcome variable.
    """

    formula = f'{outcome} ~ T * P + W'
    y_mat, X = patsy.dmatrices(formula, data, return_type='dataframe')

    # extract outcome as Series
    y_raw = y_mat.iloc[:, 0]

    return X, y_raw


def did_ols(X, y, masks, start_dates):
    """
    Fits the difference-in-differences model using the design matrix and outcome variable.
    """

    res = []    
    for i, m in enumerate(masks):
        if i%50 == 0:
            print(f'Step {i}')

        Xi = X.loc[m]
        yi = y.loc[m]

        # standardize outcome
        y_std = (yi - yi.mean()) / yi.std()

        # cluster by calendar date (recommended)
        #g = questions.loc[m, 'date']

        reg_results = sm.OLS(y_std, Xi).fit()
        #results = sm.OLS(y_std, Xi).fit(cov_type='cluster', cov_kwds={'groups': g})

        res.append({'date': start_dates[i], 'coef': reg_results.params['T:P'],
                'ci_l': reg_results.conf_int().loc['T:P'][0],
                'ci_u': reg_results.conf_int().loc['T:P'][1]})
        
        #res.append({'date': start_dates[i], 'coef': results.params['P'],
        #           'ci_l': results.conf_int().loc['P'][0],
        #           'ci_u': results.conf_int().loc['P'][1]})

    res_df = pd.DataFrame(res)
    return res_df


def rolling_window_masks(data, pre_start='2022-05-30', pre_end='2022-11-30', pre_start_c='2021-05-31', pre_end_c='2021-11-30', end='2023-04-29', window_days=31):
    """
    Creates rolling window masks for the specified start and end dates, window size, and step size.
    """

    pre_start = pd.Timestamp(pre_start)
    pre_end = pd.Timestamp(pre_end)
    pre_start_c = pd.Timestamp(pre_start_c)
    pre_end_c = pd.Timestamp(pre_end_c)

    start_dates = pd.date_range(start=pre_end, end=pd.Timestamp(end), freq='D')

    masks = []
    for s in start_dates:
        e = s + pd.Timedelta(days=window_days)

        # treated post
        post_t = (data['CreationDate'] >= s) & (data['CreationDate'] <= e) & (data['T'] == 1)

        # control post (one year earlier)
        post_c = ((data['CreationDate'] >= s - DateOffset(days=365)) & (data['CreationDate'] <= e - DateOffset(days=365)) &(data['T'] == 0))

        # treated pre
        pre_t = ((data['CreationDate'] >= pre_start) & (data['CreationDate'] <= pre_end) & (data['T'] == 1))

        # control pre
        pre_c = ((data['CreationDate'] >= pre_start_c) & (data['CreationDate'] <= pre_end_c) & (data['T'] == 0))

        mask = post_t | post_c | pre_t | pre_c
        masks.append(mask)

    return start_dates, masks


def plot_did(df, title, filename):
    """
    Plots the estimated coefficients of the difference-in-differences model with confidence intervals.
    The plot is saved in the figures folder.
    """

    # wide format
    plt.figure(figsize=(16, 4))

    # plot the time series with confidence intervals
    sns.lineplot(x='date', y='coef', data=df, marker='o')
    plt.fill_between(df['date'], df['ci_l'], df['ci_u'], alpha=0.15)

    # add horizontal line at 0
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)

    # set x-axis to show months
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))
    plt.gca().tick_params(axis='x', rotation=45)

    # set grid and labels
    plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
    plt.xlabel('Month')
    plt.ylabel('$\\beta_3$')

    # set title
    plt.title(title)
    #plt.legend()
    
    # set tight layout
    plt.tight_layout()

    # save the plot
    filepath = Path('figures/did/' + filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath)

    plt.show()
    plt.close()

def parallel_trend(outcome, data):
    """
    Tests for the parallel trend between the treated and the control group prior to the treatment for the specified outcome variable.
    """
    
    # copy the df here and then add the columns to the copy
    # this surpresses the SettingWithCopyWarning
    df = data[[outcome, 'T', 'W']].copy()

    # standardize the outcome variable
    df[f'{outcome}_std'] = (df[outcome] - df[outcome].mean()) / df[outcome].std()
    
    lr = smf.ols(formula = f'{outcome}_std ~ T * W', data=df)
    results = lr.fit()
    return results
