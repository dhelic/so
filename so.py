import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statsmodels.formula.api as smf
from bs4 import BeautifulSoup
from pathlib import Path


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
    figures_dir = Path('figures/did/')
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figures_dir / filename)

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
