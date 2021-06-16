"""
Name: Joey Roach
Date: June 10th 2020

Implements the functions required for project analysis, including cleaning the
data, calculating profits, analyzing profit trends over a variety of differing
circumstances and performing machine learning algorithms to predict a film's
IMDB score based on other factors.
"""

import pandas as pd
import plotly.express as px
import numpy as np
import os
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def clean_data(data, initial=True):
    """
    Given a pandas data frame of movie data, cleans the data for analytical
    purposes. If initial is true, this means that films in the dataset that
    do not have budget nor gross data are removed. If initial is false, only
    films that are rated G, PG, PG-13 or R are kept in the dataset.

    Arguments:
        data: A pandas dataframe of the movies.csv file.
        initial: A boolean to determine if this is the first time the data
                 is being cleaned, which operates in the manner described
                 above.

    Returns:
        A pandas data frame of movie data that has been cleaned as described
        above.
    """
    copied = data.copy()
    if initial is True:
        # Create masks for no budget and no revenue.
        no_budget = copied['budget'] == 0
        no_gross = copied['gross'] == 0
        # Obtain films with both budget and gross information.
        copied = copied[(~ no_budget) & (~ no_gross)]
    else:
        # Create masks for MPAA ratings of films.
        g = copied['rating'] == 'G'
        pg = copied['rating'] == 'PG'
        pg_13 = copied['rating'] == 'PG-13'
        r = copied['rating'] == 'R'
        # Obtain films that are rated G, PG, PG-13, or R.
        copied = copied[g | pg | pg_13 | r]
    return copied


def calculate_profit(data, average=False):
    """
    Given a pandas data frame containing movie data, calculates the profit
    (which is defined as revenue - budget) for each film within the dataset.

    Arguments:
        data: A pandas data frame containing movie data.

    Returns:
        A pandas data frame with all of the input data columns, including
        appended profit data for the films.
    """

    copied = data.copy()
    # Calculate profit for each film.
    copied['profit'] = copied['gross'] - copied['budget']
    return copied


def profit_by_genre(data):
    """
    Given a pandas data frame, groups films according to the year that they
    were released and the genre they are in, determining the mean profit for
    each genre, for each year (The musical genre is omitted as there are an
    insufficient number of musical films in the movies.csv file), returning
    a data frame containing the annual average profit for each genre.

    Arguments:
        data: A pandas data frame containing movie data from movies.csv.

    Returns:
        A pandas data frame of the form described above.
    """
    copied = data.copy()
    # Calculate average profit by genre by year, retaining data frame form.
    yearly = copied.groupby(['year', 'genre'], as_index=False)['profit'].mean()
    # Round profit to nearest whole number, for easier visualization purposes.
    yearly = yearly.round()
    # Exclude musical genre, as the dataset contains an insufficiently small
    # number of these types of films.
    musical = yearly['genre'] == 'Musical'
    yearly = yearly[~ musical]
    return yearly


def save_figure(figure, file_name):
    """
    Given a plotly figure object, and a file_name for the resulting graph,
    saves the graph to the "graphs" directory. If no such directory exists,
    it is created by this method.

    Arguments:
        figure: A plotly figure object that is the desired graph to be saved.
        file_name: A string representing the desired saved name of the graph.

    Returns:
        None.
    """
    # If the graphs directory does not exist yet, creates it.
    if not os.path.exists('graphs'):
        os.mkdir('graphs')
    # Saves figure with given file_name.
    figure.write_image('graphs/' + file_name)


def plot_genre_and_profits(data):
    """
    Given a pandas data frame of annual average profit data by genre (i.e., of
    the type returned by the profit_and_genre function), creates a plot of the
    data, saves the figure and directs the user to an online version of it.

    Arguments:
        data: A data frame of the type returned by profit_by_genre

    Returns:
        None.
    """
    # Plot profit of genres over time, faceting on the genre.
    fig = px.line(data, x='year', y='profit',
                  title='Profit of genres over time',
                  facet_col_wrap=3, facet_col='genre')
    # Save figure and display it.
    save_figure(fig, 'profit_of_genres.png')
    fig.show()


def find_filmmaker_score(data, filmmaker_type):
    """
    Given a pandas data frame, and the desired filmmaker type, calculates
    the average IMDB user score for each filmmaker of that type within
    the data.

    Arguments:
        data: A pandas data frame from the movies.csv file.
        filmmaker_type: A string argument representing the type of filmmaker to
                        calculate the average score for. Expected inputs are
                        "director", "writer", or "star".

    Returns:
        A pandas data frame with the average score for each filmmaker of the
        given filmmaker type appended to the end as a column.
    """
    copied = data.copy()
    # Calculate average IMDB score for each filmmaker of the given type,
    # and append it to the return data frame.
    copied[filmmaker_type + '_score'] = \
        copied.groupby(filmmaker_type)['score'].transform('mean')
    return copied


def profit_by_category(categories):
    """
    Given a dictionary called categories that maps from the IMDB score tier
    to the dataset containing film information for filmmakers within that tier,
    calculates the profit for each film in the tier and forms a column that
    ascribes each film to the tier, returning an updated version of the input
    dictionary with datasets that now reflect film profits and the tier they
    fall into.

    Arguments:
        categories: A dictionary that has the tier the film falls into (based
                    upon the calculated filmmaker score) as keys and the
                    filtered dataset according to filmmaker score as values.

    Returns:
        A dictionary with updated values of datasets that contain profit
        information and a column of strings that indicate which tier the
        films fall into.
    """
    for category in categories:
        # obtain dataset for the category.
        data_set = categories[category]
        # Calculate the profit for each film in the tier.
        data_set = calculate_profit(data_set)
        # Create a column in the dataset that details what tier the film is in.
        data_set['rating_category'] = category
        # Update value for the key in the dictionary to be the modified data
        # set.
        categories[category] = data_set
    return categories


def plot_filmmaker_trends(data, filmmaker_type):
    """
    Given a pandas data frame and the filmmaker_type, sorts filmmakers within
    the type according to their average IMDB scores into three tiers, and
    produces a box plot visualization of the profit by each tier.

    Arguments:
        data: A pandas data frame of movie data, containing a column
              representing each filmmaker's average IMDB score.
        filmmaker_type: A string argument representing the desired type of
                        filmmaker to display profit and tier information for.

    Returns:
        None.
    """
    # Obtain the average score for each filmmaker.
    score = data[filmmaker_type + '_score']
    # Create masks to filter filmmakers into three tiers (bottom third,
    # middle third and top third) based upon their average IMDB score.
    bottom_mask = (score >= 0.0) & (score < 3.3)
    middle_mask = (score >= 3.3) & (score < 6.6)
    top_mask = score >= 6.6
    # Obtain filtered data for each tier.
    bottom = data[bottom_mask]
    middle = data[middle_mask]
    top = data[top_mask]
    # Establish dictionary mapping from score tier to the associated
    # data frames.
    data_sets = {'bottom third': bottom, 'middle third': middle,
                 'top third': top}
    # Calculate profit for each tier.
    data_sets = profit_by_category(data_sets)
    # Bring all tier data frames into one data frame.
    categories = list(data_sets.values())
    total = pd.concat(categories)
    # Plot box plot of profit by IMDB score tiers.
    fig = px.box(total, x='rating_category', y='profit', points=False,
                 title='profit distribution by ' + filmmaker_type +
                 ' IMDB perception')
    # Save figure and display it.
    save_figure(fig, filmmaker_type + '_profit_distribution.png')
    fig.show()


def obtain_average_profit_by_rating(data):
    """
    Given a pandas data frame of film data, calculates the average profit for
    films according to their MPAA rating, returning a data frame with these
    calculations appended as a column.

    Arguments:
        data: A pandas data frame of film data that contains profit data for
              each film.

    Returns:
        A pandas data frame with the average profit for each rating appended
        as a column.
    """
    copied = data.copy()
    # Obtain column of average profit for each MPAA rating.
    copied['average_profit'] = \
        copied.groupby('rating')['profit'].transform('mean')
    return copied


def plot_profit_by_rating(data):
    """
    Given a pandas data frame containing film and average profit data, produces
    a plotly box plot of the average profit by films according to their MPAA
    ratings (including only G, PG, PG-13 and R rated films), saving this
    visualization and displaying it.

    Arguments:
        data: A pandas data frame which has film and average profit data.

    Returns:
        None.
    """
    # Obtain only films with MPAA ratings of G, PG, PG-13 or R.
    data = clean_data(data, initial=False)
    # Create box plot visualization, save it and display it.
    fig = px.box(data, x='rating', y='profit', points='all',
                 title='Film profits by MPAA ratings')
    save_figure(fig, 'film_profits_by_rating.png')
    fig.show()


def conduct_regression(data, lasso=False, rf=False):
    """
    Given a pandas data frame containing film data from the movies.csv file,
    performs machine learning algorithms in order to determine if we can
    accurately predict a film's IMDB score based upon some set of independent
    variables. By default, linear regression is performed, although the method
    can construct lasso and random forest regressions if the given parameter
    is specified to be true. Only one type of model may be built from the
    method per call.

    Arguments:
        data: A pandas data frame containing film data.
        lasso: A boolean parameter to determine if lasso regression is used.
               This occurs if lasso is True.
        rf: A boolean parameter to determine if random forest regression is
            used. This occurs if rf is True.

    Returns:
        None.
    """
    # In order to reproduce results, set random seed.
    np.random.seed(123)
    # Obtain features and label.
    desired = ['budget', 'company', 'country', 'director', 'genre', 'gross',
               'rating', 'runtime', 'star', 'writer', 'votes', 'year']
    features = data.filter(items=desired)
    label = data['score']
    # Convert binary variables.
    features = pd.get_dummies(features)
    # Obtain a 80% training, 20% testing split of the data.
    x_train, x_test, y_train, y_test = \
        train_test_split(features, label, test_size=0.20)
    # Standardize independent variables.
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    if lasso is True:
        # Fit cross-validated lasso model, with 5 k-folds, number of alphas
        # is reduced from the default to speed up computation time.
        model = linear_model.LassoCV(cv=5, n_alphas=20).fit(x_train, y_train)
    elif rf is True:
        # Fit random forest regressor model, with the maximum number
        # of features used for each tree as the square root of the
        # number of features (default behavior).
        model = RandomForestRegressor(n_estimators=100)
        model.fit(x_train, y_train)
    else:
        model = linear_model.LinearRegression().fit(x_train, y_train)
    if rf is False:
        # Print out model coefficients for parametric models.
        coefficients = pd.DataFrame(model.coef_, features.columns,
                                    columns=['Coefficient'])
        print(coefficients)
    # Obtain predictions for the IMDB score on test set.
    y_test_pred = model.predict(x_test)
    print(type(y_test_pred))
    # Round predictions to match format of actual IMDB scores.
    y_test_pred = np.round(y_test_pred, 1)
    # Acquire the mean squared error of the actual scores and the predicted
    # ones.
    test_error = mean_squared_error(y_test, y_test_pred)
    # Report on testing error rate, R2 value, and a portion of the actual
    # scores versus the predicted ones.
    results = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    print(results)
    print('testing error is:', test_error)
    print('R2 value:', r2_score(y_test, y_test_pred))
