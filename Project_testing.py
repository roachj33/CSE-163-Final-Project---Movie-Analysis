"""
Name: Joey Roach
Section: AA
Tests the functions from the Project_functions file.
"""

import Project_functions
import pandas as pd
from cse163_utils import assert_equals


def obtain_single_series_val(data, conversion, value):
    """
    A helper function that condenses a pandas data frame down to a single value
    found within a specified "value" column, returning that value after it has
    been converted to the given type of "conversion".

    Arguments:
        data: The pandas series that contains the desired value to be tested.
        conversion: A specified type that the value should be converted to.
        value: A string representing the desired column to be accessed.

    Returns:
        A single value that is of type "conversion".
    """
    copied = data.copy()
    val = copied[value]
    if len(val) > 1:
        val = val[0]
    result = conversion(val)
    return result


def test_calculate_profit(full, subset):
    """
    Tests the calculate_profit method.
    """
    subset = Project_functions.calculate_profit(subset)
    mask_one = subset['director'] == 'James Cameron'
    masked_one = subset[mask_one]
    print(type(masked_one))
    result_one = obtain_single_series_val(masked_one, int, 'profit')
    mask_two = subset['director'] == 'Rob Reiner'
    masked_two = subset[mask_two]
    result_two = obtain_single_series_val(masked_two, int, 'profit')
    assert_equals(66660248, result_one)
    assert_equals(44287414, result_two)


def test_profit_by_genre(data):
    """
    Tests the profit_by_genre method.
    """
    copied = data.copy()
    copied = Project_functions.calculate_profit(copied)
    copied = Project_functions.profit_by_genre(copied)
    expected_one = 18920319.0
    expected_two = 19108147.0
    mask_one = copied['genre'] == 'Crime'
    mask_two = copied['year'] == 2016
    masked_one = copied[mask_one & mask_two]
    result_one = obtain_single_series_val(masked_one, float, 'profit')
    mask_three = copied['genre'] == 'Drama'
    mask_four = copied['year'] == 1990
    masked_two = copied[mask_three & mask_four]
    result_two = obtain_single_series_val(masked_two, float, 'profit')
    assert_equals(expected_one, result_one)
    assert_equals(expected_two, result_two)


def obtain_filtered_filmmaker_data(data, filmmaker, name):
    """
    A helper function that is designed to obtain a data frame that contains
    only films made by that filmmaker.

    Arguments:
        data: A pandas data frame of movie data.
        filmmaker: A string of the filmmaker type.
        name: A string of the particular filmmaker of the given filmmaker type
              whose filmmaker score we are interested in testing.

    Returns:
        A pandas data frame containing only films from the given desired
        filmmaker.
    """
    mask = data[filmmaker] == name
    result = data[mask]
    return result


def test_find_filmmaker_score(data):
    """
    Tests the find_filmmaker_score method.
    """
    copied = data.copy()
    director = Project_functions.find_filmmaker_score(copied, 'director')
    writer = Project_functions.find_filmmaker_score(copied, 'writer')
    star = Project_functions.find_filmmaker_score(copied, 'star')
    director = obtain_filtered_filmmaker_data(director, 'director',
                                              'David Cronenberg')
    writer = obtain_filtered_filmmaker_data(writer, 'writer', 'John Hughes')
    star = obtain_filtered_filmmaker_data(star, 'star', 'Matthew Broderick')
    expected_dir = 6.725
    expected_wri = 6.4142857143
    expected_star = 6.6333333333
    actual_director = \
        obtain_single_series_val(director, float, 'director_score')
    actual_writer = obtain_single_series_val(writer, float, 'writer_score')
    actual_star = obtain_single_series_val(star, float, 'star_score')
    assert_equals(expected_dir, actual_director)
    assert_equals(expected_wri, actual_writer)
    assert_equals(expected_star, actual_star)


def test_obtain_average_profit_by_rating(data):
    """
    Tests the obtain_gross_average_by_country method.
    """
    expected = 27985691.555555
    data = Project_functions.calculate_profit(data)
    average = Project_functions.obtain_average_profit_by_rating(data)
    pg = average['rating'] == 'PG'
    filtered = average[pg]
    result = obtain_single_series_val(filtered, float, 'average_profit')
    assert_equals(expected, result)


def main():
    full = pd.read_csv('movies.csv', encoding='cp1252', index_col='released',
                       parse_dates=True)
    subset = full[0:47]
    subset = Project_functions.clean_data(subset)
    full = Project_functions.clean_data(full)
    test_calculate_profit(full, subset)
    test_profit_by_genre(full)
    test_find_filmmaker_score(full)
    subset = Project_functions.calculate_profit(subset)
    test_obtain_average_profit_by_rating(subset)
    print('all tests passed!')


if __name__ == '__main__':
    main()
