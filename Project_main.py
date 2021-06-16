"""
Name: Joey Roach
Date: June 10th 2020

A python module that conducts the analysis for the "Exploration and Analysis
of Trends Within the Movie Indsutry" CSE 163 project.
"""

import Project_functions
import pandas as pd


def main():
    movies = pd.read_csv('movies.csv', encoding='cp1252',
                         index_col='released', parse_dates=True)
    movies = Project_functions.clean_data(movies, initial=True)
    profit = Project_functions.calculate_profit(movies)
    profit_and_genre = Project_functions.profit_by_genre(profit)
    Project_functions.plot_genre_and_profits(profit_and_genre)
    with_trends = Project_functions.find_filmmaker_score(profit, 'director')
    with_trends = Project_functions.find_filmmaker_score(with_trends, 'writer')
    with_trends = Project_functions.find_filmmaker_score(with_trends, 'star')
    Project_functions.plot_filmmaker_trends(with_trends, 'director')
    Project_functions.plot_filmmaker_trends(with_trends, 'writer')
    Project_functions.plot_filmmaker_trends(with_trends, 'star')
    avg_profit = Project_functions.obtain_average_profit_by_rating(profit)
    Project_functions.plot_profit_by_rating(avg_profit)
    Project_functions.conduct_regression(movies)
    Project_functions.conduct_regression(movies, lasso=True)
    Project_functions.conduct_regression(movies, rf=True)


if __name__ == '__main__':
    main()
