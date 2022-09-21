import requests
import json
import time
import numpy as np
import pandas as pd
from os.path import exists

# This function gets the data from myanimelist api it sends a request every 5 seconds
# input:
# num_per_request - the number of data points per request to the api server
# min_num_data - the minimum number of data points you want
# return 
# dataset - an array of dict that's the dataset we get
# max_num_genres - the maximum number of genres for each data points
def get_data(num_per_request, min_num_data):
    headers = {"X-MAL-CLIENT-ID": "***REMOVED***"}
    main_url = "https://api.myanimelist.net/v2/anime/season/"
    season = ["winter", "spring", "summer", "fall"]

    dataset = []
    season_index = 2
    year = 2022
    num_data = 0
    parameters = {"offset": 0, "limit": num_per_request, "fields": "title,synopsis,genres,media_type"}
    run = 1
    max_num_genres = 0
    while num_data < min_num_data:
        # setting up the url for the get request
        url = main_url + str(year) + "/" + season[season_index]
        response = requests.get(url, params=parameters, headers=headers)

        # going over and organizing the data from the request
        num_usable_data = 0
        for data in response.json()["data"]:
            main_data = data["node"]
            if "genres" not in main_data or "synopsis" not in main_data or main_data["synopsis"] == "" or len(main_data["synopsis"].split()) == 0:
                continue
            temp = {}
            temp["name"] = main_data["title"]
            temp["synopsis"] = main_data["synopsis"]
            if "media_type" in main_data:
                temp["media_type"] = main_data["media_type"]
            else:
                temp["media_type"] = None
            temp["genre"] = []
            max_num_genres = max(max_num_genres, len(main_data["genres"]))
            for genre in main_data["genres"]:
                temp["genre"].append(genre["name"])
            dataset.append(temp)
            num_usable_data += 1

        print(url)
        print(response.status_code)
        print(num_usable_data)

        num_data += num_usable_data

        # checking if we have got all the anime from this season and move
        # to the previous one
        if parameters["limit"] > len(response.json()["data"]):
            parameters["offset"] = 0
            if season_index == 0:
                season_index = len(season) - 1
                year -= 1
            else:
                season_index -= 1
        else:
            parameters["offset"] += len(response.json()["data"])

        print("run", run)
        run += 1

        # pausing the loop to not make too many request to the server at once
        time.sleep(5)
    return dataset, max_num_genres

# This function orgranize the dataset and turn it into a panda dataframe
# input:
# dataset - an array of dict that's the dataset we get
# max_num_genres - the maximum number of genres for each data points
# return:
# df - a panda dataframe of the dataset
def organize_data(dataset, max_num_genres):
    index = 0
    panda_dict = {'name': [], 'synopsis': [], 'media_type': []}
    for i in range(max_num_genres):
        genre_index = "genre_" + str(i)
        panda_dict[genre_index] = []

    for data in dataset:
        panda_dict["name"].append(data["name"])
        panda_dict["synopsis"].append(data["synopsis"])
        panda_dict["media_type"].append(data["media_type"])
        for i in range(max_num_genres):
            genre_index = "genre_" + str(i)
            if i >= len(data["genre"]):
                panda_dict[genre_index].append(None)
            else:
                panda_dict[genre_index].append(data["genre"][i])

    df = pd.DataFrame(data=panda_dict)
    return df

# This function calls the get_data and organize_data function if the dataset does not exist
# and save the dataframe in a csv file
# input:
# num_per_request - the number of data points per request to the api server
# min_num_data - the minimum number of data points you want
# file_name - the name of the file you want to save in
# return
# if the file does not exist it returns the dataframe and 
# an array of dict that's the representation of the dataframe
def write_data(num_per_request, min_num_data):
    if exists('data/myanimelist.csv') == False:
        dataset, max_num_genres = get_data(num_per_request, min_num_data)
        df = organize_data(dataset, max_num_genres)
        df.to_csv('data/myanimelist.csv', index=False)
        return df, dataset
    return (None, None)

# This function that split the data set into train data, dev data and test data and
# save the data into different files
# input:
# dataset - a pd dataframe of our data
# num_test - the number of test data we want
# num_dev - the number of dev data we want
def split_data(dataset, num_test, num_dev):
    if exists('data/test_data.csv') == False and exists('data/dev_data.csv') == False and exists('data/train_data.csv') == False:
        test_data = dataset.sample(num_test)
        dataset = dataset.drop(test_data.index)
        dev_data = dataset.sample(num_dev)
        train_data = dataset.drop(dev_data.index)
        test_data.to_csv('data/test_data.csv', index=False)
        dev_data.to_csv('data/dev_data.csv', index=False)
        train_data.to_csv('data/train_data.csv', index=False)
