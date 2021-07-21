"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud
import warnings
warnings.filterwarnings(action='once')

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
rating_m = pd.read_csv('resources/data/ratings.csv')
movies = pd.read_csv('resources/data/movies.csv')
imdb = pd.read_csv('resources/data/imdb_data.csv')


# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Introduction", "Exploratory Data Analysis" , "Recommender System", "Conclusion" , "About" ]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('First Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    # ---------------- ABOUT THE APP ---------------------------------
    if page_selection == "About":
        st.title("About The App")
        st.subheader("Data Source and Collection")
        st.write("The data which will be used in building our recommendation engine was obtained from MovieLens, \
            an online movie recommendation service. The dataset describes the 5-star rating (classification) and \
            free-text tagging activity from the website. It contains over 20 million (20,000,263) ratings and \
            465,564 tag applications across 27,278 movies. The dataset was developed by collecting movie preference\
            information from 138,493 users between January 09, 1995 and March 31, 2015 - a period of more than \
            twenty years. All the 138,493 users were randomly selected and each of them was asked to rate at least\
            20 movies. Each user was given a number (id) to distinguish the different users. The user ids were\
            anonymized and so no demographic or personal information was collected. The data comes in 6 different\
            files namely: genome-scores.csv, genome-tags.csv, links.csv, movies.csv, ratings.csv and tags.csv.\
            The major features of the data set include: userId, movieId, rating, tag, tagId and genres. The data\
            set is a 200MB zip file and is publicly available for download at http://grouplens.org/datasets/.")
        st.subheader("Team Members")
        st.write("Dineo Mahlangu")
        st.write("Ndamulelo Nelwamondo")
        st.write("Mummy Mashilo")
        st.write("Valecia Malan")
        
    # --------------- CONCLUSION -------------------------------------
    if page_selection == "Conclusion":
        st.title("Conclusion")
        st.write(" Recommender systems are a powerful new technology for extracting additional value for a \
        business from its user databases. These systems benefit users by enabling them to find items they like.\
        Conversely, they help the business by generating more sales.Recommender systems open new opportunities of \
        retrieving personalized information on the Internet. It also helps to alleviate the problem of information \
        overload which is a very common phenomenon with information retrieval systems. We come up with a strategy \
        that focuses on dealing with user’s personal interests and based on his previous reviews, movies are \
        recommended to users. This strategy helps in improving accuracy of the recommendations. A personal \
        profile is created for each user, where each user has access to his own history, his likes, ratings, \
        comments, password modification processes. It also helps in collecting authentic data with improved \
        accuracy and makes the system more responsive.")  
        


    # ----------------- INTRODUCTION -----------------------------------
    if page_selection == "Introduction":
        
        st.image('resources/imgs/giphy.gif',  width=600)
        st.title("Introduction")
        st.write("Simply put a Recommendation System is a filtration program whose prime goal is to predict the “rating” \
                or “preference” of a user towards a domain-specific item or item. In our case, this domain-specific item \
                is a movie, therefore the main focus of our recommendation system is to filterand predict only those \
                movies which a user would prefer given some data about the user him or herself.")
        st.subheader("Collaborative Recommender")
        st.write("Collaborative recommender provides general recommendations for each user based on the popularity of \
         the movie and (sometimes) the genre. The basic idea behind this recommender is that movies that are \
         more popular and well received will have a greater chance of being liked by the common public \
         .This model does not give personalized recommendations based on users.")
        st.subheader("Content Based Recommender")
        st.write("Content-based recommenders use user-provided data, whether explicit (rating) or implicit \
        (by clicking a link). Based on this data, a user profile is generated, which is then used to make\
         recommendations to the user.As users provide more input or take action on recommendations, \
         the engine becomes more and more accurate.")
        st.write("Press here to follow the instructions on how to use our application")


    # --------------------------------------------------EDA-----------------------

    if page_selection == "Exploratory Data Analysis":
        st.title("EDA")
        st.subheader("Insights of how the users have been rating the movies")  
        if st.checkbox('Show Distribution Of Movies Rating Graph'):
            def human(num, units = 'M'):
                units = units.lower()
                num = float(num)
                if units == 'k':
                    return str(num/10**3) + " K"
                elif units == 'm':
                    return str(num/10**6) + " M"
                elif units == 'b':
                    return str(num/10**9) +  " B"
        
            fig, ax = plt.subplots(figsize=(15,10))
            plt.title('Distribution of ratings over Training dataset', fontsize=15)
            sns.countplot(rating_m.rating.round())
            ax.set_yticklabels([human(item, 'M') for item in ax.get_yticks()])
            ax.set_ylabel('No. of Ratings(Millions)')
            st.pyplot(fig)
        
            st.write("This is a bar graph showing the rating of movie by people who have watched them.")
            st.write("The number of ratings is the total number of rating for each scale from 0.0 up \
            to 5.0 rated by people who watched the movies.")
        

        if st.checkbox('Show Pie chart for ratings'):
            # Calculate and categorise ratings proportions
            a = len(rating_m.loc[rating_m['rating']== 0.5]) / len(rating_m)
            b = len(rating_m.loc[rating_m['rating']==1.0]) / len(rating_m)
            c = len(rating_m.loc[rating_m['rating']==1.5]) / len(rating_m)
            d = len(rating_m.loc[rating_m['rating']==2.0]) / len(rating_m)
            low_ratings= a+b+c+d
            e = len(rating_m.loc[rating_m['rating']==2.5]) / len(rating_m)
            f = len(rating_m.loc[rating_m['rating']== 3.0]) / len(rating_m)
            g = len(rating_m.loc[rating_m['rating']==3.5]) / len(rating_m)
            medium_ratings= e+f+g
            h = len(rating_m.loc[rating_m['rating']==4.0]) / len(rating_m)
            i = len(rating_m.loc[rating_m['rating']==4.5]) / len(rating_m)
            j = len(rating_m.loc[rating_m['rating']==5.0]) / len(rating_m)
            high_ratings= h+i+j 
            # To view proportions of ratings categories, it is best practice to use pie charts
            # Where the slices will be ordered and plotted clockwise:
            labels = 'Low Ratings', 'Medium Ratings', 'High Ratings'
            sizes = [low_ratings, medium_ratings,  high_ratings]
            explode = (0, 0, 0.1)  # Only "explore" the 3rd slice (i.e. 'Anti')

            # Create pie chart with the above labels and calculated class proportions as inputs
            fig, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=270)#,textprops={'rotation': 65}
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Categorised Proportions of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot(fig)
            st.write("This is a pie chart showing the rating of movies by people who have watched them.")
            st.write("Low Ratings (scale: 0.0 - 2.0)")
            st.write("Medium Ratings (scale: 2.1 - 3.9)")
            st.write("High Ratings (scale: 4.0 - 5.0)")
   
        movies = pd.read_csv('resources/data/movies.csv')


# Organise a bit and store into feather-format
        movies.sort_values(by='movieId', inplace=True)
        movies.reset_index(inplace=True, drop=True)
        rating_m.sort_values(by='movieId', inplace=True)
        rating_m.reset_index(inplace=True, drop=True)

    

# Split title and release year in separate columns in movies dataframe. Convert year to timestamp.
        movies['year'] = movies.title.str.extract("\((\d{4})\)", expand=True)
        movies.year = pd.to_datetime(movies.year, format='%Y')
        movies.year = movies.year.dt.year # As there are some NaN years, resulting type will be float (decimals)
        movies.title = movies.title.str[:-7]

# Categorize movies genres properly. Working later with +20MM rows of strings proved very resource consuming
        genres_unique = pd.DataFrame(movies.genres.str.split('|').tolist()).stack().unique()
        genres_unique = pd.DataFrame(genres_unique, columns=['genre']) # Format into DataFrame to store later
        movies = movies.join(movies.genres.str.get_dummies().astype(bool))
        movies.drop('genres', inplace=True, axis=1)

# Modify rating timestamp format (from seconds to datetime year)
#ratings.timestamp = pd.to_datetime(ratings.timestamp, unit='s')
        rating_m.timestamp = pd.to_datetime(rating_m.timestamp, infer_datetime_format=True)
        rating_m.timestamp = rating_m.timestamp.dt.year
        movies.dropna(inplace=True)
        rating_m.dropna(inplace=True)
    
# Organise a bit, then save into feather-formatand clear from memory
        movies.sort_values(by='movieId', inplace=True)
        rating_m.sort_values(by='movieId', inplace=True)
        movies.reset_index(inplace=True, drop=True)
        rating_m.reset_index(inplace=True, drop=True)
        
        # Let's work with a temp smaller slice 'dftmp' of the original dataframe to reduce runtime (ratings hass +2MM rows)
        dftmp = movies[['movieId', 'year']].groupby('year')

        if st.checkbox('Show Movies Production Over The Years Graph'):
            fig, ax1 = plt.subplots(figsize=(15,10))
            ax1.plot(dftmp.year.first(), dftmp.movieId.nunique(), "g-o")
            ax1.grid(None)
            ax1.set_ylim(0,)

            dftmp = rating_m[['rating', 'timestamp']].groupby('timestamp')
            ax2 = ax1.twinx()
            ax2.plot(dftmp.timestamp.first(), dftmp.rating.count(), "r-o")
            ax2.grid(None)
            ax2.set_ylim(0,)

            ax1.set_xlabel('Year')
            ax1.set_ylabel('Number of movies released'); ax2.set_ylabel('Number of ratings')
            plt.title('Movies per year')
            st.pyplot(fig)
        


        if st.checkbox('Show Movies Ratings Over The Years Graph'):
            dftmp = movies[['movieId', 'year']].set_index('movieId').join(
            rating_m[['movieId','rating']].groupby('movieId').mean())

            plt.figure(figsize=(15,8))
            plt.plot(dftmp.year, dftmp.rating,"g.", markersize=4)
            plt.xlabel('Year')
            plt.ylabel('Movie average rating')
            plt.title('All movies rating')
            plt.ylim(0,)
            st.pyplot()
        

        st.subheader("Insights of IMDB data set") 
        if st.checkbox('Show Popular Cast Graph'): 
            # Create dataframe containing only the movieId and title_casts
            imdb_casts = pd.DataFrame(imdb[['movieId', 'title_cast']],
                             columns=['movieId', 'title_cast'])

            # Split title_casts seperated by "|" and create a list containing the title_cast allocated to each movie
            imdb_casts.title_cast = imdb_casts.title_cast.apply(lambda x: x.split('|'))

            # Create expanded dataframe where each movie-title_cast combination is in a seperate row
            imdb_casts = pd.DataFrame([(tup.movieId, d) for tup in imdb_casts.itertuples() for d in tup.title_cast],
                             columns=['movieId', 'title_cast'])
   
        # Plot popular cast
            plt.figure(figsize = (20,5))
            casts=imdb_casts['title_cast'].explode()
            ax=sns.countplot(x=casts, order = casts.value_counts().index[:20],color='purple')
            ax.set_title('Popular casts',fontsize=15)
            plt.xticks(rotation=90)
            st.pyplot() 
        
        if st.checkbox('Show Popular Keyword Graph'):
            # Create dataframe containing only the movieId and keywords
            imdb_keywords = pd.DataFrame(imdb[['movieId', 'plot_keywords']],
                             columns=['movieId', 'plot_keywords'])

            # Split genres seperated by "|" and create a list containing the genres allocated to each movie
            imdb_keywords.plot_keywords = imdb_keywords.plot_keywords.apply(lambda x: x.split('|'))

           # Create expanded dataframe where each movie-genre combination is in a seperate row
            imdb_keywords = pd.DataFrame([(tup.movieId, d) for tup in imdb_keywords.itertuples() for d in tup.plot_keywords],
                             columns=['movieId', 'plot_keywords'])

            # Plot popular keywords
            plt.figure(figsize = (20,5))
            keyword=imdb_keywords['plot_keywords'].explode()
            ax=sns.countplot(x=keyword, order = keyword.value_counts().index[:20],color='purple')
            ax.set_title('Popular keywords',fontsize=15)
            plt.xticks(rotation=90)
            st.pyplot()
        if st.checkbox('Show Popular Directors Graph'):
            # Plot popular directors
            plt.figure(figsize = (20,5))
            directors=imdb['director'].explode()
            ax=sns.countplot(x=directors, order = directors.value_counts().index[:20],color='purple')
            ax.set_title('Popular directors',fontsize=15)
            plt.xticks(rotation=90)
            st.plotly_chart()
            
       
        
        if st.checkbox('Show WordClouts of keywords'):
            imdb_keywords = imdb_keywords['plot_keywords'].copy()

            # Join all the text in the list and remove apostrophes
            all_gtags = ' '.join([text for text in imdb_keywords.astype(str)])
            all_gtags = all_gtags.replace("'", "")

            wordcloud = WordCloud(width=2000,height=1000, random_state=21, max_font_size=200, background_color=
                      'white', min_word_length=3, max_words=20).generate(all_gtags)
            plt.figure(facecolor = 'white', edgecolor='blue', dpi=600)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot
    
        if st.checkbox('Show WordClouts of Popular Cast'):    
            # Copy imdb_casts title_cast column
            imdb_casts = imdb_casts['title_cast'].copy()

            # Join all the text in the list and remove apostrophes
            imdb_casts = ' '.join([text for text in imdb_casts.astype(str)])
            imdb_casts = imdb_casts.replace("'", "")

            wordcloud = WordCloud(width=2000,height=1000, random_state=21, max_font_size=200, background_color=
                      'white', min_word_length=1, max_words=50).generate(imdb_casts)
            plt.figure(facecolor = 'white', edgecolor='blue', dpi=600)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot()


if __name__ == '__main__':
    main()
