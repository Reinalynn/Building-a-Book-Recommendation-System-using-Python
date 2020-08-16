# Building a Book Recommender System using Python

## Objective
Recommender systems have become a part of daily life for users of Amazon and Netflix and even social media. While some sites might use these systems to improve the customer experience (if you liked movie A, you might like movie B) or increase sales (customers who bought product C also bought product D), others are focused on customized advertising and suggestive marketing. As a book lover and former book store manager, I have always wondered where I can find good book recommendations that are both personalized to my interests and also capable of introducing me to new authors and genres. The purpose of this project is to create just such a recommender system (RS).

### Collaborative Filtering vs. Content Filtering
If an RS suggests items to a user based on past interactions between users and items, that system is known as a Collaborative Filtering system. In these recommendation engines, a user-item interactions matrix is created such that every user and item pair has a space in the matrix. That space is either filled with the user's rating of that item or it is left blank. This can be used for matrix factorization or nearest neighbor classification, both of which will be addressed when we develop our models. The important thing to remember with collaborative filtering is that user id, item id, and rating are the only fields required. Collaborative models can be user-based or item-based.

Content filtering, on the other hand, focuses exclusively on either the item or the user and does not need any information about interactions between the two. Instead, content filtering calculates the similarity between items or users using attributes of the items or users themselves. For my book data, I will use book reviews and text analysis to determine which books are most similar to books that I like and thus which books should be recommended (item based).

## Data
While there are many book datasets available to use, I decided to work with Goodreads Book data. There are several full Goodreads data sets available at the [UCSD Book Graph site](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) and I initially worked with this data to analyze metadata for books, authors, series, genres, reviews, and the interactions between users and items. Once I began building the models, I quickly realized that my dataset was too large. Rather than limit myself to just one genre, I chose to use the [Goodreads 10k data set](https://www.kaggle.com/zygmunt/goodbooks-10k/version/4), a subset of the full Goodreads data sets. This data set contains book metdata, ratings, book tags, and book shelves. 

For full code, view the following files in this github:

[EDA - full Goodreads authors, works, series, genres, interactions.ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/EDA%20-%20full%20Goodreads%20authors%2C%20works%2C%20series%2C%20genres%2C%20interactions.ipynb)   
[Data prep - full Goodreads loading files, statistics, distributions.ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/Data%20prep%20-%20full%20Goodreads%20loading%20files%2C%20statistics%2C%20distributions.ipynb)

### Collection, Cleaning and Analysis

##### Full Goodreads Dataset
```python
import gzip
def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield eval(l)
books = parse('Unused data/goodreads_books.json.gz')
next(books)
```
This Python generator allowed me to view a full book record in order to understand which fields are represented:
>{'isbn': '0312853122',
>'text_reviews_count': '1',
>'series': [],
>'country_code': 'US',
>'language_code': '',
>'popular_shelves': [{'count': '3', 'name': 'to-read'},
> {'count': '1', 'name': 'p'},
> {'count': '1', 'name': 'collection'},
> {'count': '1', 'name': 'w-c-fields'},
> {'count': '1', 'name': 'biography'}],
>'asin': '',
>'is_ebook': 'false',
>'average_rating': '4.00',
>'kindle_asin': '',
>'similar_books': [],
>'description': '',
>'format': 'Paperback',
>'link': 'https://www.goodreads.com/book/show/5333265-w-c-fields',
>'authors': [{'author_id': '604031', 'role': ''}],
>'publisher': "St. Martin's Press",
>'num_pages': '256',
>'publication_day': '1',
>'isbn13': '9780312853129',
>'publication_month': '9',
>'edition_information': '',
>'publication_year': '1984',
>'url': 'https://www.goodreads.com/book/show/5333265-w-c-fields',
>'image_url': 'https://images.gr-assets.com/books/1310220028m/5333265.jpg',
>'book_id': '5333265',
>'ratings_count': '3',
>'work_id': '5400751',
>'title': 'W.C. Fields: A Life on Film',
>'title_without_series': 'W.C. Fields: A Life on Film'}

The same can be done for any of the large json files available at the [UCSD Book Graph](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) site.

I conducted basic EDA on the full Goodreads data set by first looking at the size of each file. As is clear from these counts, the data sets are very, very large.

![Full Goodreads counts.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Full%20Goodreads%20counts.png) 

The interactions file is also quite large and contains entries for shelved books (a Goodreads user can classify a book by adding to a shelf that they create, such as a favorites list or a "to be read" list), read books, rated books, and reviewed books. 

![Goodreads interactions counts.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Goodreads%20interactions%20counts.png)

When visualizating the log-log plot of user/item distributions, both plots appear to follow Zipf's law. Zipf's law is typically used in text analysis and states that the frequency of any word is inversely proportional to its rank in the frequency table. In the case of the Goodreads data, it simply means that many of the book entries are for the same small number of books and from the same small number of users. More information on Zipf's Law can be found [here](https://en.wikipedia.org/wiki/Zipf%27s_law).

![Log-log plots of interactions](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Log-log%20plots%20of%20interactions.png)

The histogram below shows the distribution of the ratings in the interactions file. The scatterplot also indicates a clear relationship between the number of books read by a user and the number of books reviewed by the same user.

![Hist and scatterplot](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Hist%20and%20scatterplot.png)

I conducted similar analysis of the author file, recognizing that there is quite a bit of overlap between authors who receive high ratings on average and authors that have a large number of text reviews.

![Author plots](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Author%20plots.png)

The genres can be plotted in a pie chart where it becomes clear that fiction is the most prevelant genre. One thing to note is that books can be tagged with multiple genres.

![Pie chart of genres](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Pie%20chart%20of%20genres.png)

##### Goodreads 10k dataset
When I switched to the Goodreads 10k dataset for my model building, I conducted EDA using the pandas_profiling functions but the smaller dataset appeared to be representative of the full data.

```python
import surprise
import numpy as np
import pandas as pd
import pandas_profiling
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

books10k = pd.read_csv('Data/books10k.csv')
ratings10k = pd.read_csv('Data/ratings10k.csv')

pandas_profiling.ProfileReport(books10k)
pandas_profiling.ProfileReport(ratings10k)
```
### Feature Selection and Engineering
For collaborative filtering, the primary features necessary are user_id, book_id, and ratings. These were already present in the Goodreads 10k dataset and could be found in the ratings10k file.

For content filtering, it was important to include all of the variables that might be used to determine which items are similar to one another. To prepare for this, I had to create a new feature that contained relevant text for each book and then conduct text analysis on that feature. I was prepared to use the review text from each book for this, but I did try out a few different text features. From simplest to most complex, I used book tags only, keywords only, review text only, cleaned review text, and then a full text field that contained review text + title + authors + publication date. As expected, the best results were found with the full text field.

#### Text Analysis
In order to prep my text data for the content based RS, I followed these steps:
1. use generator to list reviews
2. merge reviews with books
3. books have multiple reviews - concat all review_text by book title and group together
4. clean text (this step is optional and I determined it was best to skip)
5. add back in book metadata because I had mistakenly dropped too many columns in step 2 (because of large data file)

For full code, view the following file in this github:

[Text analysis - build, clean, and prep review text.ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/Text%20analysis%20-%20build%2C%20clean%2C%20prep%20review%20text.ipynb)

I learned an important lesson when I cleaned and lemmatized the review text. Because many of the reviews contained proper names for book characters or book series, cleaning the text actually led to reduced performance and increased confusion. As a result, I chose not to clean the full text field so that my model could identify these important words and recognize that books with the same proper names are probably similar.

As an analysis of the full text field, I created the following word cloud:

![Word cloud of full text](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Word%20cloud%20of%20review%20text.png)

## Building and Tuning the Models

### Collaborative Filtering
In Collaborative Filtering, the model is often predicting the user's rating for a given book. Because of this, test and train sets can be created and root mean square error (RMSE) can be used to calculate the error rate of the model (difference between actual rating and predicted rating). The lower the RMSE, the lower the error and the more accurate the model.

#### PySpark
The PySpark package in Python uses the Alternating Least Squares (ALS) method to build recommendation engines. ALS is a matrix factorization running in a parallel fashion and is built for larger scale problems. PySpark was supports the collaboration of Apache Spark and Python.

I was able to build a Collaborative Filtering RS using PySpark that performed very well according the RMSE, but it was very slow. The original model had a RMSE of 0.396:

![ALS model with rmse.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/ALS%20model%20with%20rmse.png)

After tuning the model (which took a very, very long time), I was able to drop the RMSE to 0.362:

![Tuned ALS model with best rmse.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Tuned%20ALS%20model%20with%20best%20rmse.png)

For full code, view the following file in this github:

[Collab filtering using PySpark (user-based recommendations).ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/Collab%20filtering%20using%20PySpark%20(user-based%20recommendations).ipynb)

#### Pandas corrwith (pearsonR correlation)
Some Collaborative Filtering RS are built using a memory based method such as correlation. These models are very easy to build and interpret but the accuracy cannot be measured because the model is simply grouping like items together. The corrwith function in Pandas uses PearsonR's correlation method to output a nice list of recommendations when a book is input:

![PearsonR code](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/PearsonR%20code.png)

For full code, view the following file in this github:

[Collab filtering using pearsonR (item-based recommendations).ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/Collab%20filtering%20using%20pearsonR%20(item-based%20recommendations).ipynb)

#### Surprise
The Surprise package in Python is newer but provided all the tools I needed to test out multiple algorithms for Collaborative Filtering and then guided me through tuning the parameters and cross validating to determine the optimal model. In order to this, I first chose three versions of the data to analyze. I first looked at the most popular books only, filtering down to those with at least 20 book ratings and at least 50 user ratings. I then created a list of midlist books by filtering down to 2 book rating and 20 user ratings. Finally, I used the full list to include books that have as little as 1 book rating and 1 user rating.

![Algo results](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Algo%20results.png)

Using the top rated algorithms above, I chose to run a GridSearchCV on SVDpp, KNNBaseline, BaselineOnly, and KNNWithMeans. After completing the gridsearches, I ran 10-fold cross validation on each of the tuned models and plotted the results. I was surprised to find that the KNNBaseline ultimely performed best, especially considering that the SVDpp algorithm had been a front runner initially. The SVD algorithm created by Simon Funk is used in the Netflix RS.

![Optimized algo params.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Optimized%20algo%20params.png)
![Cross validation plot.png](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/Cross%20validation%20plot.png)

The KNNBaseline algorithm's lowest RMSE was about 0.85, much higher than the PySpark RMSE of 0.36, but the model was much faster and easier to use. One downfall, though, is that the model predicts books by user so it can only be used with current readers. This is known as the 'cold start problem' because the algorithm will not provide any output until a user has built up a profile. 

For full code, view the following file in this github:

[Collab filtering using Surprise (algo list, cv, plot, user-based recommendations).ipynb](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Code/Collab%20filtering%20using%20Surprise%20(algo%20list%2C%20cv%2C%20plot%2C%20user-based%20recommendations).ipynb)

### Content Filtering
Content filtering uses cosine similarity to map attributes (in my case, text) in order to determine which items are most similar to one another. Because of this, there is no way to measure the accuracy of the models and the results are more subjective. However, I do have strong domain knowledge in this area because I used to manage a bookstore and am still fairly well read, so I was able to identify a model that I thought was better than the others. Of course, this is personal preference

#### Tfidf and Count Vectorization
Here, features are extracted using term frequency - inverse document frequency (tfidf) or count vectorization. Using cosine similarity, both count and tfidf seem viable but tfidf might be more accurate since it is better at recommending the same authors and series.

The code used to extract features from text:
```python
pd.set_option('display.max_columns', 100)
ds = pd.read_csv('Data/reviews10k_grouped_full.csv')

# for tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_rev = tfidf_vectorizer.fit_transform((ds['full_text']))

# for count
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
count_rev = count_vectorizer.fit_transform((ds['full_text']))
```
Next, I chose a book ('The Name of the Wind' by Patrick Rothfuss, the same title chosen for the PearsonR Collaborative model) and then calculated the similarity between my chosen book and the rest of the books. From there the RS produced output for both tfidf and count vectorization models.

```python
#Choose book by inserting goodreads_book_id
g = 186074 
index = np.where(ds['goodreads_book_id'] == g)[0][0]
read_book = ds.iloc[[index]]

from sklearn.metrics.pairwise import cosine_similarity
book_tfidf = tfidf_vectorizer.transform(read_book['full_text'])
cos_similarity_tfidf = map(lambda x: cosine_similarity(book_tfidf, x), tfidf_rev)
output = list(cos_similarity_tfidf)

from sklearn.metrics.pairwise import cosine_similarity
book_count = count_vectorizer.transform(read_book['full_text'])
cos_similarity_countv = map(lambda x: cosine_similarity(book_count, x), count_rev)
output2 = list(cos_similarity_countv)

def get_recommendation(top, ds, scores):
  recommendation = pd.DataFrame(columns = ['goodreads_book_id', 'authors', 'title', 'score'])
  count = 0
  for i in top:
      recommendation.at[count, 'goodreads_book_id'] = ds.iloc[i, 2]
      recommendation.at[count, 'authors'] = ds.iloc[i, 19]
      recommendation.at[count, 'title'] = ds.iloc[i, 8]
      recommendation.at[count, 'score'] =  scores[count]
      count += 1
  return recommendation
```
For the tfidf model:
```python
# for tfidf
top = sorted(range(len(output)), key=lambda i: output[i], reverse=True)[:10]
list_scores = [output[i][0][0] for i in top]
get_recommendation(top, ds, list_scores)
```
![tfidf recs](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/tfidf%20recs.png)

For the count model:
```python
# for count
top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:10]
list_scores = [output2[i][0][0] for i in top]
get_recommendation(top, ds, list_scores)
```
![count recs](https://github.com/Reinalynn/Building-a-Book-Recommendation-System-using-Python/blob/master/Images/count%20recs.png)

## Conclusions
### Collaborative Filtering
I created 2 user-based Collaborative Filtering RS via PySpark and Surprise. PySpark, while slower, had a much better RMSE rate and would thus be my preferred model if I wanted to recommend books by user. PearsonR was the only item-based Collaborative Filtering RS that I built but I was satisfied by the results. If I were to continue with this project, I would further investigate item-based models and look for an alternative method of evaluation.

### Content Filtering
This portion of the project was most interesting to me because it allowed me to conduct text analysis. Looking at the results of my models, I believe that the tfidf model is most relevant but, as I indicated above, that is a matter of personal preference. It might be interesting to do additional research on the review text for each book, even conducting sentiment analysis to determine what percentage of reviews are positive vs. negative. My interaction with the data led me to believe that most people write positive reviews, but it could be helpful to identify the negative reviews and use those to negatively weight the books for recommendations. 

In all, I felt like this was a good choice for a practicum project. The problem was interesting to me and also advanced enough to challenge me to learn more about text analysis and machine learning algorithms used in recommendation engines. The data did involve some cleaning and prep, especially since I had to change datasets in Week 3, but it was not so time consuming that I did not get to spend adequate time on building and tuning the models. I was also able to improve my Python skills and learned how to use several packages that were new to me (Surprise, pandas_profiling, PySpark - I had limited experience).

## References:
* https://heartbeat.fritz.ai/recommender-systems-with-python-part-i-content-based-filtering-5df4940bd831
* https://github.com/ArmandDS/jobs_recommendations/blob/master/job_analysis_content_recommendation.ipynb
* https://github.com/MengtingWan/goodreads
* https://github.com/NicolasHug/Surprise/blob/master/examples/top_n_recommendations.py
* https://github.com/nikitaa30/Content-based-Recommender-System/blob/master/recommender_system.py
* https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/Recommender%20Systems%20-%20The%20Fundamentals.ipynb
* https://medium.com/@armandj.olivares/building-nlp-content-based-recommender-systems-b104a709c042
* https://medium.com/@chhavi.saluja1401/recommendation-systems-made-simple-b5a79cac8862
* https://stackabuse.com/creating-a-simple-recommender-system-in-python-using-pandas/
* https://stackoverflow.com/questions/39303912/tfidfvectorizer-in-scikit-learn-valueerror-np-nan-is-an-invalid-document
* https://towardsdatascience.com/collaborative-filtering-based-recommendation-systems-exemplified-ecbffe1c20b1
* https://towardsdatascience.com/how-did-we-build-book-recommender-systems-in-an-hour-the-fundamentals-dfee054f978e
* https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada
* https://towardsdatascience.com/my-journey-to-building-book-recommendation-system-5ec959c41847
* https://towardsdatascience.com/recommendation-systems-models-and-evaluation-84944a84fb8e
* https://towardsdatascience.com/various-implementations-of-collaborative-filtering-100385c6dfe0
* https://www.kaggle.com/robottums/hybrid-recommender-systems-with-surprise
* https://www.kaggle.com/vchulski/tutorial-collaborative-filtering-with-pyspark
* https://www.tutorialspoint.com/change-data-type-for-one-or-more-columns-in-pandas-dataframe-1
* Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", in RecSys'18.
* Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley, "Fine-Grained Spoiler Detection from Large-Scale Review Corpora", in ACL'19.
