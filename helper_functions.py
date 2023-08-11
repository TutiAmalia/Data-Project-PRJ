import numpy as np
import pandas as pd

# for visualization
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

import re
import datetime as dt
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.io as pio
import matplotlib as mpl
import matplotlib.pyplot as plt
# from wordcloud import WordCloud
from PIL import Image
import sklearn
# print(sklearn.__version__)

# Define custom Indonesian stopwords
indonesian_stopwords = set([
    'di', 'ya', 'ini', 'dan', 'ga', 'ke', 'aja', 'bgt', 'yang', 'yg',
    'tapi', 'aku', 'gua', 'gue', 'kalo', 'nya', 'itu', 'dah', 'sih',
    'tp', 'ada', 'bisa', 'mau', 'iya', 'wkwk'
])

def plot_sentiment(tweet_df):
    # count the number tweets based on the sentiment
    sentiment_count = tweet_df["polarity"].value_counts()

    # plot the sentiment distribution in a pie chart
    fig = px.pie(
        values=sentiment_count.values,
        names=sentiment_count.index,
        hole=0.3,
        title="<b>Sentiment Distribution</b>",
        color=sentiment_count.index,
        # set the color of positive to blue and negative to orange
        color_discrete_map={"positive": "#1F77B4", "negative": "#D71313", "neutral": "#F94C10"},
    )
    fig.update_traces(
        textposition="inside",
        texttemplate="%{label}<br>%{value} (%{percent})",
        hovertemplate="<b>%{label}</b><br>Percentage=%{percent}<br>Count=%{value}",
    )
    fig.update_layout(showlegend=False)
    return fig

def plot_wordcloud(tweet_df, colormap="Greens"):
    stopwords = set(STOPWORDS)
    stopwords.update(indonesian_stopwords)
    cmap = plt.cm.get_cmap(colormap)(np.linspace(0, 1, 20))
    cmap = plt.colors.ListedColormap(cmap[10:15])
    mask = np.array(Image.open("twitter_mask.png"))
    font = "quartzo.ttf"
    text = " ".join(tweet_df["text_preprocessed"])
    wc = WordCloud(
        background_color="white",
        stopwords=stopwords,
        font_path=font,
        max_words=90,
        colormap=cmap,
        mask=mask,
        random_state=42,
        collocations=False,
        min_word_length=2,
        max_font_size=200,
    )
    wc.generate(text)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud", fontdict={"fontsize": 16}, fontweight="heavy", pad=20, y=1.0)
    return fig

def get_top_n_gram(tweet_df, ngram_range, n=10):
    stopwords = set(STOPWORDS)
    stopwords.update(indonesian_stopwords)
    # Split each preprocessed tweet into words
    words_list = [tweet.split() for tweet in tweet_df["text_preprocessed"]]
    # Flatten the list of lists into a single list of words
    all_words = [word for words in words_list for word in words]
    # Create a Pandas Series and count the word occurrences
    word_counts = pd.Series(all_words).value_counts().reset_index()
    word_counts.columns = ["words", "counts"]
    # Filter out stopwords
    word_counts = word_counts[~word_counts["words"].isin(stopwords)]
    # Sort by counts and select the top n
    top_n_gram = word_counts.head(n)
    top_n_gram["words"] = top_n_gram["words"].str.title()
    return top_n_gram

def plot_n_gram(n_gram_df, title, color="#54A24B"):
    fig = px.bar(
        x=n_gram_df.counts,
        y=n_gram_df.words,
        title="<b>{}</b>".format(title),
        text_auto=True,
    )
    fig.update_layout(plot_bgcolor="white")
    fig.update_xaxes(title=None)
    fig.update_yaxes(autorange="reversed", title=None)
    fig.update_traces(hovertemplate="<b>%{y}</b><br>Count=%{x}", marker_color=color)
    return fig