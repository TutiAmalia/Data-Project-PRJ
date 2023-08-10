# import module
import streamlit as st
import pandas as pd
import helper_functions as hf

st.set_page_config(
    page_title="Twitter Sentiment Analyzer", page_icon="📊", layout="wide"
)

adjust_top_pad = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
    """
st.markdown(adjust_top_pad, unsafe_allow_html=True)

# import data
data = pd.read_csv('data_clean_polarity full.csv')
st.session_state.df = data
# st.title("Analisis Sentimen Twitter")
# st.header("Pekan Raya Jakarta")

with st.sidebar:
    st.title("Twitter Sentiment Analyzer")

    st.markdown(
        """
        <div style="text-align: justify;">
            This app performs sentiment analysis on the latest tweets based on 
            the entered search term. Since the app can only predict positive or 
            negative sentiment, it is more suitable towards analyzing the 
            sentiment of brand, product, service, company, or person. 
            Only English tweets are supported.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # create a form to obtain the search parameters
    # with st.form(key="search_form"):
        # st.subheader("Search Parameters")
        # # session_state.search_term will be updated when the form is submitted
        # st.text_input("Search term", key="search_term")
        # # session_state.num_tweets will be updated when the form is submitted
        # st.slider("Number of tweets", min_value=100, max_value=2000, key="num_tweets")
        # # search_callback will be called when the form is submitted
        # st.form_submit_button(label="Search", on_click=search_callback)
        # st.markdown(
        #     "Note: it may take a while to load the results, especially with large number of tweets"
        # )

    # st.markdown("[Github link](https://github.com/tmtsmrsl/TwitterSentimentAnalyzer)")
    # st.markdown("Created by Timotius Marselo")

if "df" in st.session_state:
    def make_dashboard(data, bar_color, wc_color):

        col1, col2, col3, col4 = st.columns([25, 24, 24, 27])
        with col1:
            sentiment_plot = hf.plot_sentiment(data)
            sentiment_plot.update_layout(height=400, title_x=0.5)
            st.plotly_chart(sentiment_plot, theme=None, use_container_width=True)
        with col2:
            top_unigram = hf.get_top_n_gram(data, ngram_range=(1, 1), n=10)
            unigram_plot = hf.plot_n_gram(
                top_unigram, title="Top 10 Occuring Words", color=bar_color
            )
            unigram_plot.update_layout(height=350)
            st.plotly_chart(unigram_plot, theme=None, use_container_width=True)
        with col3:
            top_bigram = hf.get_top_n_gram(data, ngram_range=(2, 2), n=10)
            bigram_plot = hf.plot_n_gram(
                top_bigram, title="Top 10 Occuring Bigrams", color=bar_color
            )
            bigram_plot.update_layout(height=350)
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)
        with col4:
            top_trigram = hf.get_top_n_gram(data, ngram_range=(3, 3), n=10)
            bigram_plot = hf.plot_n_gram(
                top_trigram, title="Top 10 Occuring Trigrams", color=bar_color
            )
            bigram_plot.update_layout(height=350)
            st.plotly_chart(bigram_plot, theme=None, use_container_width=True)

        col1, col2 = st.columns([60, 40])
        with col1:

            def sentiment_color(sentiment):
                if sentiment == "positive":
                    return "background-color: #1F77B4; color: white"
                elif sentiment == "negative":
                    return "background-color: red; color: white"
                else:
                    return "background-color: #FF7F0E"

            st.dataframe(
                data[["polarity", "username", "text"]].style.applymap(
                    sentiment_color, subset=["polarity"]
                ),
                height=350,
            )
        with col2:
            wordcloud = hf.plot_wordcloud(data, colormap= wc_color)
            st.pyplot(wordcloud)

    adjust_tab_font = """
        <style> button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px;}</style>
    """

    st.write(adjust_tab_font, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["All", "Positive 😊", "Negative ☹️", "Neutral 😐"])
    with tab1:
        tweet_df = st.session_state.df
        make_dashboard(tweet_df, bar_color="#54A24B", wc_color="Greens")
    with tab2:
        tweet_df = st.session_state.df.query("polarity == 'positive'")
        make_dashboard(tweet_df, bar_color="#1F77B4", wc_color="Blues")
    with tab3:
        tweet_df = st.session_state.df.query("polarity == 'negative'")
        make_dashboard(tweet_df, bar_color="#D71313", wc_color="Reds")
    with tab4:
        tweet_df = st.session_state.df.query("polarity == 'neutral'")
        make_dashboard(tweet_df, bar_color="#FF7F0E", wc_color="Oranges")