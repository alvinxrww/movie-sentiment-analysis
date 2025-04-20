import pandas as pd
import altair as alt
import nltk
import requests
import streamlit as st
from nltk import pos_tag
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from transformers import pipeline
from gensim import corpora, models
from nltk.tokenize import word_tokenize
from requests.exceptions import MissingSchema

PARSER = "html.parser"
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def search_movie():
    # Create a text input for the search bar
    movie_searched = st.text_input('Search movie title:', '')

    # Api for querying the searched movie
    # Source: rotten tomatoes
    search_url = f"https://www.rottentomatoes.com/search?search={movie_searched}"
    response = requests.get(search_url)
    return response


def show_suggestions(response):
    soup = BeautifulSoup(response.content, PARSER)
    search_result_elements = soup.find(
        'search-page-result', attrs={'type': 'movie', 'data-qa': 'search-result'})
    property_elements = search_result_elements.find_all(
        'a', class_='unset', attrs={'data-qa': 'info-name', 'slot': 'title'})
    titles_urls = [(e.get('href'), e.text.strip()) for e in property_elements]
    return titles_urls


def pick_suggestion(suggestions):
    selected = 'you havent selected any'
    for suggestion in suggestions:
        # Check if the button is pressed
        if st.button(suggestion[1], key=suggestion[0]):
            selected = suggestion
    return selected


def get_reviews(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, PARSER)

    # Find all div review elements
    property_elements = soup.find_all(
        'p', class_="audience-reviews__review js-review-text")

    # Extract text content without <p> tags
    reviews_text = [element.text.strip().lower()
                    for element in property_elements]

    return reviews_text


def get_critics(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all div elements with class "ListingCell-KeyInfo-price"
    property_elements = soup.find_all(
        'p', class_='review-text', attrs={'data-qa': 'review-quote'})

    # Extract text content without <p> tags
    reviews_text = [element.text.strip().lower()
                    for element in property_elements]

    return reviews_text


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalpha()
              and token not in stop_words]
    return tokens


def extract_adj(text):
    # Tokenize and perform part-of-speech tagging
    tokens = preprocess_text(text)
    pos_tags = pos_tag(tokens)

    # Extract adjectives
    adjectives = [
        word for word, pos in pos_tags
        if pos.startswith('JJ')
        or pos.startswith('JJR')
        or pos.startswith('JJS')
    ]

    # Join adjectives into a single string
    all_adjectives = ' '.join(adjectives)

    return all_adjectives


def generate_wordcloud(words):
    # Generate WordCloud for adjectives
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(words)

    # Display the WordCloud using matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.subheader('Wordcloud')
    st.pyplot(fig)
    st.write('The figure above is a visual representation of the review data displayed in different sizes based on their frequency. The more frequent a word appears in the review, the larger and bolder it appears in the word cloud. This can be used to identify the most prominent and frequently mentioned terms. This wordcloud only displays adjectives as it allows the reader to focus on the descriptive elements and sentiments associated with the films.')


def analyze_sentiment(text_to_analyze):
    # Specify the model and revision explicitly
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    revision = "af0f99b"

    # Load the sentiment analysis pipeline with explicit model and revision
    sentiment_analyzer = pipeline(
        "sentiment-analysis", model=model_name, revision=revision)

    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(text_to_analyze)

    return sentiment_result


def get_senti_df(sentences, sentiments):
    x = []
    y = []

    for senti in sentiments:
        score = senti['score']
        if senti['label'] == 'NEGATIVE':
            x.append(1-score)
            y.append(score)
        else:
            x.append(score)
            y.append(1-score)

    sents = {
        'review': sentences,
        'score': [senti['score'] for senti in sentiments],
        'label': [senti['label'] for senti in sentiments],
        'sentiments': sentiments,
        'positivity': x,
        'negativity': y
    }

    df = pd.DataFrame(sents)
    return df


def senti_plot(df):
    scatter = alt.Chart(df).mark_circle(size=60).encode(
        x='positivity',
        y='negativity',
        color=alt.Color('label:N', scale=alt.Scale(
            domain=['POSITIVE', 'NEGATIVE'], range=['blue', 'red'])),
        tooltip=['positivity', 'negativity', 'review']
    ).interactive()

    st.subheader('Sentiments Tendencies')
    st.altair_chart(scatter, use_container_width=True)
    st.write('This plot shows the probability of each review sentence. The x-axis represents the probability of a sentiment being positive, while y represents negative. For each review, if the value of x is greater than y, it will be classified as a positive review. The plot uses blue color for positive reviews and red color for negative reviews.')


def model_topics(reviews):
    processed_reviews = [preprocess_text(review) for review in reviews]

    dictionary = corpora.Dictionary(processed_reviews)
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]

    lda_model = models.LdaModel(
        corpus,
        num_topics=10,
        id2word=dictionary,
        passes=15
    )
    modelled_topics = lda_model.print_topics(num_words=2)

    return modelled_topics


def get_topics_df(modelled_topics):
    numbers = []
    scores = []
    topics = []
    for topic in modelled_topics:
        number = topic[0]
        top_weighted = topic[1].split("+")

        score = top_weighted[0].split("*")[0]
        top_words = [top_word.split("*")[1] for top_word in top_weighted]

        numbers.append(str(number))
        scores.append(float(score))
        topics.append((top_words[0]+top_words[1]).replace('"', ''))

    sents = {
        'score': scores,
        'topic': topics
    }
    df = pd.DataFrame(sents)
    df = df.sort_values(by='score', ascending=False)
    df['number'] = numbers

    return df


def topics_bar(df):
    topic_scores = alt.Chart(df).mark_bar().encode(
        alt.X("score").title("Topic score"),
        alt.Y("number").title("Topic number"),
        tooltip=[alt.Tooltip('topic:N', title='Topic: ')]
    ).properties(height=alt.Step(40))

    st.subheader('Top 10 Topics')
    st.altair_chart(topic_scores, use_container_width=True)
    st.write('The figure above illustrates the distribution of topic weights derived from a Latent Dirichlet Allocation (LDA) model applied to movie review sentences. Each bar on the chart corresponds to a specific topic (which can be seen by hovering your mouse cursor to a specific bar), labeled as "Topic 0," "Topic 1," "Topic 2.", and so on until "Topic 10". The height of each bar indicates the relative weight or prevalence of the respective topic within the entire corpus of movie reviews.')


def show_pie_chart(poss, negs):
    total = poss + negs
    poss_perc = str(poss/total*100)+"%"
    negs_perc = str(negs/total*100)+"%"

    source = pd.DataFrame({
        "Label": ["Positive", "Negative"],
        "Count": [poss, negs],
        "Percentage": [poss_perc, negs_perc]
    })

    pie = alt.Chart(source).mark_arc().encode(
        theta="Count",
        color=alt.Color(
            "Label",
            scale=alt.Scale(
                domain=["Positive", "Negative"],
                range=["#19c2fa", "#fc3f3f"]
            )),
        tooltip=["Label", "Count", "Percentage"]
    ).properties(

        height=480
    )

    st.subheader('Sentiments Distribution')
    st.altair_chart(pie, use_container_width=True)
    st.write('This chart shows a comparison between the number of positive and negative reviews by percentage.')


def rating_prediction(positivity):
    normalized_scores = [(score * 4) + 1 for score in (positivity)]
    average_score = sum(normalized_scores) / len(normalized_scores)
    rating = f'Average rating: {average_score:.2f}/5⭐'
    caption = 'The rating above is calculated by normalizing each review\'s positivity (probability of being a positive review) into the scale of 1 to 5.'

    source = pd.DataFrame({
        "score": normalized_scores
    })
    scores_chart = alt.Chart(source).mark_bar(color='#19fae3').encode(
        alt.X("score:Q", title="Ratings", bin=True),
        y='count()'
    )
    st.subheader(rating)
    st.altair_chart(scores_chart)
    st.write(caption)


def display_analysis(sentences):
    if len(sentences) == 0:
        st.write('there are currently no reviews for this')
        return

    sentiments = analyze_sentiment(sentences)
    senti_df = get_senti_df(sentences, sentiments)

    label_count = senti_df['label'].value_counts()
    poss = label_count.get('POSITIVE')
    negs = label_count.get('NEGATIVE')

    rating_prediction(senti_df['positivity'])
    wordcloud, scatter, bar, pie = st.tabs(['Overview', 'Senti-plot', 'Topics', 'Distribution'])
    with wordcloud:
        text = ' '.join(sentences)
        generate_wordcloud(extract_adj(text))
    with scatter:
        senti_plot(senti_df)
    with bar:
        modelled_topics = model_topics(sentences)
        topics_df = get_topics_df(modelled_topics)
        topics_bar(topics_df)
    with pie:
        show_pie_chart(poss, negs)


def main():
    st.set_page_config(layout='wide')
    st.title('Movie Sentiment analysis')

    response = search_movie()
    try:
        suggestions = show_suggestions(response)
        selected_movie = pick_suggestion(suggestions)
        url = selected_movie[0]

        st.subheader('Overal Review from Critics and Audience')

        critics_url = url+"/reviews?intcmp=rt-what-to-know_read-critics-reviews"
        reviews_url = url+"/reviews?type=user&intcmp=rt-what-to-know_read-audience-reviews"

        critics = get_critics(critics_url)
        reviews = get_reviews(reviews_url)

        critics_analysys, reviews_analysis = st.tabs(['Critics', 'Audiene'])
        with critics_analysys:
            st.write('Critic reviews are typically written by professional film critics who work for newspapers, magazines, websites, or other media outlets. These individuals are often considered experts in the field of film analysis and critique. Critics usually provide in-depth analyses of various aspects of a film, such as direction, screenplay, acting, cinematography, and overall storytelling. They may also consider the film\'s cultural or social impact. While critics aim to be objective in their evaluations, their reviews are still based on personal opinions and preferences. Different critics may have varying perspectives on the same movie.')
            display_analysis(critics)

        with reviews_analysis:
            st.write('Audience reviews come from regular moviegoers or viewers who share their opinions and experiences after watching a film. Audience reviews represent a wide range of perspectives, as they come from individuals with different tastes, preferences, and cultural backgrounds. They may focus on elements that resonate with general audiences. Audience reviews tend to be more subjective and emotional, reflecting personal connections and reactions to the movie. These reviews may prioritize entertainment value and emotional impact over technical aspects.')
            display_analysis(reviews)

    except AttributeError:
        # st.write('Movie suggestions will be shown here')
        pass

    except MissingSchema:
        # st.write('Ater you picked a movie, wordcloud will be shown here')
        pass


if __name__ == '__main__':
    main()
