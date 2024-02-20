from nltk import pos_tag
from bs4 import BeautifulSoup
from wordcloud import WordCloud
from transformers import pipeline
from nltk.tokenize import word_tokenize
from requests.exceptions import MissingSchema
import nltk, requests, streamlit as st, matplotlib.pyplot as plt

def search_movie():
    # Create a text input for the search bar
    movie_searched = st.text_input('Search movie title:', '')

    # Api for querying the searched movie
    # Source: rotten tomatoes
    search_url = f"https://www.rottentomatoes.com/search?search={movie_searched}"
    response = requests.get(search_url)
    return response

def show_suggestions(response):
    soup = BeautifulSoup(response.content, 'html.parser')
    search_result_elements = soup.find('search-page-result', attrs={'type': 'movie', 'data-qa': 'search-result'})
    property_elements = search_result_elements.find_all('a', class_='unset', attrs={'data-qa': 'info-name', 'slot': 'title'})
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
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all div review elements
    property_elements = soup.find_all('p', class_="audience-reviews__review js-review-text")

    # Extract text content without <p> tags
    reviews_text = [element.text.strip().lower() for element in property_elements]

    return reviews_text

def get_critics(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all div elements with class "ListingCell-KeyInfo-price"
    property_elements = soup.find_all('p', class_='review-text', attrs={'data-qa': 'review-quote'})

    # Extract text content without <p> tags
    reviews_text = [element.text.strip().lower() for element in property_elements]

    return reviews_text

def extract_adj(text):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

    # Tokenize and perform part-of-speech tagging
    tokens = word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract adjectives
    adjectives = [word for word, pos in pos_tags if pos.startswith('JJ') or pos.startswith('JJR') or pos.startswith('JJS')]

    # Join adjectives into a single string
    all_adjectives = ' '.join(adjectives)
    return all_adjectives

def analyze_sentiment(text_to_analyze):
    # Specify the model and revision explicitly
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    revision = "af0f99b"

    # Load the sentiment analysis pipeline with explicit model and revision
    sentiment_analyzer = pipeline("sentiment-analysis", model=model_name, revision=revision)

    # Perform sentiment analysis
    sentiment_result = sentiment_analyzer(text_to_analyze)

    return sentiment_result

def get_coordinates(sentiment_result):
    pos_x = []
    pos_y = []
    neg_x = []
    neg_y = []
    for dct in sentiment_result:
        if dct['label'] == 'POSITIVE':
            x = dct['score']
            y = 1-x
            pos_x.append(x)
            pos_y.append(y)
        else:
            y = dct['score']
            x = 1-y
            neg_x.append(x)
            neg_y.append(y)
    
    return [pos_x, pos_y, neg_x, neg_y]

def generate_wordcloud(words):
    # Generate WordCloud for adjectives
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(words)
    # Display the WordCloud using matplotlib

    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    st.write('The figure above is a visual representation of the review data displayed in different sizes based on their frequency. The more frequent a word appears in the review, the larger and bolder it appears in the word cloud. This can be used to identify the most prominent and frequently mentioned terms. This wordcloud only displays adjectives as it allows the reader to focus on the descriptive elements and sentiments associated with the films.')

def show_scatter_plot(coords):
    pos_x, pos_y, neg_x, neg_y = coords[0], coords[1], coords[2], coords[3]
    # Create a figure and axis
    fig, ax = plt.subplots()

    ax.scatter(pos_x, pos_y, color='blue', label='Positive Reviews', s=5)
    ax.scatter(neg_x, neg_y, color='red', label='Negative Reviews', s=5)

    ax.set_xlabel('Positive')
    ax.set_ylabel('Negative')

    # Customize the plot
    ax.set_title('Reviews Tendency')
    ax.legend()  # Show legend with labels

    # Show the plot
    st.pyplot(fig)
    st.write('This plot shows the probability of each review sentence. The x-axis represents probability of a sentiment is positive, while y represents negative. For each review, if the value of x is greater than y, it will be classified as a positive review. Same goes the other way.')

def show_bar_chart(num_elements_set_1, num_elements_set_2):
    # Bar chart
    fig, ax = plt.subplots()
    ax.bar(['Positive', 'Negative'], [num_elements_set_1, num_elements_set_2], color=['blue', 'red'])

    # Customize the plot
    ax.set_title('Sentiments Comparison')
    ax.set_ylabel('Number of Sentiments')

    # Show the plot
    st.pyplot(fig)
    st.write('This chart shows a comparison between the number of positive and negative reviews based on their occurrence.')

import matplotlib.pyplot as plt
import streamlit as st

def show_pie_chart(num_elements_set_1, num_elements_set_2):
    # Pie chart
    labels = ['Positive', 'Negative']
    sizes = [num_elements_set_1, num_elements_set_2]
    colors = ['blue', 'red']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    
    # Customize the plot
    ax.set_title('Sentiments Distribution')

    # Show the plot
    st.pyplot(fig)
    st.write('This chart shows a comparison between the number of positive and negative reviews by percentage.')

def rating_prediction(pos_x, neg_x):
    normalized_scores = [(score * 4) + 1 for score in (pos_x + neg_x)]
    average_score = sum(normalized_scores) / len(normalized_scores)
    rating = f'Rating prediction: {average_score:.2f}/5⭐'
    caption = 'The rating above is calculated by normalizing each review\'s probability (tendency of either positive or negative) into the scale of 1 to 5, then taking their average.'

    st.subheader(rating)
    st.write(caption)

def display_analysis(sentences):
    if len(sentences) == 0:
        st.write('there are currently no reviews for this')
        return
    
    text = ' '.join(sentences)
    critics_sentiment = analyze_sentiment(sentences)
    critics_coords = get_coordinates(critics_sentiment)
    pos_x, neg_x = critics_coords[0], critics_coords[2]
    poss, negs = len(pos_x), len(neg_x)
    
    rating_prediction(pos_x, neg_x)

    wordcloud, scatter, pie, bar = st.tabs(['Overview', 'Senti-plot', 'Percentage', 'Comparisson'])
    with wordcloud:    
        generate_wordcloud(extract_adj(text))
    with scatter:
        show_scatter_plot(critics_coords)
    with pie:
        show_pie_chart(poss, negs)
    with bar:
        show_bar_chart(poss, negs)

# Streamlit app
def main():
    st.set_page_config(layout='wide')
    st.title('Movie Sentiment analysis')

    response = search_movie()
    # Scraping elements
    # print(response)
    try:
        suggestions = show_suggestions(response)
        selected_movie = pick_suggestion(suggestions)
        url = selected_movie[0]
        # title = selected_movie[1]
        
        st.subheader('Overal Review From Critics and Audience')

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
