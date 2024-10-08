from flask import Flask, render_template, request, send_file
import pandas as pd
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from youtube_API import get_youtube_comments
from urllib.parse import urlparse, parse_qs
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Initialize the flas application
app = Flask(__name__)

# Loads the toxic comment data
data = pd.read_csv('toxic_comments.csv')

# Preprocess the text by removing punctuation and changing it to lowercase
def preprocess_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    return text
# Applies the preprocessing to the text column
data['Text'] = data['Text'].apply(preprocess_text)

# Separates features and labels
x = data['Text']
y = data.drop(columns=['Text'])

# Splits data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Converts text to numerical features
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Creates a multi-output classifier using neural networks
classifier = MultiOutputClassifier(MLPClassifier(max_iter=1000))
classifier.fit(x_train_vec, y_train)

# Predicts categories for given comment
def predict_comment(comment):
    comment_vec = vectorizer.transform([preprocess_text(comment)])
    prediction = classifier.predict(comment_vec)
    labels = y.columns
    return [labels[i] for i in range(len(prediction[0])) if prediction[0][i]]

# Extracts video ID from link
def get_video_id_from_url(url):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    return query_params.get('v', [None])[0]

# Predicts categories for comments on youtube video
def predict_youtube_comments(video_id, api_key, max_results=50):
    comments = get_youtube_comments(video_id, api_key, max_results)
    predictions = []
    all_labels = []

    for comment in comments:
        results = predict_comment(comment)
        predictions.append({'comment': comment, 'results': results})
        all_labels.extend(results)

    return predictions, all_labels

# Generates bar gragh
def generate_bar_graph(labels):
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(10, 6))
    bars = label_counts.plot(kind='bar', color='skyblue')
    plt.title('Comment Category Distribution')
    plt.xlabel('Categories')
    plt.ylabel('Number of Comments')

    for idx, value in enumerate(label_counts):
        plt.text(idx, value, str(value), ha='center', va='bottom')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Generates pie chart
def generate_pie_chart(labels):
    label_counts = pd.Series(labels).value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(label_counts, labels=label_counts.index, autopct='%1.1f%%', colors=plt.cm.Paired.colors)
    plt.title('Comment Category Percentage')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Generates heatmap
def generate_metrics_heatmap():
    y_pred = classifier.predict(x_test_vec)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        'Classes': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    class_mapping = {
        0: 'isToxic',
        1: 'IsAbusive',
        2: 'IsThreat',
        3: 'IsProvocative',
        4: 'IsObscene',
        5: 'IsHatespeech',
        6: 'IsRacist',
        7: 'IsNationalist',
        8: 'IsSexist',
        9: 'IsHomophobic',
        10: 'IsReligiousHate',
        11: 'IsRadicalism'
    }

    for label, scores in report.items():
        if label not in ['accuracy', 'micro avg', 'macro avg', 'weighted avg', 'samples avg']:
            metrics['Classes'].append(class_mapping[int(label)])
            metrics['Precision'].append(scores['precision'])
            metrics['Recall'].append(scores['recall'])
            metrics['F1 Score'].append(scores['f1-score'])

    for avg in ['micro avg', 'macro avg', 'weighted avg']:
        metrics['Classes'].append(avg)
        metrics['Precision'].append(report[avg]['precision'])
        metrics['Recall'].append(report[avg]['recall'])
        metrics['F1 Score'].append(report[avg]['f1-score'])

    metrics_df = pd.DataFrame(metrics)
    metrics_df.set_index('Classes', inplace=True)

    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics_df, annot=True, cmap='Blues', fmt='.2f')
    plt.title('Accuracy Metric Heatmap')
    plt.xlabel('Metrics')
    plt.ylabel('Classes')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return img

# Defines the main route for the app
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        youtube_url = request.form['youtube_url']
        video_id = get_video_id_from_url(youtube_url)

        if video_id:
            api_key = 'YOUR API'
            predictions, all_labels = predict_youtube_comments(video_id, api_key, max_results=50)
            return render_template('results.html', predictions=predictions, has_graph=True, youtube_url=youtube_url)
        else:
            return render_template('index.html', error="Invalid YouTube URL.")

    return render_template('index.html')

# Routes to bar graph
@app.route('/graph')
def display_graph():
    youtube_url = request.args.get('youtube_url')
    video_id = get_video_id_from_url(youtube_url)

    if video_id:
        api_key = 'YOUR API'
        _, all_labels = predict_youtube_comments(video_id, api_key, max_results=50)
        img = generate_bar_graph(all_labels)
        return send_file(img, mimetype='image/png')

    return "No graph available."

# Routes to pie chart
@app.route('/pie_chart')
def display_pie_chart():
    youtube_url = request.args.get('youtube_url')
    video_id = get_video_id_from_url(youtube_url)

    if video_id:
        api_key = 'YOUR API'
        _, all_labels = predict_youtube_comments(video_id, api_key, max_results=50)
        img = generate_pie_chart(all_labels)
        return send_file(img, mimetype='image/png')

    return "No pie chart available."

# Route to heat map
@app.route('/heatmap')
def display_heatmap():
    img = generate_metrics_heatmap()
    return send_file(img, mimetype='image/png')

# Runs the flask app
if __name__ == '__main__':
    app.run(debug=True)
