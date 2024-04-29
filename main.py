from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

def secimlik(secim):
    data = pd.read_csv(f'C:\\Users\\AhmetCan\\source\\repos\\File_Upload\\File_Upload\\wwwroot\\dataset\\{secim}', encoding='ISO-8859-9')
    return  data

def create_combined_features(row, selected_features):
    return ' '.join([str(row[feature]) for feature in selected_features])

def get_recommendations(title, data, selected_features,p_name,p_type):
    try:
        # Kullanıcının seçtiği özelliklere göre 'combined_features' sütununu oluştur
        data['combined_features'] = data.apply(lambda row: create_combined_features(row, selected_features), axis=1)

        # Kullanıcının girdisine göre film önerilerini al
        if p_name in data.columns:
            if p_type == "evet":
                index = data[data[p_name] == int(title)].index[0]
            else:
                index = data[data[p_name] == title].index[0]
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(data['combined_features'])
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            similarity_scores = list(enumerate(cosine_sim[index]))
            similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
            top_recommendations = similarity_scores[1:11]
            recommended = [data[p_name][top_data[0]] for top_data in top_recommendations]
            return recommended
        else:
            return ["No recommendation available"]  # veya başka bir hata durumu işleme
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return ["Error occurred while getting recommendations"]
@app.route('/recommendations', methods=['GET'])
def get_recommendations_endpoint():
    secim = request.args.get('secim')
    data = secimlik(secim)
    selected_features = request.args.get('selected_features').split(',')
    title = request.args.get('title')
    p_name=request.args.get("p_name")
    p_type=request.args.get("p_type")
    recommendations = get_recommendations(title, data, selected_features,p_name,p_type)

    return     str(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
