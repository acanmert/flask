import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def secimlik(secim):
    data = pd.read_csv(f'{secim}', encoding='ISO-8859-9')
    return data

def create_combined_features(row, selected_features):
    return ' '.join([str(row[feature]) for feature in selected_features])

def get_recommendations(title, data, selected_features,p_name,p_type):
    try:
        # Kullanıcının seçtiği özelliklere göre 'combined_features' sütununu oluştur
        data['combined_features'] = data.apply(lambda row: create_combined_features(row, selected_features.split(",")), axis=1)

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
            return ["No recommendation available"]
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return ["Error occurred while getting recommendations"]

def get_recommendations_endpoint():
    secim = "book_data.csv"
    data = secimlik(secim)

    selected_features = "Name,Genre"
    title = "ciglik"
    p_name="Name"
    p_type="hayir"
    recommendations = get_recommendations(title, data, selected_features,p_name,p_type)
    return recommendations
get_recommendations_endpoint()
