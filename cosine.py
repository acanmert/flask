from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_combined_features(row, selected_features):
    return ' '.join([str(row[feature]) for feature in selected_features])

def get_recommendations_cosine(title, data, selected_features,p_name,p_type, top_n):
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
            top_recommendations = similarity_scores[1:top_n + 1]

            recommended_rows = []
            for top_data in top_recommendations:
                recommended_rows.append(data.iloc[top_data[0]].to_dict())

            return recommended_rows #, [score[1] for score in top_recommendations]
        else:
            return ["No recommendation available"]  # veya başka bir hata durumu işleme
    except Exception as e:
        #print(f"Error occurred: {str(e)}")
        return [f"Error occurred: {str(e)}"]




