from flask import Flask, request, jsonify
import pandas as pd
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

def get_recommendations_fuzzy(p_name, data, selected_features, p_pk, p_type, top_n):
    try:
        # İlgili sütunlardan seçilen verileri birleştirin
        data['combined_features'] = data[selected_features].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

        # Kullanıcının girdisine göre benzer öğeleri bulma
        if p_pk in data.columns:
            if p_type == "Evet":
                index = data[data[p_pk] == int(p_name)].index[0]
            else:
                index = data[data[p_pk] == p_name].index[0]

            # Kullanıcının girdiği kitap ismi veya diğer özelliklerin birleşik hali
            query_text = data.iloc[index]['combined_features']

            # FuzzyWuzzy ile en benzer kitapları bulma
            results = process.extract(query_text, data['combined_features'], scorer=fuzz.ratio, limit=top_n)

            # Sonuçları orijinal verilerle eşleştirip döndürme
            similar_items = []
            for result in results:
                match_text, score, match_index = result
                similar_item_dict = data.iloc[match_index].to_dict()
                similar_items.append(similar_item_dict)

            return similar_items
        else:
            return ["No recommendation available"]
    except Exception as e:
        return [f"Error occurred: {str(e)}"]
