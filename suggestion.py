import pandas as pd
import cosine, knn, fuzzy


def secimlik(secim,email):
    data = pd.read_csv(f'C:\\Users\\AhmetCan\\source\\repos\\Backend\\Backend\\wwwroot\\dataset\\{email}\\{secim}',
                       encoding='"utf-8-sig')
    return data


def find_common_items(recommendations_cosine, recommendations_knn, recommendations_fuzzy, key):
    # Ortak olan öğeleri bul
    common_keys = (set(item[key] for item in recommendations_cosine) &
                   set(item[key] for item in recommendations_knn) &
                   set(item[key] for item in recommendations_fuzzy))

    # Ortak öğeleri listelere göre filtrele
    common_items = [item for item in recommendations_cosine if item[key] in common_keys]
    return common_items

def get_recommendations(p_name, data, selected_features, p_pk, p_type, top_n):
    recommendations_cosine=cosine.get_recommendations_cosine(p_name, data, selected_features, p_pk, p_type, top_n)
    recommendations_knn = knn.get_recommendations_knn(p_name, data, selected_features, p_pk, p_type, top_n)
    recommendations_fuzzy = fuzzy.get_recommendations_fuzzy(p_name, data, selected_features, p_pk, p_type, top_n)

    common_elements = []

    for entity in recommendations_cosine:
        if entity in recommendations_fuzzy or entity in recommendations_knn:
            common_elements.append(entity)

    for entity in recommendations_fuzzy:
        if entity in recommendations_knn:
            if not common_elements.__contains__(entity):
                common_elements.append(entity)

    for entity in recommendations_knn:
        if entity in recommendations_fuzzy:
            if not common_elements.__contains__(entity):
                common_elements.append(entity)
    count=0

    while len(common_elements) < top_n:
        if recommendations_knn[count] not in common_elements:
            common_elements.append(recommendations_knn[count])

        if len(common_elements) >= top_n:
            break
        if recommendations_cosine[count] not in common_elements:
            common_elements.append(recommendations_cosine[count])
        if len(common_elements) >= top_n:
            break
        if recommendations_fuzzy[count] not in common_elements:
            common_elements.append(recommendations_fuzzy[count])
        count += 1

    return common_elements