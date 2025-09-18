import pandas as pd
import numpy as np
import os
import sys
from collections import defaultdict
import math
from scipy.sparse import csr_matrix
import json


def n_orders_weight_matrix(df):
    
    unique_orders = df['order_id'].unique()
    unique_products = df['plu_id'].unique()
    
    n_orders = len(unique_orders)
    n_products = len(unique_products)
    
    order_to_idx = {order_id: idx for idx, order_id in enumerate(unique_orders)}
    product_to_idx = {plu_id: idx for idx, plu_id in enumerate(unique_products)}
    idx_to_product = {idx: plu_id for plu_id, idx in product_to_idx.items()}
    
    # Построение бинарной матрицы
    rows = df['order_id'].map(order_to_idx).values
    cols = df['plu_id'].map(product_to_idx).values
    data = np.ones(len(df), dtype=np.int32)
    
    # Создание бинарной sparse матрицы
    binary_matrix = csr_matrix((data, (rows, cols)), 
                               shape=(n_orders, n_products), 
                               dtype=np.int32)
    
    # Матрица совместной встречаемости: plu_id x plu_id
    cooccurrence_matrix = binary_matrix.T.dot(binary_matrix)
    
    # Нормализация: деление ВСЕХ элементов матрицы на общее число заказов
    normalized_matrix = cooccurrence_matrix.astype(np.float64) / n_orders *10000
    
    # Преобразуем в DataFrame
    result_df = pd.DataFrame(
        normalized_matrix.toarray(),
        index=[idx_to_product[i] for i in range(n_products)],
        columns=[idx_to_product[i] for i in range(n_products)]
    )

    # Сортировка по главной диагонали (по убыванию)
    col_sums = result_df.sum(axis=0)
    sorted_indices = np.argsort(col_sums.values)[::-1]  
    
    # Переставляем строки и столбцы
    result_df = result_df.iloc[sorted_indices, sorted_indices]

    return result_df

def clear_df(df, matrix, max_height = 30, fullness = 0.85):
    # Функция для выявления наиболее часто встречающихся товаров (288 штук)
    def select_type_A(df, len = 288):

        data = df['plu_id'].value_counts().head(len)
        
        return pd.DataFrame(data).reset_index()

    type_A_df = select_type_A(df)

    # Исключение товаров типа A
    clear_df = df[~df['plu_id'].isin(type_A_df['plu_id'])]

    # Исключение замороженных товаров, ФРОВ
    clear_df = clear_df[(clear_df['plu_hierarchy_lvl_2_desc'] != 'Замороженные продукты')&
                        (clear_df['plu_hierarchy_lvl_2_desc'] != 'Овощи - Фрукты')&
                        (clear_df['plu_hierarchy_lvl_3_desc'] != 'Орехи')]
    
    # Исключение КГТ
    matrix_large = matrix[matrix['plu_height_amt'] > max_height*fullness]
    clear_df = clear_df[~clear_df['plu_id'].isin(matrix_large['plu_id'])]

    
    return clear_df.reset_index()

def box_distrib(df, matrix, compatibility_matrix, box_volume = 60*40*30*0.85, res_volume = 0.08, max_num_items = 6):

    data = df.copy()
    box_counter = 0
    result = []

    single_items = []

    while data.columns.size > 0:

        item = data.iloc[:, 0].reset_index()

        matrix['plu_id'] = pd.to_numeric(matrix['plu_id'], errors='coerce')
        item = pd.merge(item, matrix[['plu_id','volume','plu_hierarchy_lvl_2_desc']], how = 'left', left_on = 'index', right_on = 'plu_id').drop_duplicates(subset='index')

        item['relative_volume'] = item['volume']/box_volume
        item['volume_score'] = 1-item['relative_volume']

        item['cooc_score'] = item['volume_score']*item.iloc[:, 1]

        item = item.sort_values(by = 'cooc_score', ascending = False)

        main_item_row = item.iloc[0]
        main_plu = main_item_row['index']
        main_cat = main_item_row['plu_hierarchy_lvl_2_desc']
        main_vol = main_item_row['volume']
        main_rel_vol = main_item_row['relative_volume']
        main_concurrence = main_item_row.iloc[1]

        if main_rel_vol > 1 - res_volume:
            data = data.drop(index=[main_plu], columns=[main_plu], errors='ignore')
            continue    

        box_list = [(main_plu, main_vol, main_concurrence)]
        del_index = [main_plu]
        cum_vol = main_rel_vol

        for i in range(1, len(item)):

            current_vol = item.iloc[i]['relative_volume']
            
            # Проверка совместимости — безопасная
            index_alt, index_main = item.iloc[i]['plu_hierarchy_lvl_2_desc'], item.iloc[0]['plu_hierarchy_lvl_2_desc']
            if pd.isna(index_alt) or pd.isna(index_main):
                is_compatible = 1  

            if compatibility_matrix.isin([index_alt]).any().any() and compatibility_matrix.isin([index_main]).any().any():
                is_compatible = compatibility_matrix.loc[
                    (compatibility_matrix['Категория_1'] == index_main) & 
                    (compatibility_matrix['Категория_2'] == index_alt),
                    'Совместимость'
                ].iloc[0]
            else:
                is_compatible = 1  

            if is_compatible != 1:
                continue
            
            if (cum_vol + current_vol) > 1-res_volume:
                if 1 - cum_vol < item['relative_volume'].min():
                    break
                else:
                    continue
                
            if len(box_list) >= max_num_items:
                break

            box_list.append((item.iloc[i]['index'], 
                            item.iloc[i]['volume'], 
                            item.iloc[i, 1]))
            del_index.append(item.iloc[i]['index'])
            cum_vol += current_vol
            
        if not box_list:
            break

        box_counter += 1
        

        # В конце цикла, если в box_list только один товар:
        if len(box_list) == 1:
            single_items.append({
                'plu_id': int(main_plu),
                'volume': float(main_vol),
            })

        for neighbor_plu, neighbor_vol, concurrence in box_list:
            if neighbor_plu != main_plu:
                result.append({
                    'main_item': main_plu,
                    'main_item_volume': main_vol,
                    'neighbor_item': neighbor_plu,
                    'neighbor_volume': neighbor_vol,
                    'concurrence': concurrence
                })
        data = data.drop(index = del_index, columns = del_index, errors = 'ignore')

    result_df = pd.DataFrame(   
        result,
        columns=[
            'main_item', 'main_item_volume',
            'neighbor_item', 'neighbor_volume', 'concurrence'
        ]
    )
    return result_df, data, pd.DataFrame(single_items)

# Основная программа

def main():
    print("Алгоритм упаковки товаров\n")

    # Ввод путей
    json_params = input("Укажите путь к json файлу с параметрами: ").strip('"\' ')

    try:
        print("\nЗагрузка данных...")
        with open(json_params, 'r', encoding='utf-8') as file:
            params = json.load(file)

        df = pd.read_csv(params['sales_path'], on_bad_lines='skip')
        matrix = pd.read_csv(params['dim_path'])
        compatibility_matrix = pd.read_csv(params['compatibility_path'], sep=';', on_bad_lines='skip', encoding='cp1251').dropna()

        # Расчёт объёма
        size = ['plu_height_amt', 'plu_width_amt', 'plu_depth_amt', 'Квант']
        matrix['Квант'] = matrix['Квант'].fillna(1)
        matrix[size] = matrix[size].astype('float64', errors = 'ignore')
        matrix['volume'] = matrix['plu_height_amt'] * matrix['plu_width_amt'] * matrix['plu_depth_amt'] * matrix['Квант']

        print("Данные загружены")

        # Очистка
        print("\nОчистка данных...")
        clean_data = clear_df(df, matrix, params['height'], params['fullness'])

        # Матрица ко-заказов
        print("Построение матрицы совместной встречаемости...")
        mtx = n_orders_weight_matrix(clean_data)
        mtx = mtx  # ограничение для скорости

        # Упаковка
        print("Запуск алгоритма упаковки...")
        result_df, remaining, single_items = box_distrib(mtx, matrix, compatibility_matrix, 
                                           params['height']*params['width']*params['depth']*params['fullness'], 
                                           params['reserv_volume'], params['max_plu_number'])

        # Сохранение результата
        output_path = "результат_упаковки.csv"
        result_df.to_csv(output_path, index=False, encoding='utf-8-sig')  # utf-8-sig — для Excel
        single_output_path = "товары_одиночки.csv"
        single_items.to_csv(single_output_path, index=False, encoding='utf-8-sig')
        print(f"\nГотово! Результат сохранён: {os.path.abspath(output_path)}")
        print(f"Осталось товаров: {remaining.shape[1]}")

    except Exception as e:
        print(f"Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()

    finally:
        input("\nНажмите Enter, чтобы выйти...")


if __name__ == "__main__":
    main()