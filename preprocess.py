import pandas as pd
import random
import os
random.seed(42)

def read_txt_file(file_path):
    queries = []
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Extract query and label
            if 'query:' in line and 'label:' in line:
                query = line.split(' ')[0].split('query:')[1].strip()
                label = int(line.split('label:')[1].strip())
                queries.append(query)
                labels.append(label)
    return queries, labels

def split_data(queries, labels, train_ratio=0.7, val_ratio=0.1):
    data = list(zip(queries, labels))
    random.shuffle(data)
    train_size = int(len(data) * train_ratio)
    val_size = int(len(data) * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    print(f"Train data size: {len(train_data)}, Val data size: {len(val_data)}, Test data size: {len(test_data)}")
    return train_data, val_data, test_data

def save_to_csv(data, file_name):
    df = pd.DataFrame(data, columns=['query', 'label'])
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    save_dir = 'data'
    file_path = os.path.join(save_dir, 'data.txt') # Replace with your txt file path
    queries, labels = read_txt_file(file_path)

    train_data, val_data, test_data = split_data(queries, labels, train_ratio=0.7, val_ratio=0.1)

    save_to_csv(train_data, os.path.join(save_dir, 'train.csv'))
    save_to_csv(val_data, os.path.join(save_dir, 'val.csv'))
    save_to_csv(test_data, os.path.join(save_dir, 'test.csv'))

    print("Data split into train, val, and test CSV files.")