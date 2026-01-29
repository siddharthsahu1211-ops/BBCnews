import os
import pandas as pd

def create_bbc_dataset():
    """Create CSV dataset from BBC folder structure"""
    texts = []
    categories = []
    
    base_path = "../bbc"
    
    for category in os.listdir(base_path):
        category_path = os.path.join(base_path, category)
        if not os.path.isdir(category_path):
            continue
        
        for filename in os.listdir(category_path):
            file_path = os.path.join(category_path, filename)
            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    texts.append(f.read())
                    categories.append(category)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    bbc_text = pd.DataFrame({
        "text": texts,
        "category": categories
    })
    
    # Save to CSV
    bbc_text.to_csv("bbc-text.csv", index=False)
    print(f"Dataset created with {len(bbc_text)} articles")
    print(f"Categories: {bbc_text['category'].value_counts()}")
    
    return bbc_text

if __name__ == "__main__":
    df = create_bbc_dataset()
    print(df.head())