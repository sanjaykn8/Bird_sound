import re
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


ZIP_PATH = Path('/mnt/data/online-hotel-reviews.zip')
WORKDIR = Path('/mnt/data')
SUBMISSION_PATH = WORKDIR / 'submission.csv'


def read_csv_from_zip(zf: zipfile.ZipFile, name: str) -> pd.DataFrame:
    with zf.open(name) as f:
        return pd.read_csv(f)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
    return df


def norm_name(name: str) -> str:
    return str(name).strip().lower().replace(' ', '_')


def get_sentiment(rating):
    if rating > 3:
        return 'Good'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Bad'


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)          # strip HTML
    text = re.sub(r'\s+', ' ', text)             # collapse whitespace
    return text.strip()


def main():
    with zipfile.ZipFile(ZIP_PATH) as zf:
        train_raw = read_csv_from_zip(zf, 'train.csv')
        test_raw = read_csv_from_zip(zf, 'test.csv')
        sample_raw = read_csv_from_zip(zf, 'sample_submission.csv')

    train = normalize_columns(train_raw)
    test = normalize_columns(test_raw)

    review_col_train = 'review'
    rating_col_train = 'rating'

    # Find the review column in test data if the name differs slightly.
    review_col_test = next((c for c in test.columns if c in {'review', 'text', 'review_text'}), None)
    if review_col_test is None:
        raise KeyError(f'Could not find review column in test.csv. Columns: {list(test.columns)}')

    train['sentiment'] = train[rating_col_train].apply(get_sentiment)

    X = train[review_col_train].fillna('').map(clean_text)
    y = train['sentiment']

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        (
            'tfidf',
            TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=3,
                max_features=50000,
                sublinear_tf=True,
            ),
        ),
        (
            'clf',
            LinearSVC(
                class_weight='balanced',
            ),
        ),
    ])

    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)

    print('Validation macro F1:', f1_score(y_val, val_pred, average='macro'))
    print('\nClassification report:\n')
    print(classification_report(y_val, val_pred))

    # Train on full data
    model.fit(X, y)

    test_texts = test[review_col_test].fillna('').map(clean_text)
    test_pred = model.predict(test_texts)

    # Match the sample submission columns exactly.
    sentiment_to_rating = {'Bad': 1, 'Neutral': 3, 'Good': 5}
    rating_pred = [sentiment_to_rating[s] for s in test_pred]

    id_col_sample = sample_raw.columns[0]
    pred_col_sample = sample_raw.columns[1] if len(sample_raw.columns) > 1 else 'Rating'

    id_col_test = next((c for c in test_raw.columns if norm_name(c) == norm_name(id_col_sample)), None)
    if id_col_test is None:
        # fall back to the first non-review, non-rating column
        id_col_test = next((c for c in test_raw.columns if c not in {review_col_test, 'rating'}), None)
    if id_col_test is None:
        raise KeyError(f'Could not infer ID column in test.csv. Columns: {list(test_raw.columns)}')

    submission = pd.DataFrame({
        id_col_sample: test_raw[id_col_test].values,
        pred_col_sample: rating_pred,
    })

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f'\nSaved: {SUBMISSION_PATH}')


if __name__ == '__main__':
    main()
