import argparse, os
import numpy as np
import joblib
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory containing the facial features embeddings",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        required=True,
        help="Directory containing the corresponding aligned face images",
    )
    parser.add_argument(
        "--model_output",
        type=str,
        required=True,
        help="Path to save the trained model",
    )

    opt = parser.parse_args()
    embeddings_dir = opt.embeddings_dir
    images_dir = opt.images_dir
    model_output = opt.model_output

    backend = set_backend("numpy", on_error="warn")

    alpha = [0.000001,0.00001,0.0001,0.001,0.01, 0.1, 1]

    ridge = RidgeCV(alphas=alpha)

    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )    

    embeddings = []
    images = []
    for embedding_file in os.listdir(embeddings_dir):
        image_file = embedding_file.replace('.npy', '.png')
        embeddings.append(np.load(os.path.join(embeddings_dir, embedding_file)).astype("float32"))
        images.append(np.load(os.path.join(images_dir, image_file)).astype("float32").reshape([-1]))

    X = np.vstack(embeddings)
    Y = np.vstack(images)

    print(f'Now making decoding model for all embeddings and images')
    print(f'X {X.shape}, Y {Y.shape}')
    pipeline.fit(X, Y)
    scores = pipeline.predict(X)
    rs = correlation_score(Y.T,scores.T)
    print(f'Prediction accuracy is: {np.mean(rs):3.3}')

    joblib.dump(pipeline, model_output)

if __name__ == "__main__":
    main()

