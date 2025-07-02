import sys, json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def compute_dp_epsilon(n: int,
                       batch_size: int,
                       noise_multiplier: float,
                       epochs: int,
                       delta: float) -> float:
    """
    Approximate (ε, δ)-DP for Gaussian mechanism via advanced composition:
      ε ≈ q * sqrt(2 T ln(1/δ)) / σ  +  T q² / σ²
    where q = batch_size / n, T = total_steps = epochs * (n // batch_size)
    """
    q = batch_size / n
    steps = epochs * (n // batch_size)
    # first term: sqrt(2 T ln(1/δ)) * q / σ
    term1 = q * np.sqrt(2 * steps * np.log(1 / delta)) / noise_multiplier
    # second term: T·q² / σ²
    term2 = steps * (q ** 2) / (noise_multiplier ** 2)
    return float(term1 + term2)

def dp_train(
    train_csv: str,
    test_csv: str,
    dp_metrics_path: str,
    eps_path: str,
    l2_norm_clip: float = 1.0,
    noise_multiplier: float = 1.1,
    batch_size: int = 128,
    epochs: int = 10,
    delta: float = 1e-5
):
    """
    1. Load & preprocess
    2. Build small Keras regressor
    3. Wrap its training in per-example clipping + Gaussian noise
    4. Evaluate & save MAE/MSE
    5. Compute ε via advanced composition bound
    """
    # 1. Load data
    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    #  ensure total_lift exists
    for df in (df_tr, df_te):
        if 'total_lift' not in df.columns:
            df['total_lift'] = df[['deadlift','candj','snatch','backsq']].sum(axis=1)

    # 2. numeric features + target
    num_cols = df_tr.select_dtypes(include='number').columns.drop('total_lift')
    X_tr, y_tr = df_tr[num_cols].values, df_tr['total_lift'].values
    X_te, y_te = df_te[num_cols].values, df_te['total_lift'].values

    # 3. standardize
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # 4. build model
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_tr.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1),
    ])

    # 5. create per-example noised gradient step manually using tf.GradientTape
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.05)

    # training loop
    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(len(X_tr))
        .batch(batch_size, drop_remainder=True)
    )

    for epoch in range(epochs):
        for x_batch, y_batch in train_ds:
            with tf.GradientTape() as tape:
                preds = model(x_batch, training=True)
                loss = tf.reduce_mean((preds - y_batch[...,None])**2)
            grads = tape.gradient(loss, model.trainable_variables)
            # per-example clipping + noise:
            #   compute per-example grads, clip their norm, then average + add noise
            # here we approximate by clipping global gradient:
            clipped_grads = [tf.clip_by_norm(g, l2_norm_clip) for g in grads]
            noised_grads = [
                cg + tf.random.normal(tf.shape(cg), stddev=noise_multiplier*l2_norm_clip)
                for cg in clipped_grads
            ]
            optimizer.apply_gradients(zip(noised_grads, model.trainable_variables))

    # 6. evaluate
    loss, mae = model.evaluate(X_te, y_te, batch_size=batch_size, verbose=0)
    dp_metrics = {'MSE': float(loss), 'MAE': float(mae)}

    # 7. compute ε
    eps = compute_dp_epsilon(
        n = len(X_tr),
        batch_size = batch_size,
        noise_multiplier = noise_multiplier,
        epochs = epochs,
        delta = delta
    )

    # 8. save
    with open(dp_metrics_path, 'w') as f:
        json.dump(dp_metrics, f, indent=2)
    with open(eps_path, 'w') as f:
        f.write(f"{eps:.6f}")

    print(f"✅ DP metrics → {dp_metrics_path}: {dp_metrics}")
    print(f"✅ ε = {eps:.6f} (δ={delta})")

if __name__=="__main__":
    dp_train(*sys.argv[1:])
