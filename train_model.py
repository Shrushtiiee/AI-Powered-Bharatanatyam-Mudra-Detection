
import os
import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models

# Reproducibility
SEED = 42
tf.keras.utils.set_random_seed(SEED)


def build_model(num_classes, img_size):
    """
    Build a transfer-learning model using MobileNetV2.
    Input images must contain RGB pixel values from 0 to 255.
    """

    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet"
    )

    base.trainable = False

    inputs = tf.keras.Input(
        shape=(img_size, img_size, 3)
    )

    # Preprocessing is included inside the model.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(
        inputs
    )

    x = base(x, training=False)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.2)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax"
    )(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model, base


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        default="sample_dataset",
        help="Folder containing one folder per mudra"
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=15
    )

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5
    )

    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit(
            f"Dataset folder not found: {args.data_dir}"
        )

    # Load training data
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="int"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False
    )

    class_names = train_ds.class_names

    print("\nDetected mudra classes:")

    for index, name in enumerate(class_names):
        print(index, ":", name)

    if len(class_names) < 2:
        raise SystemExit(
            "At least two class folders are required."
        )

    # Save the class mapping used by the model.
    with open("class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Moderate augmentation
    augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05
        )
    ])

    def augment_images(images, labels):
        images = augmentation(
            images,
            training=True
        )
        return images, labels

    train_ds = train_ds.map(
        augment_images,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_ds = train_ds.prefetch(
        tf.data.AUTOTUNE
    )

    val_ds = val_ds.prefetch(
        tf.data.AUTOTUNE
    )

    # Build model
    model, base = build_model(
        len(class_names),
        args.img_size
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            "best_mudra_model.keras",
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # Stage 1: train classifier head
    print("\nStage 1: Training classifier")

    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks
    )

    histories = [history1]

    # Stage 2: fine-tune upper layers
    if args.fine_tune_epochs > 0:

        print("\nStage 2: Fine-tuning MobileNetV2")

        base.trainable = True

        # Freeze earlier layers
        for layer in base.layers[:-20]:
            layer.trainable = False

        # Keep BatchNormalization layers frozen
        for layer in base.layers:
            if isinstance(
                layer,
                layers.BatchNormalization
            ):
                layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.00001
            ),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks
        )

        histories.append(history2)

    # Load the best checkpoint across both stages
    model = tf.keras.models.load_model(
        "best_mudra_model.keras"
    )

    # Save final model
    model.save("mudra_model.keras")

    # Save class names again
    with open("class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    # Combine training history
    accuracy = []
    val_accuracy = []
    loss = []
    val_loss = []

    for history in histories:
        accuracy.extend(
            history.history["accuracy"]
        )

        val_accuracy.extend(
            history.history["val_accuracy"]
        )

        loss.extend(
            history.history["loss"]
        )

        val_loss.extend(
            history.history["val_loss"]
        )

    # Plot training curves
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4)
    )

    axes[0].plot(
        accuracy,
        label="Training accuracy"
    )

    axes[0].plot(
        val_accuracy,
        label="Validation accuracy"
    )

    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(
        loss,
        label="Training loss"
    )

    axes[1].plot(
        val_loss,
        label="Validation loss"
    )

    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig("training_history.png")
    plt.close(fig)

    # Final evaluation
    val_loss_value, val_accuracy_value = (
        model.evaluate(val_ds, verbose=0)
    )

    print("\nTraining completed!")
    print(
        f"Validation accuracy: "
        f"{val_accuracy_value * 100:.2f}%"
    )

    print("Saved: mudra_model.keras")
    print("Saved: class_names.json")
    print("Saved: training_history.png")


if __name__ == "__main__":
    main()