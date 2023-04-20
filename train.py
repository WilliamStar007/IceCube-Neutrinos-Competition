# Import
import os
import math
import random
from typing import List
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import argparse
import gc


def angular_dist_score(az_true, zen_true, az_pred, zen_pred):
    '''
    calculate the MAE of the angular distance between two directions.
    The two vectors are first converted to cartesian unit vectors,
    and then their scalar product is computed, which is equal to
    the cosine of the angle between the two vectors. The inverse 
    cosine (arccos) thereof is then the angle between the two input 
vectors

    Parameters:
    -----------

    az_true : float (or array thereof)
        true azimuth value(s) in radian
    zen_true : float (or array thereof)
        true zenith value(s) in radian
    az_pred : float (or array thereof)
        predicted azimuth value(s) in radian
    zen_pred : float (or array thereof)
        predicted zenith value(s) in radian

    Returns:
    --------

    dist : float
        mean over the angular distance(s) in radian
    '''

    if not (np.all(np.isfinite(az_true)) and
            np.all(np.isfinite(zen_true)) and
            np.all(np.isfinite(az_pred)) and
            np.all(np.isfinite(zen_pred))):
        raise ValueError("All arguments must be finite")

    # pre-compute all sine and cosine values
    sa1 = np.sin(az_true)
    ca1 = np.cos(az_true)
    sz1 = np.sin(zen_true)
    cz1 = np.cos(zen_true)

    sa2 = np.sin(az_pred)
    ca2 = np.cos(az_pred)
    sz2 = np.sin(zen_pred)
    cz2 = np.cos(zen_pred)

    # scalar product of the two cartesian vectors (x = sz*ca, y = sz*sa, z = cz)
    scalar_prod = sz1 * sz2 * (ca1 * ca2 + sa1 * sa2) + (cz1 * cz2)

    # scalar product of two unit vectors is always between -1 and 1, this is against nummerical instability
    # that might otherwise occure from the finite precision of the sine and cosine functions
    scalar_prod = np.clip(scalar_prod, -1, 1)

    # convert back to an angle (in radian)
    return np.average(np.abs(np.arccos(scalar_prod)))


def pred_to_angle(pred, bin_num, angle_bin_vector, epsilon=1e-8):
    # convert prediction to vector
    pred_vector = (pred.reshape((-1, bin_num * bin_num, 1))
                   * angle_bin_vector).sum(axis=1)

    # normalize
    pred_vector_norm = np.sqrt((pred_vector ** 2).sum(axis=1))
    mask = pred_vector_norm < epsilon
    pred_vector_norm[mask] = 1

    # assign <1, 0, 0> to very small vectors (badly predicted)
    pred_vector /= pred_vector_norm.reshape((-1, 1))
    pred_vector[mask] = np.array([1., 0., 0.])

    # convert to angle
    azimuth = np.arctan2(pred_vector[:, 1], pred_vector[:, 0])
    azimuth[azimuth < 0] += 2 * np.pi
    zenith = np.arccos(pred_vector[:, 2])

    return azimuth, zenith


def y_to_angle_code(batch_y, azimuth_edges, zenith_edges, bin_num):
    azimuth_code = (
            batch_y[:, 0] > azimuth_edges[1:].reshape((-1, 1))).sum(axis=0)
    zenith_code = (
            batch_y[:, 1] > zenith_edges[1:].reshape((-1, 1))).sum(axis=0)
    angle_code = bin_num * azimuth_code + zenith_code

    return angle_code


def normalize_data(data):
    data[:, :, 0] /= 1000  # time
    data[:, :, 1] /= 300  # charge
    data[:, :, 3:] /= 600  # space

    return data


def prep_validation_data(validation_ids: List[int], file_format: str):
    print("Processing Validation Data...")

    # Prepare fixed Validation Set
    val_x = None
    val_y = None

    # Summary
    print(validation_ids)

    # Loop
    for batch_id in tqdm(validation_ids):
        val_data_file = np.load(file_format.format(batch_id=batch_id))

        if val_x is None:
            val_x = val_data_file["x"][:, :, [0, 1, 2, 3, 4, 5]]
            val_y = val_data_file["y"]
        else:
            val_x = np.append(val_x, val_data_file["x"][:, :, [0, 1, 2, 3, 4, 5]], axis=0)
            val_y = np.append(val_y, val_data_file["y"], axis=0)

        val_data_file.close()
        del val_data_file
        _ = gc.collect()

    # Normalize Data
    val_x = normalize_data(val_x)

    # Shape Summary
    print(val_x.shape)

    return val_x, val_y


def prep_training_data(train_ids: List[int], file_format: str, azimuth_edges, zenith_edges, bin_num):
    # Placeholders
    train_x = None
    train_y = None
    print(train_ids)

    # Loop
    for batch_id in train_ids:
        train_data_file = np.load(file_format.format(batch_id=batch_id))

        if train_x is None:
            train_x = train_data_file["x"][:, :, [0, 1, 2, 3, 4, 5]]
            train_y = train_data_file["y"]
        else:
            train_x = np.append(
                train_x, train_data_file["x"][:, :, [0, 1, 2, 3, 4, 5]], axis=0)
            train_y = np.append(train_y, train_data_file["y"], axis=0)

        train_data_file.close()
        del train_data_file
        _ = gc.collect()

    # Normalize data
    train_x = normalize_data(train_x)

    # Output Encoding
    trn_y_anglecode = y_to_angle_code(train_y, azimuth_edges, zenith_edges, bin_num)

    return train_x, trn_y_anglecode


# Model
def create_model(strategy, pulse_count, feature_count, lstm_units, bin_num, learning_rate):
    with strategy.scope():
        inputs = tf.keras.layers.Input((pulse_count, feature_count))

        x = tf.keras.layers.Masking(mask_value=0., input_shape=(
            pulse_count, feature_count))(inputs)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(lstm_units, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(lstm_units, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(lstm_units))(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)

        outputs = tf.keras.layers.Dense(bin_num ** 2, activation='softmax')(x)

        # Finalize Model
        model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])

        # Show Model Summary
        model.summary()

        return model


def set_seeds(seed):
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(arg):
    set_seeds(42)

    # Training
    validation_files_amount = 2
    data_new_load_interval = 2
    epochs = 50
    batch_size = 4096
    learning_rate = 0.0006
    verbose = 0

    # Training Batches
    train_batch_id_min = 250
    train_batch_id_max = 399
    train_batch_ids = [*range(train_batch_id_min, train_batch_id_max + 1)]
    np.random.shuffle(train_batch_ids)
    print(train_batch_ids)

    # Model Parameters
    pulse_count = 96
    feature_count = 6
    lstm_units = 192
    bin_num = 36

    id = "{}_{}_{}_{}_{}".format(train_batch_id_min, train_batch_id_max, bin_num, batch_size, learning_rate)
    train_log_dir = 'logs/' + id
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Data
    base_dir = "/root/processed_data/"
    file_format = base_dir + 'pp_mpc96_n7_batch_{batch_id:d}.npz'

    tpu = None
    strategy = None
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.MirroredStrategy()

    model_save_path = 'gpu_pp96_n{}_bin{}_batch{}_epoch{}.h5'

    validation_ids = train_batch_ids[:validation_files_amount]
    train_batch_ids = train_batch_ids[validation_files_amount:]

    # Create Azimuth Edges
    azimuth_edges = np.linspace(0, 2 * np.pi, bin_num + 1)
    print(azimuth_edges)

    # Create Zenith Edges
    zenith_edges = []
    zenith_edges.append(0)
    for bin_idx in range(1, bin_num):
        zenith_edges.append(
            np.arccos(np.cos(zenith_edges[-1]) - 2 / (bin_num)))
    zenith_edges.append(np.pi)
    zenith_edges = np.array(zenith_edges)
    print(zenith_edges)

    angle_bin_zenith0 = np.tile(zenith_edges[:-1], bin_num)
    angle_bin_zenith1 = np.tile(zenith_edges[1:], bin_num)
    angle_bin_azimuth0 = np.repeat(azimuth_edges[:-1], bin_num)
    angle_bin_azimuth1 = np.repeat(azimuth_edges[1:], bin_num)

    angle_bin_area = (angle_bin_azimuth1 - angle_bin_azimuth0) * \
                     (np.cos(angle_bin_zenith0) - np.cos(angle_bin_zenith1))
    angle_bin_vector_sum_x = (np.sin(angle_bin_azimuth1) -
                              np.sin(angle_bin_azimuth0)) * (
                                     (angle_bin_zenith1 - angle_bin_zenith0) / 2 -
                                     (np.sin(2 * angle_bin_zenith1) -
                                      np.sin(2 * angle_bin_zenith0)) / 4)
    angle_bin_vector_sum_y = (np.cos(angle_bin_azimuth0) -
                              np.cos(angle_bin_azimuth1)) * (
                                     (angle_bin_zenith1 - angle_bin_zenith0) / 2 -
                                     (np.sin(2 * angle_bin_zenith1) -
                                      np.sin(2 * angle_bin_zenith0)) / 4)
    angle_bin_vector_sum_z = (angle_bin_azimuth1 - angle_bin_azimuth0) * \
                             ((np.cos(2 * angle_bin_zenith0) -
                               np.cos(2 * angle_bin_zenith1)) / 4)

    angle_bin_vector_mean_x = angle_bin_vector_sum_x / angle_bin_area
    angle_bin_vector_mean_y = angle_bin_vector_sum_y / angle_bin_area
    angle_bin_vector_mean_z = angle_bin_vector_sum_z / angle_bin_area

    angle_bin_vector = np.zeros((1, bin_num * bin_num, 3))
    angle_bin_vector[:, :, 0] = angle_bin_vector_mean_x
    angle_bin_vector[:, :, 1] = angle_bin_vector_mean_y
    angle_bin_vector[:, :, 2] = angle_bin_vector_mean_z

    # Create Model
    model = create_model(strategy, pulse_count, feature_count,
                         lstm_units, bin_num, learning_rate)

    start_epoch = 0
    if args.resume:
        start_epoch = args.resume
        if start_epoch < 1:
            print("incorrect argument. cannot load epoch -1")
            exit(1)
        model.load_weights(model_save_path.format(
            feature_count, bin_num, batch_size, start_epoch - 1))

    # Epoch Loop
    for e in range(start_epoch, epochs):
        _ = gc.collect()
        print(f'=========== EPOCH: {e}')

        np.random.shuffle(train_batch_ids)

        start_batch = 0
        sessions = math.ceil((len(train_batch_ids)) / data_new_load_interval)

        session_ids = []
        for i in range(sessions):
            end_batch = min(
                start_batch + data_new_load_interval, train_batch_id_max)
            session_ids.append(train_batch_ids[start_batch:end_batch])
            start_batch = end_batch

        losses = []
        accuracy = []

        for s in range(sessions):
            print(f'        ======= session: {s}')
            trn_x, trn_y_anglecode = prep_training_data(
                session_ids[s], file_format, azimuth_edges, zenith_edges, bin_num)
            # Number of batches
            batch_count = trn_x.shape[0] // batch_size

            # Random Shuffle each epoch
            indices = np.arange(trn_x.shape[0])
            np.random.shuffle(indices)
            trn_x = trn_x[indices]
            trn_y_anglecode = trn_y_anglecode[indices]

            # Batch Loop
            for batch_index in tqdm(range(batch_count), total=batch_count):
                b_train_x = trn_x[batch_index *
                                  batch_size: batch_index * batch_size + batch_size, :]
                b_train_y = trn_y_anglecode[batch_index *
                                            batch_size: batch_index * batch_size + batch_size]

                metrics = model.train_on_batch(b_train_x, b_train_y)

                losses.append(metrics[0])
                accuracy.append(metrics[1])

            del trn_x, trn_y_anglecode
            gc.collect()

        # Save Model
        model.save(model_save_path.format(
            feature_count, bin_num, batch_size, e))

        # Metrics
        # val_x, val_y = prep_validation_data(validation_ids, file_format)
        # valid_pred = model.predict(
        #     val_x, batch_size=batch_size, verbose=verbose)
        #
        # valid_pred_azimuth, valid_pred_zenith = pred_to_angle(
        #     valid_pred, bin_num, angle_bin_vector)
        # mae = angular_dist_score(
        #     val_y[:, 0], val_y[:, 1], valid_pred_azimuth, valid_pred_zenith)
        print(f'Total Train Loss: {np.mean(losses):.4f}   Accuracy: {np.mean(accuracy):.4f}')

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', np.mean(losses), step=e)
            tf.summary.scalar('accuracy', np.mean(accuracy), step=e)
            # tf.summary.scalar('MAE', mae, step=e)

        gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="deep ice LSTM training")
    parser.add_argument('--resume', type=int, default=0,
                        required=False, help='which epoch to resume')
    args = parser.parse_args()

    main(args)
