"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy

max_nepochs = 12 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
display = 500  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).

train_data_path = './data/spec/aug/'
test_data_path = './data/spec/test/'

exp_path = './results/spec/'

sample_path = exp_path + 'samples'
checkpoint_path = exp_path + 'checkpoints'
restore = ''   # Model snapshot to restore from


lambda_g = 0.1    # Weight of the classification loss
gamma_decay = 0.5 # Gumbel-softmax temperature anneal rate
lambda_c = .001   # Weight of the conciseness loss

train_data = {
    'batch_size': 64,
    #'seed': 123,
    'datasets': [
        {
            'files': train_data_path + 'spec.train.text',
            'vocab_file': test_data_path + 'vocab',
            'data_name': ''
        },
        {
            'files': train_data_path + 'spec.train.labels',
            'data_type': 'int',
            'data_name': 'labels'
        }
    ],
    'name': 'train'
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = train_data_path + 'spec.dev.text'
val_data['datasets'][1]['files'] = train_data_path + 'spec.dev.labels'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = test_data_path + 'spec.test.text'
test_data['datasets'][1]['files'] = test_data_path + 'spec.test.labels'

model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 33,
        'max_decoding_length_infer': 32,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
