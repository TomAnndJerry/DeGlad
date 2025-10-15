import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='Deglad')
    parser.add_argument('--method', type=str, default='Deglad')
    parser.add_argument('--dataset', type=str, default='BZR')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=9999)
    parser.add_argument('--lr_decay_step_size', type=int, default=9999)
    parser.add_argument('--lr_decay_factor', type=float, default=0.9)

    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--num_trials', type=int, default=5)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--encoder_layers', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--threshold', type=str, default=None)
    parser.add_argument('--pooling', type=str, default='add', choices=['add', 'max', 'mean'])
    parser.add_argument('--readout', type=str, default='concat', choices=['concat', 'add', 'last'])

    parser.add_argument('--env_mean_kl_alpha', type=int, default=1)
    parser.add_argument('--env_kl_alpha', type=int, default=1)
    parser.add_argument('--core_kl_alpha', type=int, default=1)
    parser.add_argument('--noise_alpha', type=float, default=0.1)

    parser.add_argument('--extractor_model', type=str, default='gin', choices=['mlp', 'gin'])
    parser.add_argument('--extractor_layers', type=int, default=5)
    parser.add_argument('--extractor_hidden_dim', type=int, default=8)
    parser.add_argument('--extractor_readout', type=str, default='add', choices=['concat', 'add', 'last'])

    parser.add_argument('--cga_net_lr', type=float, default=0.0001)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--emb_dim', type=int, default=300)

    return parser.parse_args()