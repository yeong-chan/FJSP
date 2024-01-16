import argparse


def get_cfg():
    map_name = '3s5z'
    learning_rate = 1.3e-4
    GNN = 'FastGTN'

    hidden_size_obs = 45  # GAT 해당(action 및 node representation의 hidden_size)
    hidden_size_comm = 64
    hidden_size_Q = 78  # GAT 해당
    hidden_size_meta_path = 48  # GAT 해당
    n_representation_obs = 42  # GAT 해당
    n_representation_comm = 72


    buffer_size = 150000
    batch_size = 32
    gamma = 0.99

    n_multi_head = 1
    dropout = 0.6
    num_episode = 1000000
    train_start = 10
    epsilon = 1
    min_epsilon = 0.05
    anneal_steps = 50000

    parser = argparse.ArgumentParser(description="")

    parser.add_argument("--map_name", type=str, default="3s5z", help="map name")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="learning rate")
    parser.add_argument("--GNN", type=str, default="FastGTN", help="learning rate")

    parser.add_argument("--hidden_size_obs", type=float, default=45, help="size of hidden layer in observation stage")
    parser.add_argument("--hidden_size_comm", type=float, default=64, help="size of hidden layer in communicaton stage / GTN only")

    parser.add_argument("--gamma", type=float, default=0.99, help="discount ratio")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--eps_clip", type=float, default=0.15, help="clipping paramter")
    parser.add_argument("--K_epoch", type=int, default=2, help="optimization epoch")
    parser.add_argument("--T_horizon", type=int, default=50, help="running horizon")
    parser.add_argument("--w_delay", type=float, default=1.0, help="weight for minimizing delays")
    parser.add_argument("--w_move", type=float, default=0.5, help="weight for minimizing the number of ship movements")
    parser.add_argument("--w_priority", type=float, default=0.5, help="weight for maximizing the efficiency")
    return parser.parse_args()