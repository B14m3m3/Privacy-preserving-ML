import os
import argparse
import pandas as pd
import util
import model_stats

from players import Player, Host
from classifiers import LogisticClassifier


def main_tester_collab():
    players = [3]
    C = [50]
    epsilon = [i*0.1 for i in range(1, 11)]
    LR = [0.3]
    batch_size = [20000]
    epochs = [50]
    order = 1 #0 is no random, 1 is L1, 2 is L2

    print('Loading Data')
    data, label = util.fetch_mnist_data()
    data_train, label_train, data_val, label_val = util.train_and_validation_split(data, label, n_samples=70000)
    print('Data Loaded')

    results = []
    for p in players:
        data_train_split, label_train_split = util.split_train_data(data_train, label_train, p)
        for c in C:
            for e in epsilon:
                for lr in LR:
                    for b in batch_size:
                        for ep in epochs:
                            config = {'batch_size': b, 'delta': 10**-3, 'epsilon': e, 'C': c, 'r': len(data_train_split[0]) // b, 'LR':lr, 'players':p, 'epochs': ep, 'order': order}
                            trained_model = reg_train_collab(data_train_split, label_train_split, config)
                            precision = trained_model.evaluate(data_val, label_val)
                            results.append([p, c, e, lr, b, ep, config['r'], config['order'], precision])

                            model_stats.confusion_mat('collab_train_e-{0}_ep-{1}.csv'.format(e, ep), trained_model.model, data_val, label_val)
                            model_stats.visualize_model('collab_train_e-{0}_ep-{1}.png'.format(e, order), trained_model.model)
    res = pd.DataFrame(results, columns=['Players', 'C', 'epsilon', 'lr', 'batch_size', 'epochs', 'rounds', 'order', 'precision'])
    print(res)
    model_stats.export_dataframe('collab_GD.csv', res)

def reg_train_collab(data, label, config):
    peer_list = [Player(LogisticClassifier(config['LR'], config['C'], config['order']), data[i], label[i], config) for i in range(config['players'])]
    trained_peers = collaborative_gradient_descent(peer_list, config)
    #since all peers are trained with the same gradients and initialized weights, just return one of them
    return trained_peers[0]

def collaborative_gradient_descent(peer_list, config):
    print("running collaborative gradient descent")
    host_1 = Host(config)
    host_2 = Host(config)
    for _ in range(config['epochs']):
        for i in range(config['r']):
            for peer in peer_list:
                rand_gradient, randomness = peer.calculate_gradient(i)
                host_1.feed_gradient(rand_gradient)
                host_2.feed_gradient(randomness)
            gradient_sum = host_1.calculate_gradient()
            randomness_sum = host_2.calculate_gradient()
            trusted_gradient = trusted_gradient_calculation(gradient_sum, randomness_sum, config)
            for peer in peer_list:
                peer.update_gradient(trusted_gradient)
    return peer_list

def trusted_gradient_calculation(g1, g2, config):
    assert len(g1) == len(g2)
    m = config['batch_size']
    C = config['C']
    epsilon = config['epsilon']
    delta = config['delta']
    order = config['order']
    size = len(g1[0])
    b = util.calculate_b(C, epsilon, delta, size, order)
    rand = [util.randomness(size, b, order) for i in range(len(g1))]
    return [util.smod(arr-g2[idx], m*C) + rand[idx] for idx, arr in enumerate(g1)]

def reg_train():
    print('Loading Data')
    data, label = util.fetch_mnist_data()
    data_train, label_train, data_val, label_val = util.train_and_validation_split(data, label, n_samples=70000)
    print('Data Loaded')

    C = [50]
    epsilon = [1]#[i*0.02 for i in range(1, 11)]
    LR = [0.3]
    batch_size = [60000]
    epochs = [50]
    order = 0 #0 is no random, 1 is L1, 2 is L2

    trained_peer = None
    results = []
    for c in C:
        for e in epsilon:
            for b in batch_size:
                for ep in epochs:
                    for lr in LR:

                        config = {'batch_size': b, 'delta': 10**-3, 'epsilon': e, 'C': c, 'r': len(data_train) // b, 'LR':lr, 'players':1, 'epochs': ep, 'order': order}
                        peer = Player(LogisticClassifier(config['LR'], config['C'], config['order']), data_train, label_train, config)
                        trained_peer = single_gradient_descent(peer, config)
                        precision = trained_peer.evaluate(data_val, label_val)
                        results.append([c, e, lr, b, ep, config['r'], config['order'], precision])

                        model_stats.confusion_mat('single_train_e-{0}_ep-{1}.csv'.format(e, ep), trained_peer.model, data_val, label_val)
                        model_stats.visualize_model('single_train_e-{0}_ep-{1}.png'.format(e, order), trained_peer.model)
    res = pd.DataFrame(results, columns=['C', 'epsilon', 'lr', 'batch_size', 'epochs', 'rounds', 'order', 'precision'])
    print(res)
    model_stats.export_dataframe('PGD.csv', res)


def single_gradient_descent(peer, config):
    C = config['C']
    epsilon = config['epsilon']
    delta = config['delta']
    order = config['order']

    for _ in range(config['epochs']):
        for i in range(config['r']):
            rand_gradient, randomness = peer.calculate_gradient(i)
            gradient = [arr-randomness[idx] for idx, arr in enumerate(rand_gradient)]
            size = len(gradient[0])
            b = util.calculate_b(C, epsilon, delta, size, order)
            rand = [util.randomness(size, b, order) for _ in range(len(gradient))]
            peer.update_gradient([gradient[j] + rand[j] for j in range(len(gradient))])
    return peer

if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    parser = argparse.ArgumentParser()
    parser.add_argument('-reg', default=False, help='reg', dest='reg', action='store_true')
    parser.add_argument('-reg_collab', default=False, help='reg collab', dest='reg_collab', action='store_true')
    parser.add_argument('-test', default=False, help='test', dest='test', action='store_true')
    if not os.path.exists('results'):
        print('create results folder')
        os.mkdir('results')

    args = parser.parse_args()
    if args.reg:
        reg_train()
    if args.reg_collab:
        main_tester_collab()
    if args.test:
        print("test")
