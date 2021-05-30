import argparse


from neurve.tsne.trainer import run_from_config


def main(config):
    run_from_config(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_charts", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--save_ckpt_freq", type=int)
    parser.add_argument("--eval_freq", type=int)
    parser.add_argument("--out_dim", type=int)
    parser.add_argument("--perplexity", type=float)
    parser.add_argument("--max_var2", type=float)
    parser.add_argument("--var2_tol", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--reg_loss_weight", type=float)
    parser.add_argument("--q_loss_weight", type=float)
    parser.add_argument("--dataset", type=str)

    args = parser.parse_args()
    main(vars(args))
