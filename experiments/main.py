import random
import os
import numpy as np
import torch
from tqdm import tqdm
from results_json import ResultsJSON
from difflogic.compiled_model import CompiledLogicNet
torch.set_num_threads(1)
from utils.accuracy_metrics import evaluate_model
from utils.args import parse_args
from utils.loading import load_dataset, load_n
from utils.model_selection import get_model
from training.base_train import train, eval, packbits_eval
import torch.nn.functional as F

BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64
}

IMPL_TO_DEVICE = {
    'cuda': 'cuda',
    'python': 'cpu'
              ''
}

if __name__ == '__main__':

    args = parse_args()
    assert args.num_iterations % args.eval_freq == 0, (
        f'iteration count ({args.num_iterations}) has to be divisible by evaluation frequency ({args.eval_freq})'
    )

    # try:
    #     import difflogic_cuda
    # except ImportError:
    #     warnings.warn('failed to import difflogic_cuda. no cuda features will be available', ImportWarning)

    if args.experiment_id is not None:
        assert 520_000 <= args.experiment_id < 530_000, args.experiment_id
        results = ResultsJSON(eid=args.experiment_id, path='./results/')
        results.store_args(args)

    device = IMPL_TO_DEVICE[args.implementation]

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    train_loader, validation_loader, test_loader = load_dataset(args)
    model, loss_fn, optim = get_model(args)

    best_acc = 0
    if test_loader is not None:
        for i, (x, y) in tqdm(
                enumerate(load_n(train_loader, args.num_iterations)),
                desc='iteration',
                total=args.num_iterations,
        ):
            x = x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)
            y = y.to(device)

            loss = train(model, x, y, loss_fn, optim)

            if (i+1) % args.eval_freq == 0:
                if args.extensive_eval:
                    train_accuracy_train_mode = eval(model, train_loader, mode=True, device=device)
                    valid_accuracy_eval_mode = eval(model, validation_loader, mode=False, device=device)
                    valid_accuracy_train_mode = eval(model, validation_loader, mode=True, device=device)
                else:
                    train_accuracy_train_mode = -1
                    valid_accuracy_eval_mode = -1
                    valid_accuracy_train_mode = -1
                train_accuracy_eval_mode = eval(model, train_loader, mode=False, device=device)
                test_accuracy_eval_mode = eval(model, test_loader, mode=False, device=device)
                test_accuracy_train_mode = eval(model, test_loader, mode=True, device=device)

                r = {
                    'train_acc_eval_mode': train_accuracy_eval_mode,
                    'train_acc_train_mode': train_accuracy_train_mode,
                    'valid_acc_eval_mode': valid_accuracy_eval_mode,
                    'valid_acc_train_mode': valid_accuracy_train_mode,
                    'test_acc_eval_mode': test_accuracy_eval_mode,
                    'test_acc_train_mode': test_accuracy_train_mode,
                }

                if args.packbits_eval:
                    r['train_acc_eval'] = packbits_eval(model, train_loader, device=device)
                    r['valid_acc_eval'] = packbits_eval(model, train_loader, device=device)
                    r['test_acc_eval'] = packbits_eval(model, test_loader, device=device)

                if args.experiment_id is not None:
                    results.store_results(r)
                else:
                    print(r)

                if valid_accuracy_eval_mode > best_acc:
                    best_acc = valid_accuracy_eval_mode
                    if args.experiment_id is not None:
                        results.store_final_results(r)
                    else:
                        print('IS THE BEST UNTIL NOW.')

                if args.experiment_id is not None:
                    results.save()
    else:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        data = train_loader.to(device)
        #data.x = data.x.to(BITS_TO_TORCH_FLOATING_POINT_TYPE[args.training_bit_count]).to(device)

        for epoch in range(10000):
            optimizer.zero_grad()
            out = model(data)
            if args.architecture == 'gcn_cora_baseline':
                loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            else:
                loss = loss_fn(out[data.train_mask], data.y[data.train_mask])

            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Gradient for {name} is None")()
            optimizer.step()
            if epoch % 100 == 0:
                print(epoch, loss.item())
                val_metrics = evaluate_model(model, data, data.test_mask, 7)
                print(val_metrics["accuracy"], val_metrics['confusion_matrix'])
    ####################################################################################################################

    if args.compile_model:
        print('\n' + '='*80)
        print(' Converting the model to C code and compiling it...')
        print('='*80)

        for opt_level in range(4):

            for num_bits in [
                # 8,
                # 16,
                # 32,
                64
            ]:
                os.makedirs('lib', exist_ok=True)
                save_lib_path = 'lib/{:08d}_{}.so'.format(
                    args.experiment_id if args.experiment_id is not None else 0, num_bits
                )

                compiled_model = CompiledLogicNet(
                    model=model,
                    num_bits=num_bits,
                    cpu_compiler='gcc',
                    # cpu_compiler='clang',
                    verbose=True,
                )

                compiled_model.compile(
                    opt_level=1 if args.num_layers * args.num_neurons < 50_000 else 0,
                    save_lib_path=save_lib_path,
                    verbose=True
                )

                correct, total = 0, 0
                with torch.no_grad():
                    for (data, labels) in torch.utils.data.DataLoader(test_loader.dataset, batch_size=int(1e6), shuffle=False):
                        data = torch.nn.Flatten()(data).bool().numpy()

                        output = compiled_model(data, verbose=True)

                        correct += (output.argmax(-1) == labels).float().sum()
                        total += output.shape[0]

                acc3 = correct / total
                print('COMPILED MODEL', num_bits, acc3)


