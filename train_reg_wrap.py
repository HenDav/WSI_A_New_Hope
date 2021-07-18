import argparse
import os
import smtplib, ssl
import subprocess
import traceback, io, sys

parser = argparse.ArgumentParser(description='WSI_REG Training of PathNet Project')
parser.add_argument('-tf', '--test_fold', default=1, type=int, help='fold to be as TEST FOLD')
parser.add_argument('-e', '--epochs', default=5, type=int, help='Epochs to run')
parser.add_argument('-ex', '--experiment', type=int, default=0, help='Continue train of this experiment')
parser.add_argument('-fe', '--from_epoch', type=int, default=0, help='Continue train from epoch')
parser.add_argument('-d', dest='dx', action='store_true', help='Use ONLY DX cut slides')
parser.add_argument('-ds', '--dataset', type=str, default='TCGA', help='DataSet to use')
parser.add_argument('-time', dest='time', action='store_true', help='save train timing data ?')
parser.add_argument('-tar', '--target', default='ER', type=str, help='label: Her2/ER/PR/EGFR/PDL1')
parser.add_argument('--n_patches_test', default=1, type=int, help='# of patches at test time')
parser.add_argument('--n_patches_train', default=10, type=int, help='# of patches at train time')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--weight_decay', default=5e-5, type=float, help='L2 penalty')
parser.add_argument('-balsam', '--balanced_sampling', dest='balanced_sampling', action='store_true', help='balanced_sampling')
parser.add_argument('--transform_type', default='rvf', type=str, help='none / flip / wcfrs (weak color+flip+rotate+scale)')
parser.add_argument('--batch_size', default=18, type=int, help='size of batch')
parser.add_argument('--model', default='PreActResNets.PreActResNet50_Ron()', type=str, help='net to use')
parser.add_argument('--bootstrap', action='store_true', help='use bootstrap to estimate test AUC error')
parser.add_argument('--eval_rate', type=int, default=5, help='Evaluate validation set every # epochs')
parser.add_argument('--c_param', default=0.1, type=float, help='color jitter parameter')
parser.add_argument('-im', dest='images', action='store_true', help='save data images?')
parser.add_argument('--mag', type=int, default=10, help='desired magnification of patches')
parser.add_argument('--loan', action='store_true', help='Localized Annotation for strongly supervised training')
parser.add_argument('--er_eq_pr', action='store_true', help='while training, take only er=pr examples')
args = parser.parse_args()

try:
    sub_command = ['python', 'train_reg.py',
                    '--test_fold', str(args.test_fold),
                    '--epochs', str(args.epochs),
                    '--experiment', str(args.experiment),
                    '--from_epoch', str(args.from_epoch),
                    #'--dx', str(args.dx),
                    '--dataset', args.dataset,
                    #'-time', str(args.time),
                    '--target', args.target,
                    '--n_patches_test', str(args.n_patches_test),
                    '--n_patches_train', str(args.n_patches_train),
                    '--lr', str(args.lr),
                    '--weight_decay', str(args.weight_decay),
                    #'--balanced_sampling', str(args.balanced_sampling),
                    '--transform_type', args.transform_type,
                    '--batch_size', str(args.batch_size),
                    '--model', args.model,
                    #'--bootstrap', str(args.bootstrap),
                    '--eval_rate', str(args.eval_rate),
                    '--c_param', str(args.c_param),
                    #'-im', str(args.images),
                    '--mag', str(args.mag),
                    #'--loan', str(args.loan),
                    #'--er_eq_pr', str(args.er_eq_pr),
                    ]
    if args.dx:
        sub_command.append('-d')
    if args.time:
        sub_command.append('-time')
    if args.balanced_sampling:
        sub_command.append('--balanced_sampling')
    if args.bootstrap:
        sub_command.append('--bootstrap')
    if args.images:
        sub_command.append('-im')
    if args.loan:
        sub_command.append('--loan')
    if args.er_eq_pr:
        sub_command.append('--er_eq_pr')

    subprocess.run(sub_command)
except:
    # failed run, send email if possible
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.stdout)
    fp = io.StringIO()
    traceback.print_exc(file=fp)
    message1 = fp.getvalue()
    print(message1)

    if os.path.isfile('mail_cfg.txt'):
        with open("mail_cfg.txt", "r") as f:
            text = f.readlines()
            receiver_email = text[0][:-1]
            password = text[1]

        port = 465  # For SSL
        sender_email = "gipmed.python@gmail.com"

        # Create a secure SSL context
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, 'Subject: failed experiment!' + '\n\n' + message1)
            server.quit()
            print('email sent to ' + receiver_email)