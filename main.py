from dataset import generate_data_loader
from models.textbert_model import TextBertModel
from trainer import Trainer
import argparse

parser = argparse.ArgumentParser(description='Train a TextBert model')
parser.add_argument('--data_path', type=str, default='data', help='Path to the data directory')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--max_length', type=int, default=512, help='Maximum length of the input sequence')
parser.add_argument('--bert_type', type=str, default='huawei-noah/TinyBERT_General_4L_312D', help='Type of pre-trained BERT model')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate for training')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
args = parser.parse_args()

def main(args):
    train_loader, val_loader, test_loader = generate_data_loader(args.data_path, args.batch_size, args.max_length, args.bert_type)
    model = TextBertModel(args.bert_type)
    trainer = Trainer(model, train_loader, val_loader, test_loader, args.epochs, args.lr, args.device)
    trainer.train()

if __name__ == '__main__':
    main(args)