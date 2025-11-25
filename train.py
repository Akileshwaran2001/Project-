import argparse
import torch
from torch.utils.data import DataLoader
from models.nbeats import NBeats
from scripts.dataloader import TimeSeriesDataset
import os

def train(args):
    dataset = TimeSeriesDataset(args.data_csv, backcast_length=args.backcast, forecast_length=args.forecast)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NBeats(input_dim=args.backcast, backcast_length=args.backcast, forecast_length=args.forecast,
                   nb_blocks_per_stack=args.blocks, hidden_dim=args.hidden, nb_layers=args.layers)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        print(f'Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(dataset):.6f}')
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/nbeats.pth')
    print('Saved checkpoint to checkpoints/nbeats.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_csv', default='data/synthetic_multivariate.csv')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--backcast', type=int, default=48)
    parser.add_argument('--forecast', type=int, default=24)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    args = parser.parse_args()
    train(args)
