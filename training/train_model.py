
def train_model(model, train_loader, val_loader, lr, weight_decay, epochs, trial=None):
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr[model], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    for epoch in range(epochs):
        # Your existing training loop code here
        model.train()
        running_loss = 0.0
        train_total = 0
        tcorrect = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Your augmentation logic here if needed
            # if epoch > 10:
            #     batch_X_np = batch_X.cpu().numpy()
            #     batch_X_aug = eeg_augment(batch_X_np)
            #     batch_X = torch.tensor(batch_X_aug, dtype=torch.float32).to(device)
                
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            tcorrect += (predicted == batch_y).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = tcorrect / train_total

        # Validation
        val_running_loss = 0.0
        model.eval()
        vcorrect = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)   
                val_running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                vcorrect += (predicted == batch_y).sum().item()

        val_loss = val_running_loss / len(val_loader)
        val_accuracy = vcorrect / val_total
        scheduler.step(val_loss)

        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break

        # Optuna pruning (only if trial is provided)
        if trial is not None:
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_val_acc
