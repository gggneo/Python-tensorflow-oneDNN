import numpy as np
import onednn as dnnl

# Load preprocessed data
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val = np.load('X_val.npy')
y_val = np.load('y_val.npy')

# Define network architecture
input_shape = X_train.shape[1:]
num_classes = len(np.unique(y_train))
hidden_size = 256
net = dnnl.Sequential()
net.add(dnnl.Input(input_shape))
net.add(dnnl.Dense(hidden_size, activation='relu'))
net.add(dnnl.Dense(num_classes, activation='softmax'))

# Define optimization settings
batch_size = 32
num_epochs = 10
learning_rate = 0.001
optimizer = dnnl.Adam(learning_rate)

# Train the model
for epoch in range(num_epochs):
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]

    # Train on batches
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        # Forward pass
        y_pred = net(X_batch)

        # Compute loss
        loss = dnnl.losses.cross_entropy(y_pred, y_batch)

        # Backward pass
        grad = loss.grad()
        net.backward(grad)

        # Update parameters
        optimizer.update(net)

    # Evaluate on validation set
    y_pred_val = net(X_val)
    acc_val = dnnl.metrics.accuracy(y_pred_val, y_val)
    print(f'Epoch {epoch+1}/{num_epochs}, Val Acc: {acc_val:.4f}')
