import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.collections as mcoll
import cv2

def multicolored_lines(x, y, heatmap, title_name):
    fig, ax = plt.subplots()
    lc = colorline(x, y, heatmap, cmap='rainbow')
    plt.colorbar(lc)
    lc.set_linewidth(2)
    lc.set_alpha(0.8)
    plt.xlim(x.min(), x.max())
    plt.ylim(np.min(y), np.max(y))  # Adjust ylim based on y values
    plt.title(title_name)
    plt.grid(False)
    plt.show()

def colorline(x, y, heatmap, cmap='rainbow'):
    z = np.array(heatmap)
    points = np.column_stack((x, y))
    segments = np.stack([points[:-1], points[1:]], axis=1)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def compute_cam_1d_output(model, data, layer_name, N):
    """
    model: The Deep Learning model
    data : A input data. Data shape has to be (n,1,1)
    layer_name : The target layer for explanation
    N: signal length in seconds
    """
    # Setting the model to evaluation mode
    model.eval()
    
    # Extracting the target layer and output layer
    conv_outs = None
    grads = None

    def forward_hook(module, input, output):
        nonlocal conv_outs
        output.requires_grad_(True)
        conv_outs = output

    def backward_hook(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    # Register hooks
    target_layer = dict(model.named_modules())[layer_name]
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Convert the input data to a tensor and expand dimensions to match expected input
    inputs = torch.tensor(data, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    
    # Forward pass to get model output
    predictions = model(inputs)
    class_idx = torch.argmax(predictions[0])
    y_c = predictions[:, class_idx]
    # print(y_c, predictions)

    # Backward pass to get gradients
    model.zero_grad()
    y_c.backward(retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    # Ensure gradients are available
    if grads is None:
        raise RuntimeError("Gradients are not available for conv_outs. Ensure conv_outs has requires_grad=True.")
    
    grads = grads[0]

    # Print shapes for debugging
    # print("conv_outs shape:", conv_outs.shape)
    # print("grads shape:", grads.shape)
    
    # First, second and third derivative of output gradient
    first = torch.exp(y_c) * grads
    second = torch.exp(y_c) * torch.pow(grads, 2)
    third = torch.exp(y_c) * torch.pow(grads, 3)
    
    # print("first shape:", first.shape)
    # print("second shape:", second.shape)
    # print("third shape:", third.shape)

    # Compute saliency maps for the class_idx prediction
    global_sum = torch.sum(conv_outs[0].reshape(-1, first.shape[1]), dim=0)
    # print("global_sum shape:", global_sum.shape)
    alpha_num = second
    alpha_denom = second * 2.0 + third * global_sum.reshape((1, 1, -1))
    alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    weights = torch.maximum(first, torch.tensor(0.0))
    alpha_normalization_constant = torch.sum(torch.sum(alphas, dim=0), dim=0)
    alphas /= alpha_normalization_constant.reshape((1, 1, -1))
    alphas_thresholding = torch.where(weights > 0, alphas, torch.tensor(0.0))

    alpha_normalization_constant = torch.sum(torch.sum(alphas_thresholding, dim=0), dim=0)
    alpha_normalization_constant_processed = torch.where(alpha_normalization_constant != 0.0, alpha_normalization_constant,
                                                         torch.ones_like(alpha_normalization_constant))

    alphas /= alpha_normalization_constant_processed.reshape((1, 1, -1))
    deep_linearization_weights = torch.sum((weights * alphas).reshape(-1, first.shape[1]), dim=0)
    grad_CAM_map = torch.sum(deep_linearization_weights * conv_outs[0], dim=-1)
    
    # Normalization
    cam = torch.maximum(grad_CAM_map, torch.tensor(0.0))
    cam = cam / torch.max(cam)  
    
    # Turn result into a heatmap
    heatmap = []
    heatmap.append(cam.detach().numpy().tolist())
    big_heatmap = cv2.resize(np.array(heatmap), dsize=(data.shape[1], 500), interpolation=cv2.INTER_CUBIC)
    x = np.linspace(0, N, data.shape[1])
    # print(len(big_heatmap[0]))
    plt.style.use("seaborn-v0_8-whitegrid")
    multicolored_lines(x, np.array([i for i in data])[0], big_heatmap[0], f"GradCAM++ Visualization")
