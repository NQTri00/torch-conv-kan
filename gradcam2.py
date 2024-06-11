import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def target_category_loss(x, category_index, nb_classes):
    return torch.mul(x, F.one_hot(category_index, nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (torch.sqrt(torch.mean(torch.square(x))) + 1e-5)

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = []
        self.activations = []

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        # Gradients are computed in reverse order
        self.gradients = [grad_output[0]] + self.gradients

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)


class BaseCAM:
    def __init__(self, model, target_layer, use_cuda=False):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.activations_and_grads = ActivationsAndGradients(self.model, target_layer)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        # print(output.size())
        return output[target_category]

    def __call__(self, input_tensor, target_category=None):
        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        if target_category is None:
            output = output.squeeze()
            target_category = np.argmax(output.cpu().data.numpy())
            # print(output)
            # print(target_category)
        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()[0, :]
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()[0, :]
        #weights = np.mean(grads, axis=(0))
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
             cam += w * activations[i, :]
        # cam = activations.T.dot(weights)
        # cam = activations.dot(weights)
        # cam = activations.dot(weights)
        # print(input_tensor.shape[1])
        # print(cam.shape)
        # x = np.arange(0, 247, 1)
        # plt.plot(x, cam.reshape(-1, 1))
        # sns.set()
        # ax = sns.heatmap(cam.reshape(-1, 1).T)
        #cam = cv2.resize(cam, input_tensor.shape[1:][::-1])
        #cam = resize_1d(cam, (input_tensor.shape[2]))
        cam = np.interp(np.linspace(0, cam.shape[0], input_tensor.shape[2]), np.linspace(0, cam.shape[0], cam.shape[0]), cam)   #Change it to the interpolation algorithm that numpy comes with.
        #cam = np.maximum(cam, 0)
        # cam = np.expand_dims(cam, axis=1)
        # ax = sns.heatmap(cam)
        # plt.show()
        # cam = cam - np.min(cam)
        # cam = cam / np.max(cam)
        heatmap = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-10)#归一化处理
        # heatmap = (cam - np.mean(cam, axis=-1)) / (np.std(cam, axis=-1) + 1e-10)
        print(heatmap.shape)
        return heatmap
class GradCAM(BaseCAM):
    def __init__(self, model, target_layer, use_cuda=False):
        super(GradCAM, self).__init__(model, target_layer, use_cuda)

    def get_cam_weights(self, input_tensor,
                        target_category,
                        activations,
                        grads):
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_activations = np.sum(activations, axis=1)
        eps = 0.000001
        aij = grads_power_2 / (2 * grads_power_2 + sum_activations[:, None] * grads_power_3 + eps)
        aij = np.where(grads != 0, aij, 0)

        weights = np.maximum(grads, 0) * aij
        weights = np.sum(weights, axis=1)
        return weights

def multicolored_lines(x, y, heatmap, title_name):
    fig, ax = plt.subplots()
    # print(x.shape, y.shape, heatmap.shape)
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
# from pytorch_grad_cam.utils.image import preprocess_image
# model = Net1()
# model.load_state_dict(torch.load('./data7/G0503_02.pt'))   #Load your own pretrained model
def plotting(model, target_layer, input_tensor, N):
    # target_layer = model.conv1
    net = GradCAM(model, dict(model.named_modules())[target_layer])
    # from settest import Test
    # from scipy.fftpack import fft
    # input_tensor = Test.Data[0:1, :]
    input_tensor = input_tensor.unsqueeze(1)
    plt.figure(figsize=(5, 1))
    output = net(input_tensor)
    # print(output.shape)
    x = np.linspace(0, N, input_tensor.shape[2])
    plt.style.use("seaborn-v0_8-whitegrid")
    multicolored_lines(x, np.array(input_tensor.squeeze()), output, f"GradCAM++ Visualization")
# import scipy.io as scio
# input_tensor = input_tensor.numpy().squeeze()
# dataNew = "G:\\datanew.mat"
# scio.savemat(dataNew, mdict={'cam': output, 'data': input_tensor})