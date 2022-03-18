from PIL import Image
import numpy as np
import networkx as nx
import os

from sliding_window import sliding_window

class image_feature_descriptors:
    def __init__(self, im, epsilon):
        '''
        Create a collection of feature descriptors of im, each with size and width
        epsilon.

        Parameters:
            im - a pillow image
            epsilon - an int, the window size
        '''
        # Pad the image so each pixel gets a descriptor.
        front_pad = int(epsilon / 2)
        back_pad = int((epsilon - 1) / 2)
        ar = np.pad(np.asarray(im, np.float32), ((front_pad, back_pad), (front_pad, back_pad)))

        # Create the descriptors using a sliding window.
        self.descriptors = sliding_window(ar, (epsilon, epsilon), (1, 1), False)


## COST FUNCTIONS ##
'''
    Return the sums of absolute differences of fd1 and fds2.

    Parameters:
        fd1 - an n-by-n array
        fds2 - an m-by-n-by-n array
    
    Returns:
        an m-array, in each position i containing the sum of
        squared differences of fds2[i] and fd1
'''
def sums_of_absolute_differences(fd1, fds2):
    return np.sum(np.absolute(np.subtract(fds2, fd1)), (1, 2))


'''
    Return the sums of squared differences of fd1 and fds2.

    Parameters:
        fd1 - an n-by-n array
        fds2 - an m-by-n-by-n array
    
    Returns:
        an m-array, in each position i containing the sum of
        squared differences of fds2[i] and fd1
'''
def sums_of_squared_differences(fd1, fds2):
    return np.sum(np.power(np.subtract(fds2, fd1), 2), (1, 2))


'''
    Return the cross correlations of fd1 and fds2.

    Parameters:
        fd1 - an n-by-n array
        fds2 - an m-by-n-by-n array
    
    Returns:
        an m-array, in each position i containing the cross
        correlation of fds2[i] and fd1
'''
def cross_correlations(fd1, fds2):
    cc = np.sum(np.multiply(fds2, fd1), (1, 2))
    fd1_energy = np.sum(np.power(fd1, 2))
    fds2_energy = np.sum(np.power(fds2, 2), (1, 2))
    energy = np.sqrt(np.multiply(fds2_energy, fd1_energy))
    norm_cc = np.divide(cc, energy)
    return norm_cc


'''
    Return the Potts model of alpha and beta.

    Parameters:
        alpha - an int
        Beta - an n-by-n int array
    
    Returns:
        an int
'''
def potts_model(alpha, Beta):
    return np.array(np.where(Beta == alpha, 0, 10))


## OPTIMIZERS ##
class window_optimizer:
    '''
        Create window-based disparity optimizer.

        Parameters:
            cost_function - a function taking a fd and an
                            array of fds
            argoptim_function - a function taking an array
                                and returning an int, the
                                index of the optimal value
            ndisp - an int, the maximum number of disparity
                    labels, > 1
            im1_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      left image
            im2_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      right image
    '''
    def __init__(self, cost_function, argoptim_function,\
                 ndisp, im1_fds, im2_fds):
        self.cost_function = cost_function
        self.argoptim_function = argoptim_function
        self.ndisp = ndisp
        self.im1_fds = im1_fds
        self.im2_fds = im2_fds
    
    '''
        Return the disparity of the pixel at position (x, y)
        in image 1.

        Parameters:
            x - an int
            y - an int
        
        Returns:
            an int, the disparity of the pixel at position (x, y)
            in image 1
    '''
    def optimize(self, x, y):
        x_left = max(0, x - self.ndisp + 1)
        disp_scores = self.cost_function(\
            self.im1_fds.descriptors[y, x],\
            self.im2_fds.descriptors[y, x_left:x+1])
        best_score_idx = self.argoptim_function(disp_scores)
        return disp_scores.shape[0] - 1 - best_score_idx


def window_sum_of_absolute_differences_optimizer(ndisp, im1_fds, im2_fds):
    return window_optimizer(sums_of_absolute_differences, np.argmin, ndisp, im1_fds, im2_fds)


def window_sum_of_squared_differences_optimizer(ndisp, im1_fds, im2_fds):
    return window_optimizer(sums_of_squared_differences, np.argmin, ndisp, im1_fds, im2_fds)


def window_cross_correlation_optimizer(ndisp, im1_fds, im2_fds):
    return window_optimizer(cross_correlations, np.argmax, ndisp, im1_fds, im2_fds)


class scan_graph_optimizer:
    '''
        Create scan line graph-based disparity optimizer.

        Parameters:
            cost_function - a function taking a fd and an
                            array of fds
            cost_processing_function - a function taking an
                                       n-array and returning
                                       an n-array, turning
                                       costs into graph
                                       weights
            ndisp - an int, the maximum number of disparity
                    labels, > 1
            im1_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      left image
            im2_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      right image
    '''
    def __init__(self, cost_function, cost_processing_function,\
                 ndisp, im1_fds, im2_fds):
        self.cost_function = cost_function
        self.cost_processing_function = cost_processing_function
        self.ndisp = ndisp
        self.im1_fds = im1_fds
        self.im2_fds = im2_fds
    
    '''
        Return the disparity of the pixel at position (x, y)
        in image 1.

        Parameters:
            x - an int
            y - an int
        
        Returns:
            an int, the disparity of the pixel at position (x, y)
            in image 1
    '''
    def optimize(self, y):
        # Create the scan line graph
        G = nx.DiGraph()

        # Loop through x in image 1, finding the disparity scores,
        # adding edges from each previous layer node to each current
        # layer node whose associated pixel in image 2 is not to the
        # left of its associated pixel in image 2.
        width = self.im1_fds.descriptors.shape[1]
        for x in range(width):
            x_left = max(0, x - self.ndisp + 1)
            disp_scores = self.cost_processing_function(\
                np.flip(self.cost_function(\
                    self.im1_fds.descriptors[y, x],\
                    self.im2_fds.descriptors[y, x_left:x+1])))
            num_scores = x - x_left + 1
            
            if x == 0:
                # Add edges from "start".
                for i in range(num_scores):
                    G.add_edge("start",\
                               str(x) + '-' + str(i),\
                               weight=disp_scores[i])
            else:
                # Add edges from the previous layer.
                x_left_prev = max(0, x - self.ndisp)
                num_scores_prev = x - x_left_prev
                for i in range(num_scores_prev):
                    for j in range(min(i + 2, num_scores)):
                        G.add_edge(str(x - 1) + '-' + str(i),\
                                str(x) + '-' + str(j),\
                                weight=disp_scores[j])
        
        # Add edges from the last layer to "end".
        for i in range(num_scores):
            G.add_edge(str(x) + '-' + str(i), "end", weight=0)
        
        # Solve for a min-cost path through G.
        path = nx.shortest_path(G, "start", "end", "weight")

        assert len(path) == width + 2

        # Fill the return array of disparities.
        disps = np.empty(width)
        for i in range(width):
            disp = int(path[i + 1].split('-')[1])
            disps[i] = disp
        
        return disps


def scan_graph_sum_of_absolute_differences_optimizer(ndisp, im1_fds, im2_fds):
    return scan_graph_optimizer(sums_of_absolute_differences, lambda a : a, ndisp, im1_fds, im2_fds)


def scan_graph_sum_of_squared_differences_optimizer(ndisp, im1_fds, im2_fds):
    return scan_graph_optimizer(sums_of_squared_differences, lambda a : a, ndisp, im1_fds, im2_fds)


def scan_graph_cross_correlation_optimizer(ndisp, im1_fds, im2_fds):
    return scan_graph_optimizer(cross_correlations, lambda a : -a + np.max(a), ndisp, im1_fds, im2_fds)


class swap_graph_cut_optimizer:
    '''
        Create scan line graph-based disparity optimizer.

        Parameters:
            data_function - a function taking an fd and an array of fds and
                            returning a number
            smoothness_function - a function taking an int and an n-by-n array and
                                  returning an n-by-n array of smoothness costs
            im1_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      left image
            im2_fds - an image_feature_descriptors instance
                      containing feature descriptors of the
                      right image
            ndisp - an int, the maximum number of disparity
                    labels, > 1
    '''
    def __init__(self, data_function, smoothness_function,\
                 im1_fds, im2_fds, ndisp):
        self.data_function = data_function
        self.smoothness_function = smoothness_function
        self.im1_fds = im1_fds
        self.im2_fds = im2_fds
        self.ndisp = ndisp
        self.height, self.width = im1_fds.descriptors.shape[0:2]

        # Start with random disparities.
        rng = np.random.default_rng()
        self.disparities = rng.integers(ndisp, size=(self.height, self.width), dtype=np.int32)
        # Force into image on left.
        self.disparities[:, 0:ndisp] = np.minimum(self.disparities[:, 0:ndisp], np.array(range(ndisp)))
        self.padded_disparities = np.pad(self.disparities, ((1, 1), (1, 1)))
    
    def E(self):
        cross = [[1, 1], [1, 0]]

        E_data = 0
        E_smooth = 0
        for y in range(self.height):
            for x in range(self.width):
                x2 = x + 1
                y2 = y + 1
                d = self.disparities[y, x]
                E_data += self.D(x, y, d)
                E_smooth += np.sum(np.multiply(self.V(d, x, y, x2, y2), cross))
        
        return E_data + E_smooth

    
    def D(self, x, y, d):
        return self.data_function(self.im1_fds.descriptors[y, x], self.im2_fds.descriptors[y, x - d:x - d + 1])
    
    def V(self, d, x1, y1, x2, y2):
        return self.smoothness_function(d, self.padded_disparities[y1 + 1:y2 + 2, x1 + 1:x2 + 2])
    
    '''
        Run an epoch of the optimization algorithm.
        
        Returns:
            a Boolean, whether energy was decreased in the epoch
    '''
    def optimize(self):
        success = False

        # Loop through disparity combinations.
        for alpha in range(self.ndisp):
            for beta in range(self.ndisp):
                if alpha != beta:
                    print("alpha: " + str(alpha) + ", beta: " + str(beta))

                    gamma = max(alpha, beta)

                    # Create the swap graph.
                    G = nx.Graph()

                    # Get the pixels with nodes.
                    indices = np.array(np.nonzero(
                        (self.disparities == alpha) | (self.disparities == beta))).T
                    indices = indices[np.nonzero(indices[:, 1] >= gamma)]

                    cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
                    
                    # Add edges from "alpha" and "beta" nodes to pixels and
                    # between graph pixels.
                    for j, i in indices:
                        alpha_data = self.D(i, j, alpha)
                        beta_data = self.D(i, j, beta)
                        x1 = i - 1
                        x2 = i + 1
                        y1 = j - 1
                        y2 = j + 1
                        mask = self.padded_disparities[y1 + 1:y2 + 2, x1 + 1:x2 + 2]
                        mask = np.multiply(np.where((mask != alpha) & (mask != beta), 1, 0), cross)
                        smoothness = self.V(self.disparities[j, i], x1, y1, x2, y2)
                        alpha_smoothness = np.sum(smoothness * mask)
                        beta_smoothness = np.sum(smoothness * mask)
                        alpha_weight = alpha_data + alpha_smoothness
                        beta_weight = beta_data + beta_smoothness

                        G.add_edge('alpha', str(i) + '-' + str(j), weight=alpha_weight)
                        G.add_edge('beta', str(i) + '-' + str(j), weight=beta_weight)

                        if x2 != self.width and mask[1, 2] == 0:
                            G.add_edge(str(i) + '-' + str(j), str(i + 1) + '-' + str(j), weight=smoothness[1, 2])
                        if y2 != self.height and mask[2, 1] == 0:
                            G.add_edge(str(i) + '-' + str(j), str(i) + '-' + str(j + 1), weight=smoothness[2, 1])
                    
                    # Find the min cut of G and update disparities.
                    cut_value, partition = nx.minimum_cut(G, 'alpha', 'beta', 'weight')
                    alpha_nodes, beta_nodes = partition

                    old_disparities = self.disparities
                    new_disparities = np.copy(self.disparities)

                    for node in alpha_nodes:
                        if node != 'alpha':
                            i, j = [int(k) for k in node.split('-')]
                            new_disparities[j, i] = beta

                    for node in beta_nodes:
                        if node != 'beta':
                            i, j = [int(k) for k in node.split('-')]
                            new_disparities[j, i] = alpha

                    old_E = self.E()
                    self.disparities = new_disparities
                    self.padded_disparities = np.pad(self.disparities, ((1, 1), (1, 1)))
                    new_E = self.E()
                    if new_E > old_E:
                        self.disparities = old_disparities
                        self.padded_disparities = np.pad(self.disparities, ((1, 1), (1, 1)))
                        print('New energy %f, higher than old energy %f. Keeping old disparities.' % (new_E, old_E))
                    else:
                        success = True
                        print('New energy %f, lower than old energy %f. Changing to new disparities.' % (new_E, old_E))
        
        return success


## SOLVERS ##
class window_stereo_solver:
    '''
        Create a window-based stereo solver for im1 and im2,
        images with parallel projection planes.

        Parameters:
            im1 - a pillow image, the left image
            im2 - a pillow image, the right image
            ndisp - an int, the maximum number of disparity
                    labels, > 1
            epsilon - an int, the window size
            optimizer - an optimizer class for determining the
                        disparity of each pixel in im1
    '''
    def __init__(self, im1, im2, ndisp, epsilon, optimizer):
        im1 = im1.convert("L")
        im2 = im2.convert("L")

        self.disparities = np.empty((im1.height, im1.width))

        im1_fds = image_feature_descriptors(im1, epsilon)
        im2_fds = image_feature_descriptors(im2, epsilon)
        optim = optimizer(ndisp, im1_fds, im2_fds)
        for y in range(im1.height):
            for x in range(im1.width):
                # Get the disparity using the optimizer.
                disp = optim.optimize(x, y)
                self.disparities[y, x] = disp

        # Create the disparity image from the disparity matrix.
        self.disparity_im = Image.fromarray(np.uint8(np.multiply(self.disparities, 255 / (ndisp - 1))))


class scan_graph_stereo_solver:
    '''
        Create a scan line graph-based stereo solver for im1 and im2,
        images with parallel projection planes.

        Parameters:
            im1 - a pillow image, the left image
            im2 - a pillow image, the right image
            ndisp - an int, the maximum number of disparity
                    labels, > 1
            epsilon - an int, the window size
            optimizer - an optimizer class for determining the
                        disparity of each pixel in im1
    '''
    def __init__(self, im1, im2, ndisp, epsilon, optimizer):
        im1 = im1.convert("L")
        im2 = im2.convert("L")

        self.disparities = np.empty((im1.height, im1.width))

        im1_fds = image_feature_descriptors(im1, epsilon)
        im2_fds = image_feature_descriptors(im2, epsilon)
        optim = optimizer(ndisp, im1_fds, im2_fds)
        for y in range(im1.height):
            # Get the row disparity using the optimizer.
            disp = optim.optimize(y)
            self.disparities[y] = disp

        # Create the disparity image from the disparity matrix.
        self.disparity_im = Image.fromarray(np.uint8(np.multiply(self.disparities, 255 / (ndisp - 1))))


class swap_graph_cut_stereo_solver:
    '''
        Create a swap graph cut-based stereo solver for im1 and im2,
        images with parallel projection planes.

        Parameters:
            im1 - a pillow image, the left image
            im2 - a pillow image, the right image
            ndisp - an int, the maximum number of disparity
                    labels, > 1
            epsilon - an int, the window size
            epochs - an int, the number of epochs to run
    '''
    def __init__(self, im1, im2, ndisp, epsilon, epochs):
        im1 = im1.convert("L")
        im2 = im2.convert("L")

        im1_fds = image_feature_descriptors(im1, epsilon)
        im2_fds = image_feature_descriptors(im2, epsilon)

        optimizer = swap_graph_cut_optimizer(sums_of_squared_differences, potts_model, im1_fds, im2_fds, ndisp)

        # Run for epochs or until local minimum found.
        success = True
        for i in range(epochs):
            if success:
                print("epoch " + str(i))
                success = optimizer.optimize()

        self.disparities = optimizer.disparities

        # Create the disparity image from the disparity matrix.
        self.disparity_im = Image.fromarray(np.uint8(np.multiply(self.disparities, 255 / (ndisp - 1))))
