import matplotlib.pyplot as plt
from itertools import product as cartesian_product
from sklearn.model_selection import train_test_split
import pandas as pd
import sympy as sm
import numpy as np
import os

class Perceptron:
    """
    Class that represents a multi layer perceptron
    ...

    Atributos
    ----------
    X_train : np.array
        Numpy matrix (N x K) of the training data, N is the number of datapoints and K is the number of features
    Y_train : N x M
        Numpy matrix (N x M) of the outputs of the training data, N is the number of datapoints and M is the number of outputs
    X_validation : [int]
        Numpy matrix (Nv x K) of the training data, Nv is the number of datapoints and K is the number of features
    Y_validation : [int]
        Numpy matrix (Nv x M) of the outputs of the validation data, Nv is the number of datapoints and M is the number of outputs
    X_test: [int]
        Numpy matrix (Nt x K) of the training data, Nt is the number of datapoints and K is the number of features
    hl : int
        Number of hidden layers
    hn : int
        Number of neurons per hidden layer
    """
    
    def __init__(self, X_train, Y_train, X_validation, Y_validation, X_test, hl, hn):

        # initialize parameters receibed

        self.hl = hl
        self.hn = hn

        self.X_validation = X_validation
        self.Y_validation = Y_validation
        self.X_test = X_test

        self.X_train = X_train
        self.Y_train = Y_train
        self.Y_dtrain = sm.Matrix(Y_train).T
        
        self.n_dim = X_train.shape[1]
        self.N = X_train.shape[0]
        self.inputs = self.n_dim
        self.M = self.Y_dtrain.shape[0]
        self.Yd_train = np.reshape(Y_train,(self.N,self.M))
        self.model_name = f"MLP hl = {hl} hn= {hn}"

        # Iniitialize neuron and layer numbers
        self.neurons_input = self.inputs
        self.layers_hidden = hl
        self.neurons_hidden = [hn] * hl
        self.outputs = self.M
        self.weights_dimensions = [self.inputs ,self.neurons_input] + self.neurons_hidden + [self.outputs]
        self.neurons_numbers = [self.neurons_input] + self.neurons_hidden + [self.outputs]
        self.n_layers = self.layers_hidden + 2

        # Compute the number of weights per layer and the total number of weights in the network

        num_weights_layer = []
        for l in range(self.n_layers):
            num_weights_layer.append(self.weights_dimensions[l]*self.weights_dimensions[l+1])
        self.num_weights_layer = num_weights_layer
        self.num_weights = sum(num_weights_layer)

        # Name each layer to label plots

        self.name_layers()

        # Initialize lists to save results

        self.validation_errors =[]
        self.validation_avg_energy_errors=[]
        self.validation_local_gradients=[]
        self.validation_delta_ks =[]


    def name_layers(self):
        """
        Auxiliar function that initializes a list with names for the layers
        """
        self.layer_names = []
        for l in range(self.n_layers):
            type_layer = ''
            if l == 0:
                type_layer = 'input'
            elif l == self.n_layers-1:
                type_layer = 'output'
            else:
                type_layer = 'hidden'
            l_name = f"{type_layer} layer"
            if type_layer == 'hidden': 
                l_name = l_name + str(l)
            self.layer_names.append(l_name)

    def get_energy(self,e_vector):
        """Function that returns the instantaneous energy error

        Parameters
        ----------
        e_vector : numpy ndarray
            Error matrix 
        
        Returns
        ----------
        int: intantaneous energy error
            
        """
        return 0.5 * (sum(e_vector**2))

    def forward(self, x, weights):
        """ Function that performs a forward step.
        It calculates the local field and the activation value and propagates forward

        Parameters
        ----------
        X : numpy ndarray
            Input matrix
        weights: numpy ndarray
            Weights matrix 

        Returns
        ----------
        Y: numpy ndarray of outputs of the last layer
        Yi: numpy ndarray of outputs of each layer
        Vi: numpy ndarray of local fields of each layer
        impulses: numpy ndarray of inputs of each layer
        """   
        Vi = []
        phi_i_v_i = x
        Yi = []
        for i in range(self.n_layers):
            wi = weights[i]
            vi = np.dot(phi_i_v_i,wi)
            Vi.append(vi)
            phi_i_v_i = self.num_phi(vi,i)
            Yi.append(phi_i_v_i)
        Y = Yi[-1]
        impulses = [x] + Yi
        return Y, Vi, Yi, impulses
    

    def num_phi(self, x, layer):
        """ Activation function
        It returns the layer's activation function applied to a specific value
        Parameters
        ----------
        X : numpy ndarray
            argument to aply the activation function to
        layer: int
            number of layer
        Returns
        ----------
        The layer's activation function applied to x
        """   
        if layer == 0:
            return 1 / (1 + np.exp(-x))
        elif layer == self.n_layers -1:
            return 1 / (1 + np.exp(-x))
        else:
            return 1 / (1 + np.exp(-x))


    def num_dphi(self, x, layer):
        """ Derivative of activation function
        It returns the derivative of the layer's activation function applied to a specific value

        Parameters
        ----------
        X : numpy ndarray
            argument to aply the derivative of the activation function to
        layer: int
            number of layer
        
        Returns
        ----------
        The derivative of the layer's activation function applied to x
        """   
        if layer == 0:
            return x * (1 - x)
        elif layer == self.n_layers -1:
            return x * (1 - x)
        else:
            return x * (1 - x)



    def gradient_descent(self, initial_values,epochs, eta, tol = 0.01):
        """ Function that implements the gradient descent algorithm

        Parameters
        ----------
        initial values : [numpy ndarray]
            list of weight matrices (one per layer)
        epochs: int
            maximum epochs
        eta: float
            learning rate
        tol: float
            tolerance for stopping condition
        
        """   
        self.model_name = self.model_name + f' eta = {eta}'
        self.epochs = epochs
        
        assert len(initial_values) == self.n_layers, "not enough initial weight matrices were passed"
        energy_errors_av = []
        errors = []
        param_values = [np.zeros((self.epochs,self.num_weights_layer[l])) for l in range(self.n_layers)]
        dif_values = [np.zeros((self.epochs,self.num_weights_layer[l])) for l in range(self.n_layers)]
        local_grad_values = [np.zeros((self.epochs, 1)) for i in range(self.n_layers)]
        mean_delta_k_output = []
    
        error_it = 10000
        it = 0
        W = initial_values

        while it < epochs and error_it > tol:
            # forward
            Y, Vi, Yi, impulses = self.forward(self.X_train,W)
            local_gradients_it = []
            # batch back propagation starting from the output layer
            for layer in range(self.n_layers-1,-1,-1): 
                if layer == self.n_layers -1:
                    d = self.Yd_train
                    error = d-Y
                    energy = self.get_energy(error)
                    avg_energy_error_it = np.mean(energy)
                    errors = np.append(errors,error)
                    energy_errors_av = np.append(energy_errors_av,avg_energy_error_it)
                else:
                    wi = W[layer+1]
                    error = np.dot(delta_k,wi.T)
                dphi_vi = self.num_dphi(Yi[layer], layer)
                delta_k = error * dphi_vi
                local_gradients_it.append(delta_k)
                if layer == self.n_layers - 1:
                    mean_delta_k_output = np.append(mean_delta_k_output,np.mean(delta_k))
                local_grad_values[layer][it] = np.mean(delta_k)
                   
            # update_weights
            for layer in range(self.n_layers):
                index = self.n_layers - 1-  layer
                impulse = impulses[layer]
                delta_k  = local_gradients_it[index]
                
                dJdw = impulse.T.dot(delta_k)
                W[layer] = W[layer] + eta*dJdw
                param_values[layer][it,:] = W[layer].flatten().tolist()

            self.validation(W) 

            it+=1
        

        self.param_values = param_values
        self.dif_values = dif_values
        self.local_gradients_array = local_grad_values

        self.errors = errors
        self.avg_energy_errors_training = energy_errors_av
        self.training_weights = W
        self.mean_delta_k_output = mean_delta_k_output

    def validation(self, W):
        """ Function that finds the predicted output on the validation set,
        it also finds one gradient to test the behaviour on the validation set.
        Training does not occur here, the model does not learn from the validation set

        Parameters
        ----------
        W : [numpy ndarray]
            weight matrix 
        """   
        N_validation = self.Y_validation.shape[0]
        Y_validation = np.reshape(self.Y_validation,(N_validation,self.M))
        #foward
        Y, Vi, Yi, impulses = self.forward(self.X_validation,W)
        #backward to find error and local gradient
        layer = self.n_layers - 1
        validation_error = Y_validation -Y 
        instantaneous_energy = 0.5 * (sum(validation_error**2))
        av_error = np.mean(instantaneous_energy)
        delta_k = validation_error * self.num_dphi(Y,layer)
        mean_delta_k = np.mean(delta_k)

        self.validation_errors = np.append(self.validation_errors,validation_error)
        self.validation_avg_energy_errors = np.append(self.validation_avg_energy_errors, av_error)
        self.validation_local_gradients = np.append(self.validation_local_gradients, delta_k)
        self.validation_delta_ks = np.append(self.validation_delta_ks, mean_delta_k)

    def test(self, X_test):
        """ Fucntion that finds the predicted output on the test set
        Parameters
        ----------
        X_test : [numpy ndarray]
            Matrix of testing data
        """   
        Y, _,_,_ = self.forward(X_test,self.training_weights)
        self.Y_hat_test = Y

    def final_validate(self):
        """ Function that finds the predicted output on the validation set
        """   
        Y, _,_,_ = self.forward(self.X_validation,self.training_weights)
        self.y_hat_val = Y
        

    def graph(self):
        """ Auxiliary function to call other methods and generate graphs """  
        self.graph_errors()
        self.graph_gradients()
        
    def graph_gradients(self):
        """ Auxiliary function to graph local gradients """  
        assert len(self.local_gradients_array) == self.n_layers
        grads_array = np.hstack(self.local_gradients_array)
        df_grads = [pd.DataFrame(grads_array,columns=self.layer_names)]

        title = [r'Average local gradient $\delta_k$']
        filepath = f'Results/Training/Gradients {self.model_name}.jpg'
        x_label = ['epoch']
        y_label = [r'$\delta_k$']
        self.graph_dfs(df_grads, title,x_label, y_label,filepath)
        
    def graph_dfs(self, df_list,titles,x_labels, y_labels,filepath, size= (5,5)):
        """ Auxiliary function to graph multiple dataframes

        Parameters
        ----------
        df_list : [df]
            List of dataframes to plot
        titles: [str]
            list of titles
        x_labels: [str]
            list of labels along the x axis
        y_labels: [str]
            list of labels along the y axis
        filepath: str
            filepath to save the image
        size: (float,float)
            size of the image
        """  
        num_dfs = len(df_list)
        fig, ax = plt.subplots(num_dfs,1,figsize=size,sharex=True)
        if num_dfs == 1:
            df = df_list[0]
            df.plot(marker=".",rot=45,ax=ax,legend=True, title= titles[0])
            ax.set_xlabel(x_labels[0])
            ax.set_ylabel(y_labels[0])

            fig.tight_layout()
            plt.savefig(filepath,bbox_inches='tight')
            plt.close()
        else:
            for l in range(num_dfs):
                df = df_list[l]
                df.plot(marker=".",rot=45,ax=ax[l],legend=False, title= titles[l])
                ax[l].set_xlabel(x_labels[l])
                ax[l].set_ylabel(y_labels[l])
                lines0, labels0 = [sum(x, []) for x in zip(*[ax[l].get_legend_handles_labels()])]
                
            fig.tight_layout()
            plt.savefig(filepath)
            plt.close()

    def graph_errors(self):
        """ Auxiliary function to graph errors """  
        df_av_instantaneous_energy = [pd.DataFrame(self.avg_energy_errors_training,columns=['$\mathcal{E}_{av}$'])]
        title = ['Average Instantaneous energy error $\mathcal{E}_{av}$']
        filepath = f'Results/Training/Errors {self.model_name}.jpg'
        x_label = ['epoch']
        y_label = ['$\mathcal{E}_{av}$']
        self.graph_dfs(df_av_instantaneous_energy, title,x_label, y_label,filepath)
        
    def save_results(self, list_dics, max_epochs,eta):
        """ Auxiliary function to run training, validation and testing and 
        save relevant results of the model to a list of dictionaries
        
        Parameters
        ----------
        list_dics : [dict]
            List of dictionaries were the results of the model will be saved
        max_epochs: int
            maximum number of epochs
        eta: float
            learning rate
        
        """  
        self.initialize_gradient_descent(max_epochs,eta)
        self.test(self.X_test)
        self.final_validate()
        self.graph()
    
        list_dics[0][self.model_name] = self.avg_energy_errors_training
        list_dics[1][self.model_name] = self.validation_avg_energy_errors
        list_dics[2][self.model_name] = self.mean_delta_k_output
        list_dics[3][self.model_name] = self.validation_delta_ks
        list_dics[4][self.model_name] = self.Y_hat_test
        list_dics[5][self.model_name] = self.validation_avg_energy_errors[-1]
        list_dics[6][self.model_name] = self.num_weights
        list_dics[7][self.model_name] = self.y_hat_val
 


    def initialize_gradient_descent(self,epochs,eta):
        """ Auxiliary function to initialize weights in the hyercube [-1,1]
        and initialize gradient descent
        
        Parameters
        ----------
        list_dics : [dict]
            List of dictionaries were the results of the model will be saved
        max_epochs: int
            maximum number of epochs
        eta: float
            learning rate
        """  
        initial_values = []
        for i in range(self.n_layers):
            sizei = (self.weights_dimensions[i], self.weights_dimensions[i+1])
            wi = np.random.uniform(low=-1, high=1,size=sizei)
            assert self.num_weights_layer[i] == wi.shape[0]*wi.shape[1]
            initial_values.append(wi)
        self.gradient_descent(initial_values, epochs, eta)


class Comparison:
    """
    Class to compare different multi layer perceptrons
    ...

    Attributes
    ----------
    data : np.array
        Numpy array (N x M) of the entire data set
    Xindex : [int]
        List of indices of the columns of the features or independent variables
    Yindex : [int]
        List of indices of the columns of the outputs or dependent variables
    hl_max : float
        maximum number of hidden layers, we iterate from 1 to hl_max
    hn_max: float
        maximum number of neurons per hidden layer, we iterate from 1 to hn_max
    max_epochs : int
        maximum number of epochs for the training
    seed : int
        seed for the random number generator
    """
    
    def __init__(self,data, Xindex, Yindex, hl_max, hn_max, etas, max_epochs, seed = None):

        self.data = data
        self.Xindex = Xindex
        self.Yindex = Yindex


        self.hl_max = hl_max
        self.hn_max = hn_max
        self.etas = etas
        self.max_epochs = max_epochs

        if seed is None:
            self.seed = int(np.pi*10**9)

        

        self.create_paths(['Results', 'Results/Training', 'Results/Validation', 'Results/Test', 'Results/csv'])

        np.random.seed(self.seed)

        self.normalize()
        self.random_sample()

        self.global_args = (
            self.X_train,
            self.Y_train,
            self.X_validation,
            self.Y_validation,
            self.X_test
        )
        
        list_hn = list(range(1,hn_max+1))
        mlp_params_list = [etas,list_hn]
        self.mlp_params_combinations = list(cartesian_product(*mlp_params_list))


        self.results = {}
        self.save_models_results()
        self.plot_results()
        

    def normalize(self):
        """Function to normalize the data between 0 and 1"""
        self.norm_data = (self.data - np.min(self.data, axis=0)) / (np.max(self.data, axis=0) - np.min(self.data, axis=0))

    
    def random_sample(self):
        """Function to randomly sample the data and split the data set into training, validation and test sets"""
        indices = np.arange(self.norm_data.shape[0])
        (
            data_train,
            data_tv,
            self.indices_train,
            indices_tv,
        ) = train_test_split(self.norm_data, indices, test_size=0.4, random_state=self.seed)
        indices_in_tv = np.arange(data_tv.shape[0])
        (
            data_test,
            data_validation,
            indices_test_in_tv,
            indices_val_in_tv,
        ) = train_test_split(data_tv, indices_in_tv, test_size=0.5, random_state=self.seed)

        self.index_test = indices_tv[indices_test_in_tv]
        self.indices_val  = indices_tv[indices_val_in_tv] 

        self.X_train = data_train[:, self.Xindex]
        self.Y_train = data_train[:, self.Yindex]

        self.X_test  =  data_test[:, self.Xindex]
        self.Y_test  =  data_test[:, self.Yindex]
        
        self.X_validation = data_validation[:, self.Xindex]
        self.Y_validation = data_validation[:, self.Yindex]

        self.Y_test = self.Y_test.reshape(-1,1)

        self.Y_train_sorted = self.X_train[np.argsort(self.indices_train)]
        self.Y_train_sorted = self.Y_train[np.argsort(self.indices_train)]

        self.X_test_sorted = self.X_test[np.argsort(self.index_test)]
        self.Y_test_sorted = self.Y_test[np.argsort(self.index_test)]

        self.X_validation_sorted = self.X_validation[np.argsort(self.indices_val)]
        self.Y_validation_sorted = self.Y_validation[np.argsort(self.indices_val)]

    
    def save_models_results(self):
        """ Function to save the results of the models in a dictionary"""
        for hl in range(1, self.hl_max + 1):
            results_dics = [{} for i in range(8)]

            for (lr,hn) in self.mlp_params_combinations:
                args = self.global_args + (hl,hn)
                mlp = Perceptron(*args)
                mlp.save_results(results_dics, self.max_epochs,lr)
            self.results[str(hl)] = results_dics

    def plot_comparison(self,data,indexes, label_names, title, x_label, y_label, filepath, size, plot_pred = False, obs=None):
        """ Function to plot comparisons between models
        Parameters
        ----------
        data : dict
            Dictionary with the data to plot, the dictionary contains the name of the model as key and the predicted values as values
        indexes : [str]
            List of the names of the models to plot
        label_names : [str]
            List of the labels of the models to plot (best, avg, worst for example)
        title : str
            Title of the plot
        x_label : str
            Label of the x axis
        y_label : str
            Label of the y axis
        filepath : str
            Path to save the image of the plot
        size : tuple
            Size of the plot
        plot_pred : bool
            Boolean that signifies if the plot is of the predictions or not
        obs : np.array
            Array of the observed values, only used if plot_pred is True
        """
        fig, ax = plt.subplots(1,1,figsize=size,sharex=True)    
        if plot_pred:
            ax.plot(obs, ".-", label="Observations")
            print("plotting obs")
        for label_name, index in zip(label_names,indexes):
            label_i = f"{index} ({label_name})"
            ax.plot(data[index], ".-", label=label_i)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend(framealpha=1)
        # fig.tight_layout()
        fig.suptitle(title)
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
            
    def set_params(self,type_plot, stage, name):
        """ Function to set the parameters of the plot
        Parameters
        ----------
        type_plot : str
            Type of plot to make, can be 'error', 'local_grad' or 'pred'
        stage : str
            Stage of the data to plot, can be 'training', 'validation' or 'test'
        name : str
            name of the model
        Returns
        -------
            plot_title : str
                Title of the plot
            x_label : str
                Label of the x axis
            y_label : str
                Label of the y axis
            filepath : str
                Path to save the image of the plot
        """
        
        if type_plot == 'error':
            x_label = "epochs"
            y_label = r"$\mathcal{E}_{av}$"
            fig_title = 'Energy error'
            if name:
                plot_title = r"$\mathcal{E}_{av}$" + f" ({stage})"
        elif type_plot == 'local_grad':
            x_label = "epochs"
            y_label = r"$\delta_k$"
            fig_title = "Local output gradients"
            plot_title = r"Avg $\delta_k$" + f" ({stage})"
        else:
            x_label = "p"
            y_label = "Y"
            fig_title = "Pred vs obs"
            plot_title = f"Comparison of predicted and observed values"
        file_path = f"Results/{stage}/{fig_title} in {stage}" 
        if name is not None:
            hl = name[-1]
            plot_title = plot_title + f"with {hl} hidden layer(s)"
            file_path = file_path + f" hl={hl}.png"
        return plot_title, x_label, y_label, file_path

    
    def sort_by_value(self,dic):
        """ FUnciton to sort a dictionary by its values"""
        return dict(sorted(dic.items(), key=lambda x:x[1]))
    

    def rank_models(self,last_errors, n_weight_matrices):
        """ Function to rank the models by their simplicity and error
        Parameters
        ----------
        last_errors : dict
            Dictionary with the name of the model as key and the error of the last epoch as value
        n_weight_matrices : dict
            Dictionary with the name of the model as key and the number of weight matrices as value
        Returns
        -------
        best : str
            Name of the best model
        avg : str
           Name of the average model
        worst : str
            Name of the worst model
        """
        sorted_by_errors = self.sort_by_value(last_errors)
        sorted_params = {}
        for key in sorted_by_errors:
            sorted_params[key] = n_weight_matrices[key]
        sorted_simplicity = self.sort_by_value(sorted_params)
        ordered_models = list(sorted_simplicity.keys())
        indexes = [0,-len(ordered_models)//2-1,-1]
        best = ordered_models[indexes[0]]
        avg = ordered_models[indexes[1]]
        worst = ordered_models[indexes[2]]
        return best, avg, worst

    def plot_stage(self, data, plot_type, hl, size, indexes, label_names):
        """ Function that plots the results of the training and validation stages
        Parameters
        ----------
        data : [dict]
            List of dictionaries with the data to plot, the dictionary contains the name of the model as key and the predicted values as values
        plot_type : str
            Type of plot to make, can be 'error', 'local_grad' or 'pred'
        hl : int
            Number of hidden layers of the models to plot
        size : tuple
            Size of the plot
        indexes : [str]
            List of the names of the models to plot
        label_names : [str]
            List of the labels of the models to plot (best, avg, worst for example)
        """

        stages = ['training', 'validation']
        for i in range(len(data)):
            plot_title, x_label, y_label, filepath = self.set_params(plot_type, stages[i], hl)
            self.plot_comparison(data[i],indexes, label_names, plot_title, x_label, y_label, filepath, size)


    
    def plot_results(self):
        """Function that finds the best, average and worst model, plots the results of the predictions,
        saves the errors of the models and makes comparison plots between models with the same number of
        hidden layers
        """
        test_errors_pairs = []
        test_errors = []
        validation_errors = []
        val_errors = []
        val_preds = []
        test_preds = []

        for hl in self.results:
            predictions_models = self.results[hl][4]
            for key in predictions_models:
                pred_not_sorted = predictions_models[key]
                y_pred = pred_not_sorted[np.argsort(self.index_test)]
                test_error = np.mean(np.power(self.Y_test_sorted - y_pred,2))
                test_errors.append(test_error)
                test_errors_pairs.append((key, test_error))
                test_preds.append(y_pred)
            predictions_models_vals = self.results[hl][7]

            for key in predictions_models_vals:
                
                pred_not_sorted_val = predictions_models_vals[key]
                y_pred_val = pred_not_sorted_val[np.argsort(self.indices_val)]

                error_val = np.mean(np.power(self.Y_validation - y_pred_val,2))
                val_preds.append(y_pred_val)
                val_errors.append(error_val)
                validation_errors.append((key, error_val))
            
        self.plot_best(val_errors, val_preds, validation_errors, 'validation')
        self.plot_best(test_errors, test_preds, test_errors_pairs, 'test')


        test_errors
        df = pd.DataFrame(test_errors_pairs, columns=['model', 'test_error'])
        df.to_csv('Results/csv/test_errors.csv', index=True)
        df = pd.DataFrame(validation_errors, columns=['model', 'validation_error'])
        df.to_csv('Results/csv/validation_errors.csv', index=True)

        for hl in self.results:
            
            results_dics = self.results[hl]
            best,avg,worst = self.rank_models(results_dics[5],results_dics[6])

            label_names = ['best', 'avg', 'worst']
            indexes = [best, avg, worst]
            
            size = (6,5)
            
            graph_types = ['error', 'local_grad']
            for i in range(len(graph_types)):
                self.plot_stage(results_dics[2*i:2*(i+1)], graph_types[i], hl, size, indexes, label_names)
            plot_title, x_label, y_label, filepath = self.set_params('pred', 'Test', hl)
            self.plot_comparison(results_dics[4],indexes, label_names, plot_title, x_label, y_label, filepath, size, plot_pred = True, obs=self.Y_test)

    def plot_best(self, val_errors, val_preds, validation_errors, stage):
        "Function to plot the predictions of the best model"
        sort_val = np.argsort(val_errors)
        best_i = sort_val[0]
        avg_i = sort_val[len(sort_val)//2 -1]
        worst_i = sort_val[-1]

        best_model = val_preds[best_i]
        avg_model = val_preds[avg_i]
        worst_model = val_preds[worst_i]

        best_model_name = validation_errors[best_i][0]
        avg_model_name = validation_errors[avg_i][0]
        worst_model_name = validation_errors[worst_i][0]


        data = {
            best_model_name: best_model,
            avg_model_name: avg_model,
            worst_model_name: worst_model
        }
        indexes = [best_model_name,avg_model_name,worst_model_name]
        label_names = ['best', 'avg', 'worst']

        size = (7,7)

        plot_title, x_label, y_label, filepath = self.set_params('pred', stage, None)
        self.plot_comparison(data,indexes, label_names, plot_title, x_label, y_label, filepath, size, plot_pred = True, obs=self.Y_validation)

        data_best = {
            best_model_name: best_model
        }
        indexes = [best_model_name]
        label_names = ['best']
        plot_title = 'Predicted values od the best model vs real observations'
        x_label = 'P'
        y_label = 'Y'
        filepath = 'Results/Test/best pred.png'
        self.plot_comparison(data_best,indexes, label_names, plot_title, x_label, y_label, filepath, size, plot_pred = True, obs=self.Y_validation)
        


    def plot_validation(self, df_list,titles,x_labels, y_labels,filepath, size= (5,5)):
        """ Function to plot the results of the validation stage
        Parameters
        ----------
        df_list : [pd.DataFrame]
            List of dataframes with the data to plot
        titles : [str]
            List of the titles of the plots
        x_labels : [str]
            List of the labels of the x axis
        y_labels : [str]
            List of the labels of the y axis
        filepath : str
            Path to save the image of the plot
        size : tuple
            Size of the plot
        
        """
        num_dfs = len(df_list)
        fig, ax = plt.subplots(num_dfs,1,figsize=size,sharex=True)
        for l in range(num_dfs):
            df = df_list[l]
            df.plot(marker=".",rot=45,ax=ax[l],legend=False, title= titles[l])
            ax[l].set_xlabel(x_labels[l])
            ax[l].set_ylabel(y_labels[l])
            lines0, labels0 = [sum(x, []) for x in zip(*[ax[l].get_legend_handles_labels()])]
                
        fig.tight_layout()
        plt.savefig(filepath,bbox_inches='tight')
        plt.close()
        

    def create_paths(self, directories):
        """ Function to create directories to save the results"""
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)


def get_data():
    """ Function to get the data from the file data.txt
    Returns
    -------
    data : np.array
        Numpy array (N x M) of the entire data set
    Xindex : [int]
        List of indices of the columns of the features or independent variables
    Yindex : [int]
        List of indices of the columns of the outputs or dependent variables
    """
    data = np.loadtxt('Data/data.txt', delimiter=',')
    Xindex = [0,1,2,3]
    Yindex = [4]
    return data, Xindex, Yindex


if __name__ == "__main__":

    data, Xindex, Yindex = get_data()
    compare = Comparison(data, Xindex, Yindex ,3,5,[0.2,0.5,0.9],50)