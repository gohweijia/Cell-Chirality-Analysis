import glob
import os
from ruamel.yaml import YAML

class Network_Config(object):
    def __init__(self, model_dir = None, config_filepath = None, **kwargs):
        """Creates Network_Config object that contains the network parameters and functions needed to manipulate these parameters.
    
        Parameters
        ----------
        model_dir : `str`, optional
            [Default: None] Folder where the model is to be saved/read from
        config_filepath : `str`, optional
            [Default: None] Filepath to the config file that will be loaded
        **kwargs
            For network parameters that are to be changed from the loaded config file

        Attributes
        ----------
        yaml : :class:`ruamel.yaml.YAML`
            YAML class with function needed to read/write YAML files 
        config : `dict`
            Dictionary containing the config parameters
        """
        self.yaml=YAML()
        
        # load config file from model_dir
        if config_filepath is not None:
            
            self.config = self.load_config_from_file(config_filepath)
            print("Loaded config file from {}".format(config_filepath))
        elif model_dir is not None:
            try:
                self.config = self.load_config_from_model_dir(model_dir)
                print("Loaded config file from {}".format(model_dir))
            except:
                print("Please ensure that config_filepath is set or there is a config file in model_dir")
                raise
            
        if model_dir is not None:
            # update model_dir in config
            print("Updating model_dir to {}".format(model_dir))
            self.update_parameter(["general", "model_dir"], model_dir)
        
        # overwrite network parameters with parameters given during initialization
        for key, value in kwargs.items():
            self.update_parameter(self.find_key(key), value)
            
        # perform calculations
        self.update_parameter(["model", "input_size"], self.get_parameter("tile_size") + [self.get_parameter("image_channel"),])
        self.update_parameter(["model", "batch_size"], self.get_parameter("batch_size_per_GPU")) # * self.gpu_count
                  
    ######################
    # Accessors/Mutators
    ######################
    def get_parameter(self, parameter, config = []):
        """Output the value from the config file using the given key

        Parameters
        ----------
        parameter : `list` or `str`
            Key or list of keys used to find for the value in the config file
        
        config : `list`, optional
            Used to iterate through nested dictionaries. Required to recursively iterate through neseted dictionary
            
        Returns
        ----------
        value : `str` or `int` or `list`
            Value obtained from the specified key
            
        See Also
        ----------
        find_key : Function to identify the list of keys to address the correct item in a nested dictionary
        """
        assert isinstance(parameter, (list, str))
        
        # find for key in nested dictionary
        if isinstance(parameter, str):
            parameter = self.find_key(parameter)
        
        if config == []:
            config = self.config
        if config is None:
            return None
        
        if not parameter:
            return config
        
        return self.get_parameter(parameter[1:], config = config.get(parameter[0]))

    def update_parameter(self, parameter, value, config = None):
        """Updates the parameter in the config file using a full addressed list

        Parameters
        ----------
        parameter : `list`
            List of keys that point to the correct item in the nested dictionary
            
        value : `str` or `int` or `list`
            Value that is updated in the nested dictionary
            
        config : `list` or `none`, optional
            Used to iterate through nested dictionaries
            
        Returns
        ----------
        TODO
        """
        assert type(parameter) is list
                
        if config == None:
            config = self.config
        
        if len(parameter) == 1:
            config.update({parameter[0]: value})
            return config
        return self.update_parameter(parameter[1:], value, config = self.config.get(parameter[0]))

    def find_key(self, key, config = None):
        """Find the list of keys to address the correct item in a nested dictionary

        Parameters
        ----------
        key : `str`
            Key that needs to be correctly addressed in a nested dictionary
            
        config : `list` or `none`, optional
            Used to iterate through nested dictionaries
            
        Returns
        ----------
        key : `list`
            Address of the key in the nested dictionary
        """
        
        if config == None:
            config = self.config
            
        key_path = []
        for k, v in config.items():
            if k == key:
                return [k]
            elif isinstance(v, dict):
                found_key = self.find_key(key, config = v)
                if found_key is not None:
                    return [k] + found_key
    
    ######################
    # Config IO options
    ######################
    def load_config_from_file(self, file_path):
        """Load parameters from yaml file

        Parameters
        ----------
        file_path : `str`
            Path of config file to load
            
        Returns
        ----------
        config : `dict`
            Dictionary containing the config parameters
        """

        with open(file_path, 'r') as input_file: 
            config = self.yaml.load(input_file)
            input_file.close()

        return config
    
    def load_config_from_model_dir(self, model_dir):
        """Finds for a config file from the model directory and loads it
    
        Parameters
        ----------
        model_dir : `str`
            Folder to search for and load the config file

        Returns
        ----------
        config : `dict`
            Dictionary containing the config parameters
            
        Raises
        ------
        IndexError
            If there are no config file in the model_dir
        """
        
        # check if yaml file exists in model_dir
        try:
            list_config_files = glob.glob(os.path.join(model_dir,'*config.yml'))
            if len(list_config_files) > 1:
                print("Multiple config files found. Loading {}".format(list_config_files[0]))
            else:
                print("Config file exists in model directory. Loading {}".format(list_config_files[0]))
            return self.load_config_from_file(list_config_files[0])
        except IndexError:
            print("No config file found in model_dir.")
            raise

    def write_config(self, file_path):
        """Writes parameters to yaml file

        Parameters
        ----------
        file_path : `str`
            Path of config file to write to
        """
        
        with open(file_path, 'w') as output_file:  
            self.yaml.dump(self.config, output_file)

        output_file.close()
        
        print("Config file written to: {}".format(file_path))
    
    def write_model(self, model, file_path):
        """Writes parameters to yaml file

        Parameters
        ----------
        model : :class:`Keras.model`
            Keras model that will be parsed and written to a yaml file
        
        file_path : `str`
            Path of model file to write to
        """
        
        with open(file_path, 'w') as output_file:  
            output_file.write(model.to_yaml())

        output_file.close()
        
        print("Model file written to: {}".format(file_path))