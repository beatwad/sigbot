import json
from optimizer import Optimizer


class ConfigUpdater:
    """
    Class responsible for updating configuration files.

    Parameters
    ----------
    ttype : str
        Type of configuration.
    timeframe : str
        Timeframe for the configuration.
    """
    def __init__(self, ttype: str, timeframe: str):
        """
        Initialize the ConfigUpdater with a trade type and timeframe.

        Parameters
        ----------
        ttype : str
            Type of trade ('buy', 'sell').
        timeframe : str
            Timeframe value.
        """
        self.ttype = ttype
        self.timeframe = timeframe

    @staticmethod
    def dict_convert(optim_dict: dict) -> dict:
        """
        Convert a nested configuration dictionary. Replace lists with their first element.

        Parameters
        ----------
        optim_dict : dict
            The dictionary to convert.

        Returns
        -------
        dict
            The converted dictionary.
        """
        def helper(d: dict) -> dict:
            """
            Helper function to recursively convert nested configuration dictionaries.

            Parameters
            ----------
            d : dict
                The dictionary being processed.

            Returns
            -------
            dict
                The updated dictionary.
            """
            for key, value in d.items():
                if type(value) == list:
                    d[key] = value[0]
                else:
                    d[key] = helper(value)
            return d

        return helper(optim_dict)

    def config_update(self, optim_dict: dict) -> None:
        """
        Update the configuration file based on the optimizer dictionary.

        This method loads the existing configuration file, updates it using
        the Optimizer class, and saves the updated configuration.

        Parameters
        ----------
        optim_dict : dict
            Dictionary containing the optimizer configuration.
        """
        # Load existing configuration from the specified JSON file
        with open(f'../config/config_{self.timeframe}.json', 'r') as f:
            configs = json.load(f)

        # Convert the optimizer dictionary to a suitable format
        optim_dict = self.dict_convert(optim_dict)

        # Create an Optimizer instance with the converted dictionary and configs
        opt = Optimizer([], optim_dict, **configs)

        # Save the updated configuration using the optimizer
        confs = opt.save_configs(optim_dict, self.ttype)

        # Write the updated configuration back to the file
        with open(f'../config/config_{self.timeframe}.json', 'w') as f:
            json.dump(confs, f)

